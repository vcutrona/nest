import re
import string as string_utils
from typing import List
from typing import Tuple, Dict, Set

import numpy as np
from dateutil.parser import parse
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler

from data_model.generator import ScoredCandidate, CandidateEmbeddings


def chunk_list(list_, chunk_size):
    """
    Utility function to split a list into chunks of size chunk_size.
    The last chunk might be smaller than chunk_size.
    :param list_: the list to split
    :param chunk_size: chunk length
    :return: a generator of lists of size chunk_size
    """
    for i in range(0, len(list_), chunk_size):
        yield list_[i:i + chunk_size]


def strings_subsequences(strings: List[str], max_subseq_len) -> Tuple[Dict[str, List[str]], Set[str]]:
    """
    Given a list of strings, this method computes all the subsequences of different lengths, up to ``max_subseq_len``.
    Returns a tuple with a Dict(label: List(subsequences)) to preserve the mapping label-subsequence,
    and the set with all the subsequences.
    :param strings: a list of strings
    :param max_subseq_len: length of the longest subsequence to compute
    :return: a tuple (<subsequences_dict>, <subsequences_set>)
    """
    subsequences = {}
    subsequences_set = set()
    for string in strings:
        tokens = string.split()
        subsequences[string] = [" ".join(tokens[:i + 1])
                                for i in reversed(range(min(max_subseq_len, len(tokens))))]
        subsequences_set.update(subsequences[string])
    return subsequences, subsequences_set


def truncate_string(string, max_tokens) -> str:
    """
    Truncate a string after a certain number of tokens.
    :param string: the string to truncate
    :param max_tokens: number of desired tokens
    :return: the truncated string
    """
    return " ".join(string.split(" ")[:max_tokens]).strip()


def weighting_by_ranking(candidates: List[CandidateEmbeddings],
                         alpha=0.5,
                         default_score=np.nan) -> List[ScoredCandidate]:
    """
    Rank the candidates accordingly with the cosine distance between their vectors
    and their original ranks. If the default_score is provided, instances with one or more missing embeddings
    are assigned that score, np.nan otherwise.
    :param candidates: a list of CandidateEmbeddings
    :param alpha: a value in [0.0, 1.0], which represents the weight of the original rank component.
           1 - alpha is the weight of the cosine distance between vectors.
    :param default_score: default score >= 0.0 to assign to instances with missing embeddings
    :return: a list of ScoredCandidate ranked by score
    """

    if not candidates:
        return []

    assert np.isnan(default_score) or default_score >= 0.0
    assert 0.0 <= alpha <= 1.0

    distances = [cosine(c_emb.context_emb, c_emb.abstract_emb) for c_emb in candidates]
    if default_score >= 0.0:
        distances = np.nan_to_num(distances, nan=default_score)

    rank_scaler = MinMaxScaler()
    distance_scaler = MinMaxScaler()
    rank_scaler.fit(np.arange(len(candidates)).reshape(-1, 1))
    distance_scaler.fit(np.array(distances).reshape(-1, 1))

    scored_candidates = [ScoredCandidate(c_emb.candidate,
                                         rank,
                                         distances[rank],
                                         np.nansum([
                                             alpha * rank_scaler.transform([[rank]])[0][0],
                                             (1 - alpha) * distance_scaler.transform([[distances[rank]]])[0][0]
                                         ]))
                         for rank, c_emb in enumerate(candidates)]

    return sorted(scored_candidates, key=lambda s_cand: s_cand.score)


def _remove_dates(input_str):
    """
    Remove dates from a string.
    :param input_str: a string
    :return:
    """
    s = re.sub(r'([a-zA-Z]+)([0-9]+)', r'\1 \2', input_str)  # split tokens like 2011-11-29November -> 2011-11-29 November
    s = re.sub(r'([0-9]+)([a-zA-Z]+)', r'\1 \2 ', s)  # split tokens like November2011 -> November 2011

    tokens = s.split()
    f = []
    for token in tokens:
        try:
            parse(token)
        except:
            try:
                parse(re.sub(f"[{string_utils.punctuation}]", '', token))  # remove punctuation (?3,600 -> 3600)
            except:
                f.append(token)

    return " ".join(f)


def _remove_single_char(input_str):
    return " ".join(filter(lambda x: len(x) > 1 or x == 'a', input_str.split()))


def _remove_numbers(input_str):
    return " ".join(filter(lambda x: not x.replace('.', '').replace(',', '').isdigit(), input_str.split()))


def _remove_brackets(input_str):
    """
    Remove brackets content (if it starts in the first 5 tokens). If the text starts with a bracket,
    the content is removed first, then the remaining part of string is processed.
    E.g.:
    - _remove_brackets("Barack Hussein Obama II (US /bəˈrɑːk huːˈseɪn oʊˈbɑːmə/; born August 4, 1961)")
      > Barack Hussein Obama II
    - _remove_brackets("Alessandro Del Piero Ufficiale OMRI (born 9 November 1974)")
      > Alessandro Del Piero Ufficiale OMRI (born 9 November 1974)
    - _remove_brackets("(This article is about India (state).) Punjab (/pʌnˈdʒɑːb/) is a state in North India.")
      > Punjab  is a state in North India.
    :param input_str:
    :return:
    """

    max_pos = len(" ".join(input_str.split()[:5]))  # check if the bracket occurs in the first 5 tokens
    # check missplaced brackets (see dbr:Sherkot comment)
    if '(' in input_str and ')' in input_str and input_str.index('(') < input_str.index(')'):
        bracket_idx = input_str.index("(")  # get the bracket index
    else:
        return input_str

    # Remove brackets until the index of the first bracket changes (if not, you're deleting nested brackets)
    while bracket_idx < max_pos and '(' in input_str and ')' in input_str and bracket_idx == input_str.index("("):
        input_str = re.subn(r'\([^()]*\)', '', input_str, count=1)[0].strip()

    # Process the obtained string again if the string started with a bracket
    if bracket_idx == 0:
        return _remove_brackets(input_str)

    return input_str


def simplify_string(input_str, dates=True, numbers=True, single_char=True, brackets=True):
    s = input_str
    if brackets:
        s = _remove_brackets(s)
    if dates:
        s = _remove_dates(s)
    if numbers:
        s = _remove_numbers(s)
    if single_char:
        s = _remove_single_char(s)
    return s


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    :return a boolean which indicates if it is a date
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def is_float(string):
    """
    Return whether the string can be interpreted as a float.

    :param string: str, string to check for float
    :return a boolean which indicates if it is a float
    """
    try:
        float(string)
        return True

    except ValueError:
        return False


def toList(list_of_list):
    """
    Convert a list of list in a list

    :param list_of_list: a list of list
    :return: the flattened list
    """
    return [item for sublist in list_of_list if sublist is not None for item in sublist]
