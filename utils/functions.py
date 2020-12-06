import re
import string as string_utils
from collections import Counter
from typing import List, Iterable
from typing import Tuple, Dict, Set

import nltk
import numpy as np
from dateutil.parser import parse
from gensim import matutils
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from numpy import dot
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler

from data_model.generator import CandidateEmbeddings, ScoredCandidateEmbeddings

nltk.download('stopwords')


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


def get_most_frequent(list_: Iterable, n: int = 1) -> List:
    """
    Get the `n` most frequent values in a list

    :param list_: a list
    :param n: number of values to be returned
    :return: a list with the most frequent values
    """
    if not list_:
        return []
    counter = Counter(list_)
    return [list(tup_) for tup_ in zip(*counter.most_common(n))][0]


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
                         default_score=np.nan) -> List[ScoredCandidateEmbeddings]:
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

    scored_candidates = [
        ScoredCandidateEmbeddings(c_emb.candidate,
                                  np.nansum([
                                      alpha * rank_scaler.transform([[rank]])[0][0],
                                      (1 - alpha) * distance_scaler.transform([[distances[rank]]])[0][0]
                                  ]),
                                  rank,
                                  distances[rank]
                                  )
        for rank, c_emb in enumerate(candidates)]

    return sorted(scored_candidates)  # key=lambda s_cand: s_cand.score)


def _remove_dates(input_str):
    """
    Remove dates from a string.
    :param input_str: a string
    :return:
    """
    s = re.sub(r'([a-zA-Z]+)([0-9]+)', r'\1 \2',
               input_str)  # split tokens like 2011-11-29November -> 2011-11-29 November
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


def tokenize(sentence: str, language: str = 'english', stemming: bool = False) -> List[str]:
    """
    Simple preprocessing: removes punctuation and stopwords and apply stemming if needed
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [w for w in tokenizer.tokenize(sentence.lower()) if w not in stopwords.words(language)]
    if stemming:
        porter = PorterStemmer()
        tokens = [porter.stem(t) for t in tokens]
    return tokens


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


def first_sentence(input_str, min_length=5):
    """
    Return the text before the first full stop index.
    We consider a dot "." a full stop if precedes a space and it follows:
    - a word with at least 2 chars
    - a punctuation symbol

    :param input_str: the text
    :param min_length: minimum required length
    :return: the first sentence
    """
    # list_ = re.findall("(\s.[.]|\s..[.])", input_str)
    # for x in list_:
    #     if x in input_str:
    #         input_str = input_str.replace(x, '')
    # if len(input_str) > 0:
    #     return input_str.partition('.')[0]
    #
    # pieces = input_str.split('.')  # get all pieces separated by dots (mainly sentences)
    # pieces.reverse()
    # short_sentence = ""
    # while pieces and len(short_sentence.split()) < min_length:
    #     short_sentence += pieces.pop() + '.'  # pop a piece
    #     while len(pieces[-1]) == 1:  # pop all the subsequent pieces if they are chars (e.g., acronyms like "U.S.")
    #         short_sentence += pieces.pop() + '.'

    punct_possible = '!#$%&\(\)\*\+,\-:;<=>?@_|~\{\}'

    end = None
    regex = "([\w" + punct_possible + "]{2,}|[" + punct_possible + "]|\d)\.\s"
    full_stop = re.search(regex, input_str)
    if full_stop:
        end = full_stop.end()
    partial_string = input_str[:end]
    # Check if we reach the required length, otherwise find a new full stop
    while len(partial_string.split()) < min_length and partial_string != input_str:
        sub_str = input_str[len(partial_string):]
        full_stop = re.search(regex, sub_str)
        end = None
        if full_stop:
            end = full_stop.end()
        app = input_str[len(partial_string):][:end]
        partial_string += app

    return partial_string.strip()


def cosine_similarity(v1, v2):
    return dot(matutils.unitvec(v1), matutils.unitvec(v2))

# def flatten(list_):
#     """
#     Convert a list of list in a list
#
#     :param list_: flatten a list of lists
#     :return: the flattened list
#     """
#     return [item for sublist in list_ if sublist is not None for item in sublist]

#
# a = ['Koei (a little test)  2011-11-29November 29, 2011 ?3,600JP 5.67 GB',
#      'Ubisoft 2012-01-19 2012-01-19  PG (a test)',
#      'Factor 5 2008-02-29 2008-02-29 7 PG']
# for s in a:
#     print(s, '|||', simplify_string(s))

# print(_remove_brackets("Barack Hussein Obama II (US /bəˈrɑːk huːˈseɪn oʊˈbɑːmə/; born August 4, 1961)"))
# print(_remove_brackets("Del Piero (pronunciation: [del ˈpjɛːro]) Ufficiale OMRI (born 9 November 1974)"))
# print(_remove_brackets("Alessandro Del Piero Ufficiale OMRI (born 9 November 1974)"))
