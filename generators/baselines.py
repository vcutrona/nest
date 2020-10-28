import multiprocessing as mp
from collections import Counter
from typing import List, Iterable, Tuple, Any, Dict

from nltk import edit_distance

from data_model.generator import GeneratorResult, CandidateGeneratorConfig, FactBaseConfig
from data_model.lookup import SearchKey
from generators import CandidateGenerator
from lookup import LookupService
from utils.functions import tokenize, simplify_string, first_sentence
from utils.kgs import DBpediaWrapper


class LookupGenerator(CandidateGenerator):
    """
    A generator that just forwards lookup results.
    """
    def __init__(self, lookup_service: LookupService, config=CandidateGeneratorConfig(5),
                 threads=mp.cpu_count(), chunk_size=5000):
        super().__init__(lookup_service, config, threads, chunk_size)

    def _select_candidates(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        """
        Candidate selection method. This implementation just forwards the LookupService results.
        :param search_keys: a list of SearchKeys to use for the candidate retrieval
        :return: a list of GeneratorResult
        """
        return self._lookup_candidates(search_keys)


class FactBase(CandidateGenerator):
    def __init__(self, lookup_service: LookupService, config: FactBaseConfig = FactBaseConfig(0)):
        super().__init__(lookup_service, config, threads=1, chunk_size=10000)  # do not split samples in chunks!
        self._dbp = DBpediaWrapper()
        # self._cache = Cache(
        #     os.path.join(
        #         os.path.dirname(__file__),
        #         '.cache',
        #         self.__class__.__name__,
        #         self._config.cache_dir()),
        #     size_limit=int(8e9))

    # def _get_cached_entries(self, keys: List[SearchKey]) -> Tuple[List[GeneratorResult],
    #                                                               List[SearchKey]]:
    #     """
    #     Retrieve already computed results from cache
    #     :param keys: a list of keys to retrieve
    #     :return: a tuple (<cached results>, <labels to embed>)
    #     """
    #     to_compute = []
    #     cached_entries = []
    #
    #     for key in keys:
    #         entry = self._cache.get(key)
    #         if entry is None:
    #             to_compute.append(key)
    #         else:
    #             cached_entries.append(entry)
    #
    #     return cached_entries, to_compute

    def _get_description_tokens(self, uri: str) -> List[str]:
        """
        Get the description tokens of the given entity

        :param uri: the URI to be read
        :return: a list of keywords
        """
        descriptions = self._dbp.get_descriptions(uri)
        if descriptions:
            desc = simplify_string(descriptions[0], dates=False, numbers=False, single_char=False, brackets=True)
            short_desc = first_sentence(desc)
            if short_desc:
                return tokenize(descriptions[0])

        return []

    def _get_most_frequent(self, list_: Iterable, n: int = 1) -> List:
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

    def _contains_facts(self, facts: Dict[Any, List[Tuple[str, str]]], min_occurrences: int):
        """
        Return a dict with a list of properties sorted by fact occurrences.
        Properties with less than `min_occurrences` are filtered out.

        :param facts: a dict {id: [uri-literal pairs]}
        :param min_occurrences: minimum number of occurrences
        :return: a dict {id: [list of prop-#occurrences]}
        """
        relations = {}
        for col, pairs in facts.items():
            all_col_relations = []
            for pair, col_relations in self._dbp.get_relations(pairs).items():
                all_col_relations += col_relations
            relations[col] = sorted([(rel, count) for rel, count in Counter(all_col_relations).items()
                                     if count >= min_occurrences],
                                    key=lambda item: item[1], reverse=True)
        return relations

    def _search_strict(self, candidates: List[str], types: List[str], description_tokens: List[str]) -> List[str]:
        """
        Execute a search operation on a given label restricting the results to those of an acceptable
        type, having one of the most frequent tokens in their description values

        :param candidates: a list of candidates from LookupResult
        :param types: a list of acceptable types
        :param description_tokens: a list of tokens
        :return: a list of candidates
        """
        refined_candidates = []
        types_set = set(types)
        description_tokens_set = set(description_tokens)
        for candidate in candidates:
            c_tokens = set(self._get_description_tokens(candidate))
            c_types = set(self._dbp.get_types(candidate))
            if c_tokens & description_tokens_set and c_types & types_set:
                refined_candidates.append(candidate)  # preserve ordering

        return refined_candidates

    def _search_loose(self, label: str, relation: str, value: str) -> List[str]:
        """
        Execute a fuzzy search (Levenshtein) and get the results for which there exist a fact <result, relation, value>.
        Return the subject with the minimal edit distance from label.

        :param label: the label to look for
        :param relation: a relation
        :param value: a value
        :return: a list of results
        """
        candidates = []
        for candidate, c_labels in self._dbp.get_subjects(relation, value).items():
            scores = sorted([(candidate, edit_distance(label, c_label)) for c_label in c_labels], key=lambda s: s[1])
            if scores:
                candidates.append(scores[0])  # keep the best label for each candidate

        return [c[0] for c in sorted(candidates, key=lambda s: s[1])] # sort by edit distance

    def _select_candidates(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        """
        Candidate selection method that implements the Efthymiou's FactBase lookup.
        :param search_keys: a list of SearchKeys to use for the candidate retrieval
        :return: a list of GeneratorResult
        """

        # Get candidates for each label
        # lr = self._lookup_candidates(search_keys)
        # print(lr)
        lookup_results = dict(self._lookup_candidates(search_keys))
        generator_results = {}

        all_types = []  # list of candidates types
        desc_tokens = []  # list of tokens in candidates descriptions
        facts = {}  # dict of possible facts in table (fact := <top_concept, ?p, support_col_value>)

        # First scan - raw results
        for search_key, candidates in lookup_results.items():
            # Handle cells with some candidates (higher confidence)
            if candidates:
                top_result = candidates[0]
                all_types += self._dbp.get_types(top_result)
                desc_tokens += self._get_description_tokens(top_result)
                # Check for relationships if there is only one candidate (very high confidence)
                if len(candidates) == 1:
                    generator_results[search_key] = GeneratorResult(search_key, candidates)
                    for col_id, col_value in search_key.context:
                        if col_id not in facts:
                            facts[col_id] = []
                        facts[col_id].append((top_result, col_value))

        acceptable_types = self._get_most_frequent(all_types, n=5)
        description_tokens = self._get_most_frequent(desc_tokens)
        relations = {col_id: candidate_relations[0][0]
                     for col_id, candidate_relations in self._contains_facts(facts, min_occurrences=5).items()
                     if candidate_relations}

        # Second scan - refinement and loose searches
        for search_key, candidates in lookup_results.items():
            # Skip already annotated cells
            if search_key in generator_results:
                continue

            # Strict search: filter lists of candidates by removing entities that do not match types and tokens
            refined_candidates = self._search_strict(candidates,
                                                     acceptable_types,
                                                     description_tokens)
            if refined_candidates:
                generator_results[search_key] = GeneratorResult(search_key, refined_candidates)
                continue

            # Loose search: increase the recall by allowing a big margin of edit distance (Levenshtein)
            context_dict = dict(search_key.context)
            for col_id, relation in relations.items():
                refined_candidates = self._search_loose(search_key.label, relation, context_dict[col_id])
                if len(refined_candidates) > 0:
                    generator_results[search_key] = GeneratorResult(search_key, refined_candidates)
                    break

            # Coarse- and fine-grained searches failed: check for an exact match!
            if search_key not in generator_results:
                if candidates:
                    labels = self._dbp.get_label(candidates[0])
                    if search_key.label in labels:
                        generator_results[search_key] = GeneratorResult(search_key, candidates)
                else:  # No results
                    generator_results[search_key] = GeneratorResult(search_key, [])

        return list(generator_results.values())
