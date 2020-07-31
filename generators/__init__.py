import functools
import operator
from typing import List

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from data_model.generator import CandidateGeneratorConfig, EmbeddingCandidateGeneratorConfig, \
    CandidateEmbeddings, GeneratorResult
from data_model.lookup import SearchKey
from generators.utils import AbstractCollector
from lookup import LookupService
from utils.functions import chunk_list, weighting_by_ranking


class CandidateGenerator:
    """
    Abstract Candidate Generator.
    """

    def __init__(self, lookup_service: LookupService, config: CandidateGeneratorConfig, threads, chunk_size):
        assert threads > 0
        self._threads = threads
        self._chunk_size = chunk_size

        self._config = config
        self._lookup_service = lookup_service

    @property
    def id(self):
        return self.__class__.__name__, self._lookup_service.__class__.__name__, self._config.config_str()

    def _lookup_candidates(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        """
        Get a set of candidates from the lookup service.
        :param search_keys: a list of SearchKeys
        :return: a list of LookupResult
        """
        labels = [search_key.label for search_key in search_keys]
        if self._config.max_subseq_len and self._config.max_subseq_len > 0:
            lookup_results = dict(self._lookup_service.lookup_subsequences(labels,
                                                                           self._config.max_subseq_len))
        else:
            lookup_results = dict(self._lookup_service.lookup(labels))
        return [GeneratorResult(search_key, lookup_results[search_key.label]) for search_key in search_keys]

    def _select_candidates(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        """
        Candidate selection method. To be implement in all the subclasses.
        :param search_keys: a list of SearchKeys to use for the candidate retrieval
        :return: a list of GeneratorResult
        """
        raise NotImplementedError

    def multi_search(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        """
        Parallel candidate retrieval execution
        :param search_keys: a list of search keys
        :return: a list of GeneratorResult
        """
        if self._threads > 1:
            results = functools.reduce(operator.iconcat,
                                       process_map(self._select_candidates,
                                                   list(chunk_list(search_keys, self._chunk_size)),
                                                   max_workers=self._threads),
                                       [])

        else:  # avoid CUDA re-initialization in forked subprocess
            results = []
            for search_keys_chunk in tqdm(chunk_list(search_keys, self._chunk_size)):
                results += self._select_candidates(search_keys_chunk)

        return results

    def search(self, search_key: SearchKey) -> List[GeneratorResult]:
        """
        Commodity method to perform a single query
        :param search_key: a search key
        :return: a list of GeneratorResult
        """
        return self.multi_search([search_key])


class EmbeddingCandidateGenerator(CandidateGenerator):
    """
    Abstract generator that re-rank candidates accordingly with vector similarities
    For each candidate, both the abstract and label embeddings are computed and then compared using
    the cosine distance measure.
    """

    def __init__(self, lookup_service: LookupService, config: EmbeddingCandidateGeneratorConfig, threads, chunk_size):
        super().__init__(lookup_service, config, threads, chunk_size)

        self._abstract_helper = AbstractCollector()

    def _embed_search_keys(self, search_keys: List[SearchKey]) -> List[np.ndarray]:
        """
        Abstract method to compute search keys embeddings.
        :param search_keys: the list of SearchKey to embed
        :return: a list of embeddings
        """
        raise NotImplementedError

    def _embed_abstracts(self, abstracts: List[str]) -> List[np.ndarray]:
        """
        Abstract method to compute abstracts embeddings.
        :param abstracts: the list of abstracts to embed
        :return: a list of embeddings
        """
        raise NotImplementedError

    def _select_candidates(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        """
        Return a list of candidates, sorted by the cosine distance between their label and context embeddings.
        :param search_keys: a list of SearchKey to search
        :return: a list of GeneratorResult
        """
        lookup_results = dict(self._lookup_candidates(search_keys))  # collect lookup result from the super class

        # create embed for each label and context pair
        search_keys_embs = dict(zip(search_keys, self._embed_search_keys(search_keys)))

        if self._config.abstract == 'short':
            abstracts = self._abstract_helper.fetch_short_abstracts(
                functools.reduce(operator.iconcat, lookup_results.values(), []),
                int(self._config.abstract_max_tokens))
        else:
            abstracts = self._abstract_helper.fetch_long_abstracts(
                functools.reduce(operator.iconcat, lookup_results.values(), []),
                int(self._config.abstract_max_tokens))
        abstracts_embs = dict(zip(abstracts.keys(), self._embed_abstracts(list(abstracts.values()))))

        results = []
        for search_key in search_keys:
            candidates_embeddings = []
            context_emb = None
            if search_key.context and search_keys_embs[search_key].size:
                context_emb = search_keys_embs[search_key]
            for candidate in lookup_results[search_key]:
                abstract_emb = None
                if candidate in abstracts and abstracts_embs[candidate].size:
                    abstract_emb = abstracts_embs[candidate]
                candidates_embeddings.append(CandidateEmbeddings(candidate, context_emb, abstract_emb))

            results.append(GeneratorResult(
                search_key, [c.candidate for c in weighting_by_ranking(candidates_embeddings,
                                                                       self._config.alpha,
                                                                       self._config.default_score)]))

        return results
