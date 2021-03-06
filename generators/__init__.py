import functools
import operator
import os
from typing import List, Tuple, Union

import numpy as np
from diskcache import Cache

from data_model.dataset import Table
from data_model.generator import EmbeddingCandidateGeneratorConfig, CandidateEmbeddings, GeneratorResult, Embedding
from data_model.lookup import SearchKey
from lookup import LookupService
from utils.functions import weighting_by_ranking, truncate_string
from utils.kgs import DBpediaWrapper


class Generator:
    """
    An interface for Generator
    """

    @property
    def id(self) -> str:
        raise NotImplementedError

    def get_candidates(self, table: Table) -> List[GeneratorResult]:
        """
        Candidate selection method. To be implement in all the subclasses.
        :param table: a Table object
        :return: a list of GeneratorResult
        """
        raise NotImplementedError


class CandidateGenerator(Generator):
    """
    Abstract Candidate Generator.
    A Candidate Generator uses a bunch of LookupService object for retrieving candidates; if candidates are missing
    for a key, the subsequent LookupService is used.
    """

    def __init__(self, *lookup_services: LookupService, config):
        self._lookup_services = lookup_services
        self._config = config

    @property
    def id(self) -> str:
        return "__".join([self.__class__.__name__] +
                         [f"{lookup_service.__class__.__name__}{lookup_service.max_hits}"
                          for lookup_service in self._lookup_services] +
                         [self._config.config_str()])

    def _lookup_candidates(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        """
        Get a set of candidates from the lookup service.
        :param search_keys: a list of SearchKeys
        :return: a list of LookupResult
        """
        lookup_results = {}
        for lookup_service in self._lookup_services:
            labels = [search_key.label for search_key in search_keys
                      if search_key.label not in lookup_results or not lookup_results[search_key.label]]
            if not labels:
                break

            if self._config.max_subseq_len and self._config.max_subseq_len > 0:
                lookup_results.update(dict(lookup_service.lookup_subsequences(labels, self._config.max_subseq_len)))
            else:
                lookup_results.update(dict(lookup_service.lookup(labels)))

        return [GeneratorResult(search_key, lookup_results[search_key.label]) for search_key in search_keys]

    def get_candidates(self, table: Table) -> List[GeneratorResult]:
        """
        Candidate selection method. To be implement in all the subclasses.
        :param table: a Table object
        :return: a list of GeneratorResult
        """
        raise NotImplementedError


class EmbeddingCandidateGenerator(CandidateGenerator):
    """
    Abstract generator that re-rank candidates accordingly with vector similarities.
    For each candidate, both the abstract and label embeddings are computed and then compared using
    the cosine distance measure.
    """

    def __init__(self, *lookup_services: LookupService, config: EmbeddingCandidateGeneratorConfig):
        super().__init__(*lookup_services, config=config)

        self._abstract_helper = DBpediaWrapper()
        self._cache = Cache(
            os.path.join(
                os.path.dirname(__file__),
                '.cache',
                self.__class__.__name__,
                self._config.cache_dir()),
            size_limit=int(8e9))

    def _embed_search_keys(self, search_keys: List[SearchKey]) -> List[Embedding]:
        """
        Abstract method to compute search keys embeddings.
        :param search_keys: the list of SearchKey to embed
        :return: a list of embeddings
        """
        raise NotImplementedError

    def _embed_abstracts(self, abstracts: List[str]) -> List[Embedding]:
        """
        Abstract method to compute abstracts embeddings.
        :param abstracts: the list of abstracts to embed
        :return: a list of embeddings
        """
        raise NotImplementedError

    def _update_cache(self, embeddings: List[Embedding]):
        """
        Update cache entries with new embeddings
        :param embeddings: a list of Embedding
        :return:
        """
        for embedding in embeddings:
            self._cache.set(embedding.key, embedding)  # ALWAYS override!

    def _get_cached_entries(self, keys: List[Union[str, SearchKey]]) -> Tuple[List[Embedding],
                                                                              List[Union[str, SearchKey]]]:
        """
        Retrieve already computed embeddings from cache
        :param keys: a list of keys to retrieve
        :return: a tuple (<cached results>, <labels to embed>)
        """
        to_compute = []
        cached_entries = []

        for key in keys:
            entry = self._cache.get(key)
            if entry is None:
                to_compute.append(key)
            else:
                cached_entries.append(entry)

        return cached_entries, to_compute

    def get_candidates(self, table: Table) -> List[GeneratorResult]:
        """
        Return a list of candidates, sorted by the cosine distance between their label and context embeddings.
        :param table: a Table object
        :return: a list of GeneratorResult
        """
        search_keys = [table.get_search_key(cell_) for cell_ in table.get_gt_cells()]
        lookup_results = dict(self._lookup_candidates(search_keys))  # collect lookup result from the super class

        # create embed for each label and context pair
        cached_entries, to_compute = self._get_cached_entries(search_keys)
        new_results = self._embed_search_keys(to_compute)
        self._update_cache(new_results)  # write new entries to cache

        search_keys_embs = dict(cached_entries + new_results)

        # create embed for the candidates' abstracts
        candidates_list = functools.reduce(operator.iconcat, lookup_results.values(), [])
        if self._config.abstract == 'short':
            abstracts = self._abstract_helper.fetch_short_abstracts(candidates_list)
        else:
            abstracts = self._abstract_helper.fetch_long_abstracts(candidates_list)
        abstracts = {candidate: truncate_string(abstract, self._config.abstract_max_tokens)
                     for candidate, abstract in abstracts.items()}

        cached_entries, to_compute = self._get_cached_entries(abstracts.values())
        new_results = self._embed_abstracts(to_compute)
        self._update_cache(new_results)
        abstracts_embeddings = dict(cached_entries + new_results)

        # do not zip! abstracts.values() might contain duplicates...
        abstracts_embs = {candidate: abstracts_embeddings[abstract] for candidate, abstract in abstracts.items()}

        results = []
        for search_key in search_keys:
            candidates_embeddings = []
            context_emb = np.nan
            if search_key.context and search_keys_embs[search_key].size:
                context_emb = search_keys_embs[search_key]
            for candidate in lookup_results[search_key]:
                abstract_emb = np.nan
                if candidate in abstracts and abstracts_embs[candidate].size:
                    abstract_emb = abstracts_embs[candidate]
                candidates_embeddings.append(CandidateEmbeddings(candidate, context_emb, abstract_emb))

            results.append(GeneratorResult(
                search_key, [c.candidate for c in weighting_by_ranking(candidates_embeddings,
                                                                       self._config.alpha,
                                                                       self._config.default_score)]))

        return results


class HybridGenerator(Generator):
    """
    An HybridGenerator that executes a sequence of CandidateGenerator.
    When candidates are missing for some entries, the next CandidateGenerator is executed.
    """

    def __init__(self, *generators: CandidateGenerator):
        self._generators = generators

    @property
    def id(self) -> str:
        return "__".join([self.__class__.__name__] + [generator.id for generator in self._generators])

    @property
    def generators(self) -> List[CandidateGenerator]:
        return list(self._generators)

    def get_candidates(self, table: Table) -> List[GeneratorResult]:
        res_dict = dict()
        empty_candidates = []
        for generator in self._generators:
            res = generator.get_candidates(table)
            if not res_dict:
                res_dict = dict(res)
            else:
                res_dict.update({search_key: candidates for search_key, candidates in dict(res).items()
                                 if search_key in empty_candidates})
            empty_candidates = [search_key for search_key, candidates in res_dict.items()
                                if not candidates]
            if not empty_candidates:  # no more cells to annotate
                break
        return [GeneratorResult(search_key, candidates) for search_key, candidates in res_dict.items()]


class HybridGeneratorSimulator:
    """
    Simulate HybridGenerator results on already computed tables.
    """

    @staticmethod
    def get_candidates(*tables: Table):
        """
        Combine results of already computed tables.
        :param tables: a list of annotated tables (i-th table annotated by the HybridGenerator i-th generator)
        :return:
        """
        results = []
        missing_cells = tables[0].get_gt_cells()
        for table in tables:
            new_missing = []
            for cell in missing_cells:
                if cell in table.cell_annotations:
                    entities = table.cell_annotations[cell].entities
                    if entities:
                        results.append(GeneratorResult(search_key=table.get_search_key(cell),
                                                       candidates=[entity.uri for entity in entities]))
                    else:
                        new_missing.append(cell)
                else:
                    new_missing.append(cell)
            missing_cells = new_missing
            if not missing_cells:
                break

        return results
