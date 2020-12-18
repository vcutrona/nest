import functools
import operator
from collections import Counter
from concurrent.futures.process import ProcessPoolExecutor
from itertools import product
from typing import List, Tuple, Any, Dict

import networkx as nx
import numpy as np
from nltk import edit_distance

from data_model.dataset import Table
from data_model.generator import GeneratorResult, FactBaseConfig, EmbeddingOnGraphConfig, ScoredCandidate, \
    LookupGeneratorConfig
from data_model.lookup import SearchKey
from experiments.stats import FactBaseStats
from generators import CandidateGenerator
from lookup import LookupService
from utils.embeddings import WORD2Vec, RDF2Vec
from utils.functions import tokenize, simplify_string, first_sentence, cosine_similarity, chunk_list, get_most_frequent
from utils.kgs import DBpediaWrapper


class LookupGenerator(CandidateGenerator):
    """
    A generator that just forwards lookup results.
    """

    def __init__(self, *lookup_services: LookupService, config: LookupGeneratorConfig):
        super().__init__(*lookup_services, config=config)

    def get_candidates(self, table: Table) -> List[GeneratorResult]:
        """
        Candidate selection method. This implementation just forwards the LookupService results.
        :param table: a Table object
        :return: a list of GeneratorResult
        """
        search_keys = [table.get_search_key(cell_) for cell_ in table.get_gt_cells()]
        if self._config.max_workers == 1:
            results = self._lookup_candidates(search_keys)
        else:  # Parallelize at cell level (no dependencies between cells in the same col/row)
            with ProcessPoolExecutor(self._config.max_workers) as pool:
                results = pool.map(self._lookup_candidates, chunk_list(search_keys, self._config.chunk_size))

        return functools.reduce(operator.iconcat, results, [])


class FactBase(CandidateGenerator):
    """
    Candidate generation method that implements the FactBase lookup [Efthymiou+, 2017].
    """

    def __init__(self, *lookup_services: LookupService, config: FactBaseConfig):
        super().__init__(*lookup_services, config=config)
        self._dbp = DBpediaWrapper()
        self._stats = FactBaseStats(self.__class__.__name__)

    def _get_descriptions_tokens(self, uris: List[str]) -> Dict[str, List[str]]:
        """
        Get the tokens of descriptions of a given list of entities

        :param uris: a list of URIs
        :return: a dict uri: tokens
        """
        tokens = {}
        for uri, descriptions in self._dbp.get_descriptions_for_uris(uris).items():
            if descriptions:
                desc = simplify_string(descriptions[0], dates=False, numbers=False, single_char=False, brackets=True)
                short_desc = first_sentence(desc)
                if short_desc:
                    tokens[uri] = tokenize(short_desc)
                else:
                    tokens[uri] = []
            else:
                tokens[uri] = []

        return tokens

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

    def _search_strict(self, candidates: List[str],
                       acceptable_types: List[str], types: Dict[str, List[str]],
                       acceptable_tokens: List[str], description_tokens: Dict[str, List[str]]) -> List[str]:
        """
        Execute a search operation on a given label restricting the results to those of an acceptable
        type, having one of the most frequent tokens in their description values.
        If the acceptable types (or tokens) are not given, all types (tokens) are considered as valid.

        :param candidates: a list of candidates from LookupResult
        :param acceptable_types: a list of acceptable types
        :param types: a dict uri: types
        :param acceptable_tokens: a list of acceptable tokens
        :param description_tokens: a dict uri: tokens
        :return: a list of candidates
        """
        refined_candidates = []
        acceptable_types_set = set(acceptable_types)
        acceptable_tokens_set = set(acceptable_tokens)
        for candidate in candidates:
            type_hit = True
            if acceptable_types:
                type_hit = set(types[candidate]) & acceptable_types_set

            token_hit = True
            if acceptable_tokens:
                token_hit = set(description_tokens[candidate]) & acceptable_tokens_set

            if type_hit and token_hit:
                refined_candidates.append(candidate)  # preserve ordering

        return refined_candidates

    def _search_loose(self, label: str, relation: str, value: str, threshold: float = -1) -> List[str]:
        """
        Execute a fuzzy search (Levenshtein) and get the results for which there exist a fact <result, relation, value>.
        Return the subject with the minimal edit distance from label.

        :param label: the label to look for
        :param relation: a relation
        :param value: a value
        :param threshold: max edit distance to consider. Set a negative value to ignore it
        :return: a list of results
        """
        candidates = []
        for candidate, c_labels in self._dbp.get_subjects(relation, value).items():
            scores = sorted(
                [(candidate, edit_distance(label, c_label) / max(len(label), len(c_label))) for c_label in c_labels],
                key=lambda s: s[1])
            if scores:  # keep the best label for each candidate
                if threshold < 0:
                    candidates.append(scores[0])
                elif scores[0][1] <= threshold:
                    candidates.append(scores[0])

        return [c[0] for c in sorted(candidates, key=lambda s: s[1])]  # sort by edit distance

    def _get_candidates_for_column(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        """
        Generate candidate for a set of search keys.
        The assumption is that all the search keys belong to the same column.
        :param search_keys: a list of search_keys
        :return:
        """
        lookup_results = dict(self._lookup_candidates(search_keys))
        generator_results = {}

        # Pre-fetch types and description of the top candidate of each candidates set
        candidates_set = list({candidates[0] for candidates in lookup_results.values() if candidates})
        types = functools.reduce(operator.iconcat,
                                 self._dbp.get_direct_types_for_uris(candidates_set).values(),
                                 [])
        description_tokens = functools.reduce(operator.iconcat,
                                              self._get_descriptions_tokens(candidates_set).values(),
                                              [])
        facts = {}  # dict of possible facts in table (fact := <top_concept, ?p, support_col_value>)

        # First scan - raw results
        for search_key, candidates in lookup_results.items():
            if candidates:  # Handle cells with some candidates (higher confidence)
                if len(candidates) == 1:
                    generator_results[search_key] = GeneratorResult(search_key, candidates)
                    # Check for relationships if there is only one candidate (very high confidence)
                    for col_id, col_value in search_key.context:
                        if col_id not in facts:
                            facts[col_id] = []
                        facts[col_id].append((candidates[0], col_value))
                    self._stats.incr_exact()

        acceptable_types = get_most_frequent(types, n=5)
        acceptable_tokens = get_most_frequent(description_tokens)
        relations = {col_id: candidate_relations[0][0]
                     for col_id, candidate_relations in self._contains_facts(facts, min_occurrences=5).items()
                     if candidate_relations}

        # Second scan - refinement and loose searches
        for search_key, candidates in lookup_results.items():
            # Skip already annotated cells
            if search_key in generator_results:
                continue

            if candidates:
                # Pre-fetch types and description of all the candidates of not annotated cells
                types = self._dbp.get_direct_types_for_uris(candidates)
                description_tokens = self._get_descriptions_tokens(candidates)

                # Strict search: filter lists of candidates by removing entities that do not match types and tokens
                refined_candidates = self._search_strict(candidates,
                                                         acceptable_types,
                                                         types,
                                                         acceptable_tokens,
                                                         description_tokens)
                if refined_candidates:
                    generator_results[search_key] = GeneratorResult(search_key, refined_candidates)
                    self._stats.incr_strict()
                    continue

            # Loose search: increase the recall by allowing a big margin of edit distance (Levenshtein)
            context_dict = dict(search_key.context)
            for col_id, relation in relations.items():
                refined_candidates = self._search_loose(search_key.label, relation, context_dict[col_id])
                if len(refined_candidates) > 0:
                    generator_results[search_key] = GeneratorResult(search_key, refined_candidates)
                    self._stats.incr_loose()
                    break

            # Coarse- and fine-grained searches failed: no results
            if search_key not in generator_results:
                generator_results[search_key] = GeneratorResult(search_key, [])
                self._stats.incr_empty()

        return list(generator_results.values())

    def get_candidates(self, table: Table) -> List[GeneratorResult]:
        """
        This method annotates each table column separately, by finding which are the column types and
        the relationships between the current column and the other.
        :param table: a list of search_keys, which must belong to the same table column
        :return: a list of GeneratorResult
        """
        self._stats.init(table.dataset_id, table.tab_id)
        col_search_keys = {}
        for cell in table.get_gt_cells():
            if cell.col_id not in col_search_keys:
                col_search_keys[cell.col_id] = []
            col_search_keys[cell.col_id].append(table.get_search_key(cell))

        if self._config.max_workers == 1:
            results = [self._get_candidates_for_column(search_keys) for search_keys in col_search_keys.values()]
        else:
            with ProcessPoolExecutor(self._config.max_workers) as pool:
                results = pool.map(self._get_candidates_for_column, col_search_keys.values())

        return functools.reduce(operator.iconcat, results, [])


class EmbeddingOnGraph(CandidateGenerator):

    def __init__(self, *lookup_services: LookupService,
                 config: EmbeddingOnGraphConfig = EmbeddingOnGraphConfig(max_subseq_len=0,
                                                                         max_candidates=8,
                                                                         thin_out_frac=0.25)):
        super().__init__(*lookup_services, config=config)
        self._dbp = DBpediaWrapper()
        self._w2v = RDF2Vec()

    def _get_candidates_for_column(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        lookup_results = dict(self._lookup_candidates(search_keys))

        # Create a complete directed k-partite disambiguation graph where k is the number of search keys.
        disambiguation_graph = nx.DiGraph()
        sk_nodes = {}
        personalization = {}  # prepare dict for pagerank with normalized priors
        embeddings = {}
        for search_key, candidates in lookup_results.items():
            degrees = self._dbp.get_degree_for_uris(candidates)
            embeddings.update(self._w2v.get_vectors(candidates))

            # Filter candidates that have an embedding in w2v.
            nodes = sorted([(candidate, {'weight': degrees[candidate]})
                            for candidate in candidates
                            if embeddings[candidate] is not None],
                           key=lambda x: x[1]['weight'], reverse=True)

            # Take only the max_candidates most relevant (highest priors probability) candidates.
            nodes = nodes[:self._config.max_candidates]
            disambiguation_graph.add_nodes_from(nodes)
            sk_nodes[search_key] = [n[0] for n in nodes]

            # Store normalized priors
            weights_sum = sum([x[1]['weight'] for x in nodes])
            for node, props in nodes:
                if node not in personalization:
                    personalization[node] = []
                personalization[node].append(props['weight'] / weights_sum if weights_sum > 0 else 0)

        # Add weighted edges among the nodes in the disambiguation graph.
        # Avoid to connect nodes in the same partition.
        # Weights of edges are the cosine similarity between the nodes which the edge is connected to.
        # Only positive weights are considered.
        for search_key, nodes in sk_nodes.items():
            other_nodes = set(disambiguation_graph.nodes()) - set(nodes)
            for node, other_node in product(nodes, other_nodes):
                v1 = embeddings[node]
                v2 = embeddings[other_node]
                cos_sim = cosine_similarity(v1, v2)
                if cos_sim > 0:
                    disambiguation_graph.add_weighted_edges_from([(node, other_node, cos_sim)])

        # Thin out a fraction of edges which weights are the lowest
        thin_out = int(self._config.thin_out_frac * len(disambiguation_graph.edges.data("weight")))
        disambiguation_graph.remove_edges_from(
            sorted(disambiguation_graph.edges.data("weight"), key=lambda tup: tup[2])[:thin_out])

        # Page rank computaton - epsilon is increased by a factor 2 until convergence
        page_rank = None
        epsilon = 1e-6
        while page_rank is None:
            try:
                page_rank = nx.pagerank(disambiguation_graph,
                                        tol=epsilon, max_iter=50, alpha=0.9,
                                        personalization={node: np.mean(weights)
                                                         for node, weights in personalization.items()})
            except nx.PowerIterationFailedConvergence:
                epsilon *= 2  # lower factor can be used too since pagerank is extremely fast

        # Sort candidates -> the higher the score, the better the candidate (reverse=True)
        return [GeneratorResult(search_key,
                                [c.candidate for c in sorted([ScoredCandidate(candidate, page_rank[candidate])
                                                              for candidate in candidates],
                                                             reverse=True)])
                for search_key, candidates in sk_nodes.items()]

    def get_candidates(self, table: Table) -> List[GeneratorResult]:
        """
        This method annotates each table column separately, by finding which are the column types and
        the relationships between the current column and the other.
        :param table: a list of search_keys, which must belong to the same table column
        :return: a list of GeneratorResult
        """
        col_search_keys = {}
        for cell in table.get_gt_cells():
            if cell.col_id not in col_search_keys:
                col_search_keys[cell.col_id] = []
            col_search_keys[cell.col_id].append(table.get_search_key(cell))

        col_search_keys = {col: chunk_list(search_keys, 500) for col, search_keys in col_search_keys.items()}

        if self._config.max_workers == 1:
            results = [self._get_candidates_for_column(search_keys)
                       for search_keys_list in col_search_keys.values()
                       for search_keys in search_keys_list]
        else:
            with ProcessPoolExecutor(self._config.max_workers) as pool:
                results = pool.map(self._get_candidates_for_column,
                                   [search_keys for search_keys_list in col_search_keys.values()
                                    for search_keys in search_keys_list])
        return functools.reduce(operator.iconcat, results, [])
