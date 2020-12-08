import functools
import operator
import os

from itertools import product
from typing import List

import networkx as nx
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from sentence_transformers import SentenceTransformer

from data_model.dataset import Table
from data_model.generator import EmbeddingCandidateGeneratorConfig, FastBertConfig, Embedding, FactBaseConfig, \
    GeneratorResult, ScoredCandidate, EmbeddingOnGraphConfig
from data_model.lookup import SearchKey
from generators import EmbeddingCandidateGenerator
from generators.baselines import FactBase, EmbeddingOnGraph
from lookup import LookupService
from utils.embeddings import RDF2Vec, TEE
from utils.functions import get_most_frequent, cosine_similarity, simplify_string
from utils.nn import RDF2VecTypePredictor


class FastElmo(EmbeddingCandidateGenerator):
    """
    A method to re-rank candidates accordingly with vector similarities, based on the ELMO embeddings.
    """

    def __init__(self, *lookup_services: LookupService,
                 config=EmbeddingCandidateGeneratorConfig(max_subseq_len=0,
                                                          abstract='short',
                                                          abstract_max_tokens=15)):
        super().__init__(*lookup_services, config=config)
        self._model = ElmoEmbedder(cuda_device=0,
                                   weight_file=os.path.join(os.path.dirname(__file__),
                                                            'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'),
                                   options_file=os.path.join(os.path.dirname(__file__),
                                                             'elmo_2x4096_512_2048cnn_2xhighway_options.json'))

    def _embed_sentences(self, sentences: List[str], mode) -> List[np.ndarray]:
        """
        Generic method to generate sentence embeddings from ELMO.
        :param sentences: the list of sentences to embed
        :param mode: from which layer of ELMO you want the embedding.
                     "mean" gets the embedding of the three elmo layers for each token
        :return: a list of embeddings
        """
        model_outputs = self._model.embed_sentences([sentence.split() for sentence in sentences], batch_size=16)

        embeds = []
        if mode == "layer_2":
            embeds = [model_output[2] for model_output in model_outputs]

        if mode == "layer_1":
            embeds = [model_output[1] for model_output in model_outputs]

        if mode == "layer_0":
            embeds = [model_output[0] for model_output in model_outputs]

        if mode == "mean":
            embeds = [(model_output[0] + model_output[1] + model_output[2]) / 3 for model_output in model_outputs]

        embeds = [np.mean(embed, axis=0) if embed.size else embed for embed in embeds]

        return embeds

    def _embed_search_keys(self, search_keys: List[SearchKey], mode="layer_2") -> List[Embedding]:
        """
        Generates the sentence embeddings from ELMO for each search key in a list of SearchKey items.
        :param search_keys: the list of SearchKey to embed
        :param mode: from which layer of ELMO you want the embedding.
                     "mean" gets the embedding of the three elmo layers for each token
        :return: a list of embeddings
        """
        sentences = [" ".join([search_key.label] + [x[1] for x in search_key.context]) for search_key in search_keys]
        return [Embedding(search_key, embedding)
                for search_key, embedding in zip(search_keys, self._embed_sentences(sentences, mode))]

    def _embed_abstracts(self, abstracts: List[str], mode='layer_2') -> List[Embedding]:
        """
        Generates the sentence embeddings from ELMO for each abstract in list.
        :param abstracts: the list of abstracts to embed
        :return: a list of embeddings
        """
        return [Embedding(abstract, embedding)
                for abstract, embedding in zip(abstracts, self._embed_sentences(abstracts, mode))]


class FastBert(EmbeddingCandidateGenerator):
    """
    A method to re-rank candidates accordingly with vector similarities, based on the BERT embeddings.
    """

    def __init__(self, *lookup_services: LookupService,
                 config: FastBertConfig = FastBertConfig(max_subseq_len=0,
                                                         abstract='short',
                                                         abstract_max_tokens=512)):
        super().__init__(*lookup_services, config=config)
        self._model = SentenceTransformer('bert-base-nli-mean-tokens')

    def _embed_search_keys(self, search_keys: List[SearchKey]) -> List[Embedding]:
        """
        Generates the sentence/contextual embeddings from BERT for each search key in a list of SearchKey items.
        :param search_keys: the list of SearchKey to embed
        :return: a list of embeddings
        """
        sentences = ["%s %s" % (search_key.label, simplify_string(" ".join([x[1] for x in search_key.context])))
                     for search_key in search_keys]
        if self._config.strategy == 'sentence':
            return [Embedding(search_key, embedding)
                    for search_key, embedding in zip(search_keys, self._model.encode(sentences))]
        else:
            token_embeddings_list = self._model.encode(sentences, output_value='token_embeddings')
            contextual_embeddings = []
            if self._config.strategy == 'context':
                for search_key, token_embeddings in zip(search_keys, token_embeddings_list):
                    label_tokens = self._model.tokenize(search_key.label)
                    contextual_embeddings.append(Embedding(search_key,
                                                           np.mean(token_embeddings[1:len(label_tokens) + 1], axis=0)))
            elif self._config.strategy == 'cls':
                for search_key, token_embeddings in zip(search_keys, token_embeddings_list):
                    contextual_embeddings.append(Embedding(search_key, token_embeddings[0]))
            else:
                raise Exception

            return contextual_embeddings

    def _embed_abstracts(self, abstracts: List[str]) -> List[Embedding]:
        """
        Generates the sentence embeddings from BERT for each abstract in list.
        :param abstracts: the list of abstracts to embed
        :return: a list of embeddings
        """
        return [Embedding(abstract, embedding)
                for abstract, embedding in zip(abstracts, self._model.encode([simplify_string(abstract)
                                                                              for abstract in abstracts]))]


class FactBaseV2(FactBase):

    def _get_candidates_for_column(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        """
        Generate candidate for a set of search keys.
        The assumption is that all the search keys belong to the same column.
        :param search_keys: a list of search_keys
        :return:
        """
        lookup_results = dict(self._lookup_candidates(search_keys))
        generator_results = {}

        # Pre-fetch types, descriptions and labels of the top candidate of each candidates set
        candidates_set = list({candidates[0] for candidates in lookup_results.values() if candidates})
        types = functools.reduce(operator.iconcat,
                                 self._dbp.get_types_for_uris(candidates_set).values(),
                                 [])
        description_tokens = functools.reduce(operator.iconcat,
                                              self._get_descriptions_tokens(candidates_set).values(),
                                              [])
        labels = self._dbp.get_labels_for_uris(candidates_set)  # needed after the loose search
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

        acceptable_types = get_most_frequent(types, n=3)  # take less types
        acceptable_tokens = get_most_frequent(description_tokens, n=3)  # take more tokens
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
                types = self._dbp.get_types_for_uris(candidates)
                description_tokens = self._get_descriptions_tokens(candidates)

                # Strict search: filter lists of candidates by removing entities that do not match types and tokens
                refined_candidates_strict = self._search_strict(candidates,
                                                                acceptable_types, types,
                                                                acceptable_tokens, description_tokens)

                if len(refined_candidates_strict) == 1:
                    generator_results[search_key] = GeneratorResult(search_key, refined_candidates_strict)
                    continue

                # Loose search: increase the recall by allowing a big margin of edit distance (Levenshtein)
                context_dict = dict(search_key.context)
                for col_id, relation in relations.items():
                    refined_candidates_loose = self._search_loose(search_key.label, relation, context_dict[col_id], 0.3)
                    # Take candidates found with both the search strategies (strict and loose)
                    refined_candidates = [candidate for candidate in refined_candidates_loose
                                          if candidate in refined_candidates_strict]
                    if refined_candidates:  # Annotate only if there are common candidates
                        generator_results[search_key] = GeneratorResult(search_key, refined_candidates)
                        break
            else:
                refined_candidates_strict = []

            # Coarse- and fine-grained searches did not find common candidates:
            if search_key not in generator_results:
                if refined_candidates_strict:  # Resort to the strict search, if there are results
                    generator_results[search_key] = GeneratorResult(search_key, refined_candidates_strict)
                elif candidates and search_key.label in labels[candidates[0]]:  # Perfect label match
                    generator_results[search_key] = GeneratorResult(search_key, candidates)
                else:  # No results
                    generator_results[search_key] = GeneratorResult(search_key, [])

        return list(generator_results.values())


class FactBaseMLType(FactBaseV2):

    def __init__(self, *lookup_services: LookupService, config: FactBaseConfig):
        super().__init__(*lookup_services, config=config)
        self._type_predictor = None

    def _get_candidates_for_column(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        """
        Override the parent method just to initialize the model in the parallel process
        :param search_keys:
        :return:
        """
        if not self._type_predictor:  # Lazy init
            self._type_predictor = RDF2VecTypePredictor()

        lookup_results = dict(self._lookup_candidates(search_keys))
        generator_results = {}

        # Pre-fetch types and labels of the top candidate of each candidates set
        candidates_set = list({candidates[0] for candidates in lookup_results.values() if candidates})
        # Predict types using the classifier
        types = functools.reduce(operator.iconcat,
                                 self._type_predictor.predict_types(candidates_set).values(),
                                 [])
        labels = self._dbp.get_labels_for_uris(candidates_set)  # needed after the loose search
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

        acceptable_types = get_most_frequent(types)  # compute frequencies on classifier results
        # description_tokens = self._get_most_frequent(desc_tokens, n=3)
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
                types = self._type_predictor.predict_types(list(candidates), size=2)  # consider the best two types
                missing = [uri for uri in types if not types[uri]]
                dbp_types = self._dbp.get_types_for_uris(missing)
                types.update(dbp_types)

                # Strict search: filter lists of candidates by removing entities that do not match types and tokens
                refined_candidates_strict = self._search_strict(candidates,
                                                                acceptable_types, types,
                                                                [], {})
                if len(refined_candidates_strict) == 1:
                    generator_results[search_key] = GeneratorResult(search_key, refined_candidates_strict)
                    continue

                # Loose search: increase the recall by allowing a big margin of edit distance (Levenshtein)
                context_dict = dict(search_key.context)
                for col_id, relation in relations.items():
                    refined_candidates_loose = self._search_loose(search_key.label, relation, context_dict[col_id], 0.3)
                    # Take candidates found with both the search strategies (strict and loose)
                    refined_candidates = [candidate for candidate in refined_candidates_loose
                                          if candidate in refined_candidates_strict]
                    if refined_candidates:  # Annotate only if there are common candidates
                        generator_results[search_key] = GeneratorResult(search_key, refined_candidates)
                        break
            else:
                refined_candidates_strict = []

            # Coarse- and fine-grained searches did not find common candidates:
            if search_key not in generator_results:
                if refined_candidates_strict:  # Resort to the strict search, if there are results
                    generator_results[search_key] = GeneratorResult(search_key, refined_candidates_strict)
                elif candidates and search_key.label in labels[candidates[0]]:  # Perfect label match
                    generator_results[search_key] = GeneratorResult(search_key, candidates)
                else:  # No results
                    generator_results[search_key] = GeneratorResult(search_key, [])

        return list(generator_results.values())


class EmbeddingOnGraphV2(EmbeddingOnGraph):

    def __init__(self, *lookup_services: LookupService, config: EmbeddingOnGraphConfig):
        super().__init__(*lookup_services, config=config)
        self._w2v = RDF2Vec()
        self._type_predictor = None

    def get_candidates(self, table: Table) -> List[GeneratorResult]:
        if not self._type_predictor:  # Lazy init
            self._type_predictor = RDF2VecTypePredictor()

        search_keys = [table.get_search_key(cell_) for cell_ in table.get_gt_cells()]
        lookup_results = dict(self._lookup_candidates(search_keys))

        # Pre-fetch types of the top candidate of each candidates set
        candidates_set = list({candidates[0] for candidates in lookup_results.values() if candidates})
        # Predict types using the classifier
        types = functools.reduce(operator.iconcat,
                                 self._type_predictor.predict_types(candidates_set).values(),
                                 [])
        acceptable_types = get_most_frequent(types)

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
                            if embeddings[candidate]
                            if set(acceptable_types) &
                            set(self._type_predictor.predict_types([candidate], size=2)[candidate])],
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


class EmbeddingOnGraphMLType(EmbeddingOnGraph):

    def __init__(self, *lookup_services: LookupService, config: EmbeddingOnGraphConfig):
        super().__init__(*lookup_services, config=config)
        self._w2v = RDF2Vec()
        self._type_predictor = None
        self._tee = TEE()

    def get_candidates(self, table: Table) -> List[GeneratorResult]:
        if not self._type_predictor:  # Lazy init
            self._type_predictor = RDF2VecTypePredictor()

        search_keys = [table.get_search_key(cell_) for cell_ in table.get_gt_cells()]
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
                            if embeddings[candidate]],
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
            nodes_type = self._type_predictor.predict_types(nodes)
            other_nodes = set(disambiguation_graph.nodes()) - set(nodes)
            other_nodes_type = self._type_predictor.predict_types(other_nodes)
            type_embeddings = self._tee.get_vectors(list(nodes_type) + list(other_nodes_type))
            for node, other_node in product(nodes, other_nodes):
                v1 = np.concatenate([embeddings[node], type_embeddings[nodes_type[node][0]]])
                v2 = np.concatenate([embeddings[other_node], type_embeddings[other_nodes_type[node][0]]])
                cos_sim = cosine_similarity(v1, v2)
                if cos_sim > 0:
                    disambiguation_graph.add_weighted_edges_from([(node, other_node, cos_sim)])

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


# ITERATIVE PAGERANK DO NOT SEEM IMPROVE RESULTS ON T2D -> LOT OF ANNOTATIONS PROVIDED ALREADY FROM 1st ITERATION
#
# class EmbeddingOnGraphV2(EmbeddingOnGraph):
#
#     def __init__(self, *lookup_services: LookupService, config: EmbeddingOnGraphConfig):
#         super().__init__(*lookup_services, config=config)
#         self._iterations: int = 2
#
#     def get_candidates(self, table: Table) -> List[GeneratorResult]:
#         search_keys = [table.get_search_key(cell_) for cell_ in table.get_gt_cells()]
#         lookup_results = dict(self._lookup_candidates(search_keys))
#         generator_results = {}
#         embeddings = {}
#
#         assert self._iterations > 0
#         for search_key, candidates in lookup_results.items():
#             embeddings.update(self._w2v.get_vectors(candidates))
#         for i in range(0, self._iterations):
#             # Create a complete directed k-partite disambiguation graph where k is the number of search keys.
#             disambiguation_graph = nx.DiGraph()
#             sk_nodes = {}
#             personalization = {}  # prepare dict for pagerank with normalized priors
#             if not [sk for sk, x in lookup_results.items() if sk not in generator_results]:  # if empty -> finish
#                 break
#
#             for search_key, candidates in lookup_results.items():
#
#                 if search_key not in generator_results:
#                     # Filter candidates that have an embedding in w2v.
#                     nodes = sorted([(candidate, {'weight': self._dbp.get_degree(candidate)})
#                                     for candidate in candidates
#                                     if embeddings[candidate]],
#                                    key=lambda x: x[1]['weight'], reverse=True)
#                     alpha = 1
#                     if i == self._iterations - 1:
#                         alpha = 2
#                     # Take only the max_candidates most relevant (highest priors probability) candidates.
#                     nodes = nodes[:self._config.max_candidates * alpha]
#                 else:
#                     # Filter candidates that have an embedding in w2v.
#                     nodes = [(generator_results[search_key].candidates[0],
#                               {'weight': self._dbp.get_degree(generator_results[search_key].candidates[0])})]
#
#                 disambiguation_graph.add_nodes_from(nodes)
#                 sk_nodes[search_key] = [n[0] for n in nodes]
#
#                 # Store normalized priors
#                 weights_sum = sum([x[1]['weight'] for x in nodes])
#                 for node, props in nodes:
#                     if node not in personalization:
#                         personalization[node] = []
#                     personalization[node].append(props['weight'] / weights_sum if weights_sum > 0 else 0)
#
#             # Add weighted edges among the nodes in the disambiguation graph.
#             # Avoid to connect nodes in the same partition.
#             # Weights of edges are the cosine similarity between the nodes which the edge is connected to.
#             # Only positive weights are considered.
#             for search_key, nodes in sk_nodes.items():
#                 other_nodes = set(disambiguation_graph.nodes()) - set(nodes)
#                 for node, other_node in product(nodes, other_nodes):
#                     v1 = embeddings[node]
#                     v2 = embeddings[other_node]
#                     cos_sim = cosine_similarity(v1, v2)
#                     if cos_sim > 0:
#                         disambiguation_graph.add_weighted_edges_from([(node, other_node, cos_sim)])
#
#             # Page rank computaton - epsilon is increased by a factor 2 until convergence
#             page_rank = None
#             epsilon = 1e-6
#             while page_rank is None:
#                 try:
#                     page_rank = nx.pagerank(disambiguation_graph,
#                                             tol=epsilon, max_iter=50, alpha=0.9,
#                                             personalization={node: np.mean(weights)
#                                                              for node, weights in personalization.items()})
#                 except nx.PowerIterationFailedConvergence:
#                     epsilon *= 2  # lower factor can be used too since pagerank is extremely fast
#
#             # Sort candidates -> the higher the score, the better the candidate (reverse=True)
#             for search_key, candidates in sk_nodes.items():
#                 scored_candidates = sorted([ScoredCandidate(candidate, page_rank[candidate])
#                                             for candidate in candidates], reverse=True)
#                 if i == self._iterations - 1:
#                     generator_results[search_key] = GeneratorResult(search_key,
#                                                                     [c.candidate for c in scored_candidates])
#                     continue
#
#                 if len(candidates) == 1:
#                     generator_results[search_key] = GeneratorResult(search_key, candidates)
#                 elif len(candidates) >= 3:  # at least 3 because threshold=best with only 2 candidates
#                     score = [candidate.score for candidate in scored_candidates]
#                     threshold = np.mean(score) + np.std(score) * 0.5  # * 0.5 try multiply std by alpha(<1) to increase precision
#                     if scored_candidates[1].score < threshold:  # if 2nd-best is lower than threshold -> 1st correct
#                         generator_results[search_key] = GeneratorResult(search_key, [scored_candidates[0].candidate])
#
#         return list(generator_results.values())
