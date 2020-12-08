import functools
import operator
import os
from typing import List

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from sentence_transformers import SentenceTransformer

from data_model.generator import EmbeddingCandidateGeneratorConfig, FastBertConfig, Embedding, FactBaseConfig, \
    GeneratorResult
from data_model.lookup import SearchKey
from generators import EmbeddingCandidateGenerator
from generators.baselines import FactBase
from lookup import LookupService
from utils.functions import simplify_string, get_most_frequent
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
                                 self._type_predictor.predict_types(list(candidates_set)).values(),
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
