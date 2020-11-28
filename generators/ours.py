import os
from typing import List

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from nltk import edit_distance
from sentence_transformers import SentenceTransformer

from data_model.generator import EmbeddingCandidateGeneratorConfig, FastBertConfig, Embedding, FactBaseConfig, \
    GeneratorResult
from data_model.lookup import SearchKey
from generators import EmbeddingCandidateGenerator
from generators.baselines import FactBase
from lookup import LookupService
from utils.functions import simplify_string
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
            scores = sorted(
                [(candidate, edit_distance(label, c_label) / max(len(label), len(c_label))) for c_label in c_labels],
                key=lambda s: s[1])
            if scores and scores[0][1] <= 0.2:  # set a threshold for the edit distance
                candidates.append(scores[0])  # keep the best label for each candidate

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

        all_types = []  # list of candidates types
        desc_tokens = []  # list of tokens in candidates descriptions
        facts = {}  # dict of possible facts in table (fact := <top_concept, ?p, support_col_value>)

        # First scan - raw results
        for search_key, candidates in lookup_results.items():
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
        description_tokens = self._get_most_frequent(desc_tokens, n=3)  # take more tokens
        relations = {col_id: candidate_relations[0][0]
                     for col_id, candidate_relations in self._contains_facts(facts, min_occurrences=5).items()
                     if candidate_relations}

        # Second scan - refinement and loose searches
        for search_key, candidates in lookup_results.items():

            # Skip already annotated cells
            if search_key in generator_results:
                continue

            # Strict search: filter lists of candidates by removing entities that do not match types and tokens
            refined_candidates_strict = self._search_strict(candidates,
                                                            acceptable_types,
                                                            description_tokens)
            if len(refined_candidates_strict) == 1:
                generator_results[search_key] = GeneratorResult(search_key, refined_candidates_strict)
                continue

            # Loose search: increase the recall by allowing a big margin of edit distance (Levenshtein)
            context_dict = dict(search_key.context)
            for col_id, relation in relations.items():
                refined_candidates_loose = self._search_loose(search_key.label, relation, context_dict[col_id])
                # Take candidates found with both the search strategies (strict and loose)
                refined_candidates = [candidate for candidate in refined_candidates_strict
                                      if candidate in refined_candidates_loose]
                if refined_candidates:  # Annotate only if there are common candidates
                    generator_results[search_key] = GeneratorResult(search_key, refined_candidates)
                    break

            # Coarse- and fine-grained searches did not find common candidates:
            if search_key not in generator_results:
                if refined_candidates_strict:  # Resort to the strict search, if there are results
                    generator_results[search_key] = GeneratorResult(search_key, refined_candidates_strict)
                elif candidates and search_key.label in self._dbp.get_label(candidates[0]):  # Perfect label match
                    generator_results[search_key] = GeneratorResult(search_key, candidates)
                else:  # No results
                    generator_results[search_key] = GeneratorResult(search_key, [])

        return list(generator_results.values())


class FactBaseMLType(FactBase):

    def __init__(self, *lookup_services: LookupService, config: FactBaseConfig):
        super().__init__(*lookup_services, config=config)
        self._type_predictor = None

    def _search_strict(self, candidates: List[str], types: List[str], description_tokens: List[str]) -> List[str]:
        refined_candidates = []
        types_set = set(types)
        description_tokens_set = set(description_tokens)
        types = self._type_predictor.predict_types(candidates, size=2)

        for candidate in candidates:
            c_tokens = set(self._get_description_tokens(candidate))
            if types[candidate]:
                c_types = set(types[candidate])
            else:
                c_types = set(self._dbp.get_types(candidate))

            if c_tokens & description_tokens_set and c_types & types_set:
                refined_candidates.append(candidate)  # preserve ordering
        return refined_candidates

    def _get_candidates_for_column(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        """
        Override the parent method just to initialize the model in the parallel process
        :param search_keys:
        :return:
        """
        if not self._type_predictor:
            self._type_predictor = RDF2VecTypePredictor()
        return super()._get_candidates_for_column(search_keys)
