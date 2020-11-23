import os
from typing import List

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from sentence_transformers import SentenceTransformer

from data_model.generator import EmbeddingCandidateGeneratorConfig, FastBertConfig, Embedding
from data_model.lookup import SearchKey
from generators import EmbeddingCandidateGenerator
from lookup import LookupService
from utils.functions import simplify_string


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

# fe = FastElmo()
# print(fe.search("Bobtail", "Cat Female 7 10 Red"))  # breed, species, sex, age, weight, colour
# print(fe.search("Bobtail", ""))  # breed, species, sex, age, weight, colour
# print(fe.search("Beagle", "Dog Male 4 11 Black, tan and white"))  # breed, species, sex, age, weight, colour
# print(fe.search("Boys Don't Cry",
#                 "84 4-May-02 59 USA Peirce 1999 B-    2.7  "))  # CEA_ROUND1 row 5138 -> all abstracts are empty
