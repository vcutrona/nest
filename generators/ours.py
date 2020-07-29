import os

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from sentence_transformers import SentenceTransformer

from data_model.generator import EmbeddingCandidateGeneratorConfig
from generators import EmbeddingCandidateGenerator
from lookup import LookupService


class FastElmo(EmbeddingCandidateGenerator):
    """
    Baseline method to re-rank candidates accordingly with vector similarities, based on the ELMO embeddings.
    """

    def __init__(self, lookup_service: LookupService,
                 config=EmbeddingCandidateGeneratorConfig(max_subseq_len=5,
                                                          abstract='short',
                                                          abstract_max_tokens=15)):
        super().__init__(lookup_service, config, threads=1, chunk_size=10000)  # force single-process execution
        self._model = ElmoEmbedder(cuda_device=0,
                                   weight_file=os.path.join(os.path.dirname(__file__),
                                                            'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'),
                                   options_file=os.path.join(os.path.dirname(__file__),
                                                             'elmo_2x4096_512_2048cnn_2xhighway_options.json'))

    def _get_embeddings_from_sentences(self, sentences, mode="layer_2"):
        """
        Generates the sentence embeddings from ELMO for each sentence in a list of strings.
        :param sentences: the sentences to embed
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


class FastBert(EmbeddingCandidateGenerator):
    """
    Baseline method to re-rank candidates accordingly with vector similarities, based on the BERT embeddings.
    """

    def __init__(self, lookup_service: LookupService,
                 config=EmbeddingCandidateGeneratorConfig(max_subseq_len=5,
                                                          abstract='short',
                                                          abstract_max_tokens=512)):
        super().__init__(lookup_service, config, threads=1, chunk_size=10000)  # force single-process execution
        self._model = SentenceTransformer('bert-base-nli-mean-tokens')

    def _get_embeddings_from_sentences(self, sentences):
        """
        Generates the sentence embeddings from BERT for each sentence in a list of strings.
        :param sentences: the sentences to embed
        :return: a list of embeddings
        """
        return self._model.encode(sentences)

# fe = FastElmo()
# print(fe.search("Bobtail", "Cat Female 7 10 Red"))  # breed, species, sex, age, weight, colour
# print(fe.search("Bobtail", ""))  # breed, species, sex, age, weight, colour
# print(fe.search("Beagle", "Dog Male 4 11 Black, tan and white"))  # breed, species, sex, age, weight, colour
# print(fe.search("Boys Don't Cry",
#                 "84 4-May-02 59 USA Peirce 1999 B-    2.7  "))  # CEA_ROUND1 row 5138 -> all abstracts are empty
