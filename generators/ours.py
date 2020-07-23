import os

import numpy
from allennlp.commands.elmo import ElmoEmbedder
from sentence_transformers import SentenceTransformer

from generators import EmbeddingContextGenerator
from generators.baselines import ESLookup


class FastElmo(EmbeddingContextGenerator):
    def __init__(self, config='FastElmo', lookup=ESLookup()):
        super().__init__(config, 1, 10000, lookup)  # force single-process execution
        self._model = ElmoEmbedder(cuda_device=0,
                                   weight_file=os.path.join(os.path.dirname(__file__),
                                                            'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'),
                                   options_file=os.path.join(os.path.dirname(__file__),
                                                             'elmo_2x4096_512_2048cnn_2xhighway_options.json'))
        d = dict(self._config)
        self._cache_key_suffix = "%s_%s" % (self.__class__.__name__,
                                            "|".join(sorted(["%s:%s" % (k, d[k])
                                                             for k in d if k in ['abstract', 'abstract_max_tokens']])))

    def _get_cache_key(self, label, context):
        return label, context, self._cache_key_suffix

    def _get_embeddings_from_sentences(self, sentences, mode="layer_2"):
        """
        Generates the sentence embeddings from ELMO for each sentence in a list of strings.
        :param sentences: the sentences you want to embed
        :param mode: from which layer of ELMO you want the embedding.
                     "mean" gets the embedding of the three elmo layers for each token
        :return:
        """
        # model_outputs = []
        # for i in range(0, len(sentences), 10000):  # avoid CUDA going out of memory
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

        embeds = [numpy.mean(embed, axis=0) if embed.size else embed for embed in embeds]

        return embeds


class FastTransformers(EmbeddingContextGenerator):
    def __init__(self, config='FastTransformer'):
        super().__init__(config, 1, 10000, ESLookup())  # force single-process execution
        self._model = SentenceTransformer(self._config['model'])

        d = dict(self._config)
        self._cache_key_suffix = "%s_%s" % (self.__class__.__name__,
                                            "|".join(sorted(["%s:%s" % (k, d[k])
                                                             for k in d if k in ['model', 'abstract',
                                                                                 'abstract_max_tokens']])))

    def _get_cache_key(self, label, context):
        return label, context, self._cache_key_suffix

    def _get_embeddings_from_sentences(self, sentences):
        return self._model.encode(sentences)

# fe = FastElmo()
# print(fe.search("Bobtail", "Cat Female 7 10 Red"))  # breed, species, sex, age, weight, colour
# print(fe.search("Bobtail", ""))  # breed, species, sex, age, weight, colour
# print(fe.search("Beagle", "Dog Male 4 11 Black, tan and white"))  # breed, species, sex, age, weight, colour
# print(fe.search("Boys Don't Cry",
#                 "84 4-May-02 59 USA Peirce 1999 B-    2.7  "))  # CEA_ROUND1 row 5138 -> all abstracts are empty
