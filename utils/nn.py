import pickle
from typing import List

import numpy as np

from utils.embeddings import RDF2Vec
from utils.kgs import TYPES_BLACKLIST


class RDF2VecTypePredictor:

    def __init__(self):

        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:  # Currently, memory growth needs to be the same across GPUs
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)  # Memory growth must be set before GPUs have been initialized

        subject_input = keras.Input(
            shape=(200,), name="title"
        )

        x = layers.Dense(300, activation="relu")(subject_input)
        dense_x = layers.Dense(300, activation="relu")(x)
        likelihood_pred = layers.Dense(236, activation="softmax")(dense_x)
        self._model = keras.Model(inputs=subject_input, outputs=likelihood_pred, )
        self._model.compile(loss="bce",
                            optimizer=keras.optimizers.Adam(lr=0.01)),
        self._model.load_weights('knowledge_lh.keras')

        self._classes = pickle.load(open('classes_list.pkl', 'rb'))
        self._r2v = RDF2Vec()

    def predict_types(self, uris, size: int = 1):
        """
        Predict types for the given URIs
        :param uris: a DBpedia class URI, or a list of DBpedia class URIs
        :param size: the number of types predicted for each URI
        :return: a dict {<uri>: [<type>]}
        """
        types = {}
        rdf2vectors = {k: v for k, v in self._r2v.get_vectors(uris).items() if v is not None}
        vectors = np.array(list(rdf2vectors.values()))
        if vectors.size > 0:
            pred = self._model.predict(vectors)

            for idx, uri in enumerate(rdf2vectors):
                types[uri] = [self._classes[index] for index in np.argsort(-pred[idx])[:size]
                              if self._classes[index] not in TYPES_BLACKLIST]

        for uri in uris:
            if uri not in types:
                types[uri] = []

        return types
