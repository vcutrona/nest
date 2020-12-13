import os
import pickle
from enum import Enum

import numpy as np
import requests

from utils.caching import CacheWrapper, KVPair
from utils.embeddings import RDF2Vec, ABS2Vec
from utils.kgs import TYPES_BLACKLIST


class TypePredictorService(Enum):
    RDF2VEC = 'http://titan:5995/predict/r2v'
    ABS2VEC = 'http://titan:5995/predict/a2v'

    def __init__(self, _: str):
        self._cache = CacheWrapper(os.path.join(os.path.dirname(__file__),
                                                '.cache',
                                                self.__class__.__name__,
                                                self.name),
                                   int(4e9))

    def predict_types(self, uris, size=1):
        cached_entries, to_compute = self._cache.get_cached_entries(uris)
        results = {}
        for uri, types in cached_entries:
            if 0 < len(types) < size:
                to_compute.append(uri)
            else:
                results[uri] = types[:size]
        if to_compute:
            data = {'uri': to_compute, 'size': size}
            response = requests.get(self.value, params=data)
            results.update(response.json())
            self._cache.update_cache_entries([KVPair(uri, (uri, results[uri])) for uri in to_compute])
        return results


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

        model_filepath = os.path.join(os.path.dirname(__file__), 'data', 'rdf2vec_pred.keras')
        classes_filepath = os.path.join(os.path.dirname(__file__), 'data', 'rdf2vec_pred_classes.pkl')

        subject_input = keras.Input(shape=(200,), name="title")
        x = layers.Dense(300, activation="relu")(subject_input)
        dense_x = layers.Dense(300, activation="relu")(x)
        likelihood_pred = layers.Dense(236, activation="softmax")(dense_x)
        self._model = keras.Model(inputs=subject_input, outputs=likelihood_pred, )
        self._model.compile(loss="bce",
                            optimizer=keras.optimizers.Adam(lr=0.01)),
        self._model.load_weights(model_filepath)

        self._classes = pickle.load(open(classes_filepath, 'rb'))
        self._r2v = RDF2Vec()

    def predict_types(self, uris, size=1):
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


class ABS2VecTypePredictor:

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

        model_filepath = os.path.join(os.path.dirname(__file__), 'data', 'abs2vec_pred.keras')
        classes_filepath = os.path.join(os.path.dirname(__file__), 'data', 'abs2vec_pred_classes.pkl')

        subject_input = keras.Input(shape=(1024,), name="title")
        x = layers.Dense(300, activation="relu")(subject_input)
        likelihood_pred = layers.Dense(228, activation="softmax")(x)
        self._model = keras.Model(inputs=subject_input, outputs=likelihood_pred, )
        self._model.compile(loss="bce",
                            optimizer=keras.optimizers.Adam(lr=0.01)),
        self._model.load_weights(model_filepath)

        self._classes = pickle.load(open(classes_filepath, 'rb'))
        self._a2v = ABS2Vec()

    def predict_types(self, uris, size=1):
        types = {}
        abs2vectors = {k: v for k, v in self._a2v.get_vectors(uris).items() if v is not None}
        vectors = np.array(list(abs2vectors.values()))
        if vectors.size > 0:
            pred = self._model.predict(vectors)

            for idx, uri in enumerate(abs2vectors):
                types[uri] = [self._classes[index] for index in np.argsort(-pred[idx])[:size]
                              if self._classes[index] not in TYPES_BLACKLIST]

        for uri in uris:
            if uri not in types:
                types[uri] = []

        return types
