import pickle
import numpy as np
from utils.kgs import KGEmbedding


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
        self._r2v = KGEmbedding.RDF2VEC

    def predict_types(self, uris, size=1):
        types = {}
        vectors = {k: v for k, v in self._r2v.get_vectors(uris).items() if v}
        pred = self._model.predict(np.array(list(vectors.values())))

        for idx, uri in enumerate(vectors):
            types[uri] = [self._classes[index] for index in np.argsort(-pred[idx])[:size]]

        for uri in uris:
            if uri not in types:
                types[uri] = []

        return types
