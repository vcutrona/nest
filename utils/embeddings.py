from enum import Enum
from typing import List

import numpy as np
import requests
from gensim.models import KeyedVectors


class EmbeddingModel:
    def get_vectors(self, uris):
        """
        Get vectors for the given URIs
        :param uris: a list of URIs
        """
        raise NotImplementedError


class EmbeddingModelService(EmbeddingModel):
    def __init__(self, url):
        self._url = url

    def get_vectors(self, uris: List[str]):
        """
        Get vectors for the given URIs
        :param uris: a DBpedia resource URI, or a list of DBpedia resource URIs
        :return: a dict {<uri>: <vec>}. <vec> is None if it does not exist a vector for <uri>.
        """
        data = {'uri': uris}
        response = requests.get(self._url, params=data)
        return {uri: np.array(vec) if vec else None for uri, vec in response.json().items()}


class RDF2Vec(EmbeddingModelService):
    def __init__(self, uri='http://titan:5999/r2v/uniform'):
        super().__init__(uri)


class WORD2Vec(EmbeddingModelService):
    def __init__(self, uri='http://titan:5998/w2v/dbp-300'):
        super().__init__(uri)


class OWL2Vec(EmbeddingModel):
    def __init__(self):
        self._model = KeyedVectors.load_word2vec_format('./dbpedia_owl2vec')

    def get_vectors(self, uris: List[str]):
        """
        Get vectors for the given URIs
        :param uris: a DBpedia class URI, or a list of DBpedia class URIs
        :return: a dict {<uri>: <vec>}. <vec> is None if it does not exist a vector for <uri>.
        """
        return {uri: self._model[uri] if uri in self._model else None for uri in uris}


class TEE(EmbeddingModel):
    def __init__(self):
        self._model = KeyedVectors.load('./tee.wv')

    def get_vectors(self, uris: List[str]):
        """
        Get vectors for the given URIs
        :param uris: a DBpedia class URI, or a list of DBpedia class URIs
        :return: a dict {<uri>: <vec>}. <vec> is None if it does not exist a vector for <uri>.
        """
        vectors = {}
        for uri in uris:
            entity = uri.split('/')[-1]
            if entity in self._model:
                vectors[uri] = self._model[entity]
            else:
                vectors[uri] = None

        return vectors
