import os
from typing import List

import numpy as np
import requests
from gensim.models import KeyedVectors

from utils.caching import CacheWrapper, KVPair


class EmbeddingModel:
    def get_vectors(self, uris: List[str]):
        """
        Get vectors for the given URIs
        :param uris: a list of URIs
        """
        raise NotImplementedError


class EmbeddingModelService(EmbeddingModel):
    def __init__(self, url):
        self._url = url
        self._cache = CacheWrapper(os.path.join(os.path.dirname(__file__),
                                                '.cache',
                                                'EmbeddingModel',
                                                self.__class__.__name__,),
                                   int(4e9))

    def get_vectors(self, uris: List[str]):
        """
        Get vectors for the given URIs
        :param uris: a DBpedia resource URI, or a list of DBpedia resource URIs
        :return: a dict {<uri>: <vec>}. <vec> is None if it does not exist a vector for <uri>.
        """
        cached_entries, to_compute = self._cache.get_cached_entries(uris)
        results = dict(cached_entries)
        if to_compute:
            data = {'uri': to_compute}
            response = requests.get(self._url, params=data)
            results.update({uri: np.array(vec) if vec else None for uri, vec in response.json().items()})
            self._cache.update_cache_entries([KVPair(uri, (uri, results[uri])) for uri in to_compute])
        return results


class RDF2Vec(EmbeddingModelService):
    def __init__(self, uri='http://titan:5999/r2v/uniform'):
        super().__init__(uri)


class WORD2Vec(EmbeddingModelService):
    def __init__(self, uri='http://titan:5998/w2v/dbp-300'):
        super().__init__(uri)


class ABS2Vec(EmbeddingModelService):
    def __init__(self, uri='http://titan:5997/a2v/bert-1024'):
        super().__init__(uri)


class OWL2Vec(EmbeddingModel):
    def __init__(self):
        model_filepath = os.path.join(os.path.dirname(__file__), 'data', 'dbpedia_owl2vec')
        self._model = KeyedVectors.load_word2vec_format(model_filepath)

    def get_vectors(self, uris: List[str]):
        """
        Get vectors for the given URIs
        :param uris: a DBpedia class URI, or a list of DBpedia class URIs
        :return: a dict {<uri>: <vec>}. <vec> is None if it does not exist a vector for <uri>.
        """
        return {uri: self._model[uri] if uri in self._model else None for uri in uris}


class TEE(EmbeddingModel):
    def __init__(self):
        model_filepath = os.path.join(os.path.dirname(__file__), 'data', 'tee.wv')
        self._model = KeyedVectors.load(model_filepath)

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
