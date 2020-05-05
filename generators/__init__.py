import functools
import operator
import os
from abc import ABC
from configparser import ConfigParser
import multiprocessing as mp
from diskcache import Cache

cache = Cache('cache')


class Generator(ABC):
    def __init__(self, config):
        cfg = ConfigParser()
        cfg.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        self._config = cfg[config]

    @staticmethod
    def _get_short_labels(label, max_tokens=5):
        """
        Utility function to shorten long labels (that are useless for lookups)
        :param label: the label to shorten
        :param max_tokens: max length for the longest short label
        :return:
        """
        tokens = label.split()
        return [" ".join(tokens[:i+1]) for i in reversed(range(min(max_tokens, len(tokens))))]

    def _get_short_labels_set(self, labels):
        ext_labels = []
        for label in labels:
            ext_labels = ext_labels + self._get_short_labels(label)

        return list(dict.fromkeys(ext_labels))


class SimpleGenerator(Generator):
    def _multi_search(self, labels):
        raise NotImplementedError

    def multi_search(self, labels):
        cached_entries = {}
        to_compute = []

        cache_on = self._config.getboolean('cache')

        # collect results from cache
        for label in labels:
            if cache_on:
                cache_key = (label, self.__class__.__name__)
                entry = cache.get(cache_key)
                if entry is not None:
                    cached_entries[label] = entry
                else:
                    to_compute.append(label)
            else:
                to_compute.append(label)

        docs = dict(self._multi_search(to_compute))  # lookup labels not in cache

        for label, candidates in docs.items():  # update cache with lookup results
            cache_key = (label, self.__class__.__name__)
            if cache_on:
                cache.add(cache_key, candidates)  # not override existing entries
            else:
                cache.set(cache_key, candidates)  # override!

        docs = dict(cached_entries, **docs)
        return {label: functools.reduce(operator.iconcat,
                                        [docs[x] for x in self._get_short_labels(label) if x in docs],
                                        [])
                for label in labels}

    def search(self, label):
        return self.multi_search([label])


class ContextGenerator(Generator):
    def search(self, label, context):
        raise NotImplementedError
