import functools
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
import operator
import os
from abc import ABC
from configparser import ConfigParser

from diskcache import Cache

cache = Cache('cache')


class Generator(ABC):
    def __init__(self, config, threads, chunk_size):
        cfg = ConfigParser()
        cfg.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        self._config = cfg[config]
        assert threads > 0
        self._threads = threads
        self._chunk_size = chunk_size

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

    def _get_short_labels_set(self, labels, max_tokens=5):
        """
        Utility function to get the set of all the short labels computed for
        all the labels given as input.
        :param labels: a list of labels to shorten
        :param max_tokens: max length (words) for the longest short label
        :return:
        """
        ext_labels = []
        for label in labels:
            ext_labels = ext_labels + self._get_short_labels(label, max_tokens)

        return list(dict.fromkeys(ext_labels))


class SimpleGenerator(Generator):
    def _multi_search(self, labels):
        raise NotImplementedError

    def _chunk_it(self, labels):
        for i in range(0, len(labels), self._chunk_size):
            yield labels[i:i + self._chunk_size]

    def _update_cache(self, label, candidates):
        cache_key = (label, self.__class__.__name__)
        if self._config.getboolean('cache'):
            cache.add(cache_key, candidates)  # not override existing entries, but add new entries
        else:
            cache.set(cache_key, candidates)  # override everything!

    def multi_search(self, labels):
        cached_entries = []
        to_compute = []

        # collect results from cache, if active
        for label in labels:
            if self._config.getboolean('cache'):
                cache_key = (label, self.__class__.__name__)
                entry = cache.get(cache_key)
                if entry is None:
                    to_compute.append(label)
                else:
                    cached_entries.append((label, entry))
            else:
                to_compute.append(label)

        new_entries = []
        if to_compute:
            if self._threads > 1:
                # p = mp.Pool(self._threads)
                # new_entries = functools.reduce(operator.iconcat, p.map(self._multi_search, self._chunk_it(to_compute)), [])
                new_entries = functools.reduce(operator.iconcat, process_map(self._multi_search, list(self._chunk_it(to_compute)), max_workers=self._threads), [])
            else:
                new_entries = self._multi_search(to_compute)

        for label, short_entries in new_entries:
            label_candidates = []
            for short_label, short_label_candidates in short_entries:
                self._update_cache(short_label, short_label_candidates)
                label_candidates = label_candidates + short_label_candidates  # append the candidates to the long label
            label_candidates = list(dict.fromkeys(label_candidates))
            self._update_cache(label, label_candidates)  # it might happen that label == a short labels -  no matter
            cached_entries.append((label, label_candidates))  # return only the long labels

        return dict(cached_entries)

    def search(self, label):
        return self.multi_search([label])


class ContextGenerator(Generator):
    def search(self, label, context):
        raise NotImplementedError
