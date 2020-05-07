import functools
from tqdm.contrib.concurrent import process_map
# import multiprocessing as mp
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
    def _get_candidates(self, labels):
        raise NotImplementedError

    def _chunk_it(self, labels):
        for i in range(0, len(labels), self._chunk_size):
            yield labels[i:i + self._chunk_size]

    def _get_cache_key(self, label):
        return label, self.__class__.__name__

    def _update_cache(self, entries):
        for label, candidates in entries:
            cache.set(self._get_cache_key(label), candidates)  # override everything ALWAYS!

    def _get_cached_entries(self, labels):
        to_compute = []
        cached_entries = []
        for label in labels:

            cache_key = self._get_cache_key(label)
            entry = cache.get(cache_key)
            if entry is None:
                to_compute.append(label)
            else:
                cached_entries.append((label, entry))

        return cached_entries, to_compute

    def _multi_search(self, labels):
        cached_entries, to_compute = [], labels
        if self._config.getboolean('cache'):
            cached_entries, to_compute = self._get_cached_entries(labels)

        new_short_entries = list(self._get_candidates(self._get_short_labels_set(to_compute)))
        self._update_cache(filter(lambda x: x[0] not in labels, new_short_entries))  # cache short entries

        new_short_entries_dict = dict(new_short_entries)

        new_entries = []
        for label in to_compute:
            label_candidates = []
            short_labels_of_label = self._get_short_labels(label)
            for short_label in short_labels_of_label:
                label_candidates = label_candidates + new_short_entries_dict[short_label]  # append candidates to the long label
            new_entries.append((label, list(dict.fromkeys(label_candidates))))  # remove duplicates without sorting
        self._update_cache(new_entries)  # cache new entries

        return cached_entries + new_entries

    def multi_search(self, labels):
        if self._threads > 1:
            # p = mp.Pool(self._threads)
            # results = functools.reduce(operator.iconcat, p.map(self._multi_search, self._chunk_it(to_compute)), [])
            results = functools.reduce(operator.iconcat,
                                       process_map(self._multi_search,
                                                   list(self._chunk_it(labels)),
                                                   max_workers=self._threads),
                                       [])
        else:
            results = self._multi_search(labels)

        return dict(results)

    def search(self, label):
        return self.multi_search([label])


class ContextGenerator(Generator):
    def search(self, label, context):
        raise NotImplementedError
