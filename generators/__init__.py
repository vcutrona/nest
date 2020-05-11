import functools
# import multiprocessing as mp
import operator
import os
from abc import ABC
from configparser import ConfigParser

from diskcache import Cache
from scipy.spatial.distance import cosine
from tqdm.contrib.concurrent import process_map

from generators.helpers import AbstractHelper

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

    def _chunk_it(self, _list):
        """
        Utility function to split a list into chunks of size self._chunk_size.
        :param _list: the list to split
        :return:
        """
        for i in range(0, len(_list), self._chunk_size):
            yield _list[i:i + self._chunk_size]


class SimpleGenerator(Generator):
    def _get_candidates(self, labels):
        raise NotImplementedError

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
        cached_entries, to_compute = [], self._get_short_labels_set(labels)

        # check which short labels have been already computed
        if self._config.getboolean('cache'):
            cached_entries, to_compute = self._get_cached_entries(to_compute)
        # compute all the unseen short labels
        new_entries = list(self._get_candidates(to_compute))
        self._update_cache(new_entries)  # write new entries to cache

        entries_dict = dict(cached_entries + new_entries)
        results = []
        for label in labels:  # aggregate results for short labels
            label_candidates = []
            for short_label in self._get_short_labels(label):
                label_candidates = label_candidates + entries_dict[short_label]
            results.append((label, list(dict.fromkeys(label_candidates))))  # remove duplicates without sorting

        return results

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
    def _get_cache_key(self, label, context):
        return label, context, self.__class__.__name__

    def _get_cached_entries(self, lc_pairs):
        to_compute = []
        cached_entries = []

        for label, context in lc_pairs:
            cache_key = self._get_cache_key(label, context)
            candidates = cache.get(cache_key)
            if candidates is None:
                to_compute.append((label, context))
            else:
                cached_entries.append(((label, context), candidates))

        return cached_entries, to_compute

    def _update_cache(self, entries):
        for lc_pair, candidates in entries:
            cache.set(self._get_cache_key(*lc_pair), candidates)  # override everything ALWAYS!

    def _multi_search(self, lc_pairs):
        raise NotImplementedError

    def multi_search(self, labels, contexts):
        lc_pairs = list(zip(labels, contexts))
        if self._threads > 1:
            # p = mp.Pool(self._threads)
            # results = functools.reduce(operator.iconcat, p.map(self._multi_search, self._chunk_it(lc_pairs)), [])
            results = functools.reduce(operator.iconcat,
                                       process_map(self._multi_search,
                                                   list(self._chunk_it(lc_pairs)),
                                                   max_workers=self._threads),
                                       [])
        else:
            results = self._multi_search(lc_pairs)

        return dict(results)

    def search(self, label, context):
        return self.multi_search([label], [context])


class EmbeddingContextGenerator(ContextGenerator):
    def __init__(self, config, threads, chunk_size, generator: SimpleGenerator):
        super().__init__(config, threads, chunk_size)
        self._generator = generator
        self._abstract_helper = AbstractHelper()

    def _get_embeddings_from_sentences(self, sentences):
        raise NotImplementedError

    def _multi_search(self, lc_pairs):
        cached_entries, to_compute = [], lc_pairs
        if self._config.getboolean('cache'):
            cached_entries, to_compute = self._get_cached_entries(lc_pairs)

        # lookup candidates for each label
        lookup_results = self._generator.multi_search([pair[0] for pair in to_compute])

        # embed each context
        contexts_embs = dict(zip([pair[1] for pair in to_compute],
                                 self._get_embeddings_from_sentences([pair[1] for pair in to_compute])))
                                 # self._get_embeddings_from_sentences(['%s %s' % (pair[0], pair[1]) for pair in to_compute])))

        # get the abstract of each candidate and embed it
        if self._config['abstract'] == 'short':
            abstracts = self._abstract_helper.fetch_short_abstracts(
                functools.reduce(operator.iconcat, lookup_results.values(), []),
                int(self._config['abstract_max_tokens']))
        else:
            abstracts = self._abstract_helper.fetch_long_abstracts(
                functools.reduce(operator.iconcat, lookup_results.values(), []),
                int(self._config['abstract_max_tokens']))
        abstracts_embs = dict(zip(abstracts.keys(), self._get_embeddings_from_sentences(list(abstracts.values()))))

        new_entries = []
        for label, context in to_compute:
            if context and contexts_embs[context].size:  # no context -> nothing to compare with -> return basic lookup
                scored_candidates = []
                for candidate in lookup_results[label]:
                    scored_candidate = (candidate, 2)  # TODO is this a good initialization?
                    if abstracts[candidate] and abstracts_embs[candidate].size:
                        scored_candidate = (candidate, cosine(abstracts_embs[candidate], contexts_embs[context]))
                    scored_candidates.append(scored_candidate)
                scored_candidates = [candidate for candidate, score in
                                     sorted(scored_candidates, key=lambda k: k[1])]
                # scored_candidates = sorted(scored_candidates, key=lambda k: k[1])  # print as DEBUG info?
            else:
                scored_candidates = lookup_results[label]  # no context -> return basic lookup
            new_entries.append(((label, context), scored_candidates))

        self._update_cache(new_entries)  # write new entries to cache
        return cached_entries + new_entries
