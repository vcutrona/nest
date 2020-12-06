import os
from abc import ABC
from typing import List, Tuple

from diskcache import Cache

from data_model.lookup import LookupResult, LookupServiceConfig
from utils.functions import strings_subsequences


class LookupService(ABC):
    """
    Abstract model for a lookup service.
    """

    def __init__(self, config: LookupServiceConfig):
        self._config = config
        self._cache = Cache(os.path.join(
            os.path.dirname(__file__),
            '.cache',
            self.__class__.__name__,
            self._config.cache_dir()))

    @property
    def max_hits(self):
        return self._config.max_hits

    def _update_cache(self, results: List[LookupResult]):
        """
        Update cache entries with new results
        :param results: a list of LookupResults
        :return:
        """
        for lookup_result in results:
            self._cache.set(lookup_result.label, lookup_result)  # ALWAYS override!

    def _get_cached_entries(self, labels: List[str]) -> Tuple[List[LookupResult], List[str]]:
        """
        Return a tuple (<cached results>, <labels to lookup>)
        :param labels: a list of strings to lookup
        :return:
        """
        to_compute = []
        cached_entries = []

        for label in labels:
            entry = self._cache.get(label)
            if entry is None:
                to_compute.append(label)
            else:
                cached_entries.append(entry)

        return cached_entries, to_compute

    def _lookup(self, labels: List[str]) -> List[LookupResult]:
        """
        Actual lookup function. To be implemented by all the subclasses.
        :param labels:
        :return:
        """
        raise NotImplementedError

    def lookup(self, labels: List[str]) -> List[LookupResult]:
        """
        Returns a list of LookupResult, based on the given labels.
        Results are fetched from a local cache, if enabled. If not enabled, already cached results (if any)
        will be updated with the new results.
        :param labels: a list of strings to lookup
        :return: a list of LookupResult
        """
        cached_entries, to_compute = [], labels

        if self._config.enable_cache:
            cached_entries, to_compute = self._get_cached_entries(to_compute)

        new_results = self._lookup(to_compute)
        self._update_cache(new_results)  # write new entries to cache

        return cached_entries + new_results

    def lookup_subsequences(self, labels: List[str], max_subseq_len) -> List[LookupResult]:
        """
        Given a label, the method searches for all its subsequences (from length max_subseq_len to 1)
        Results are then aggregated, keeping the original rank.
        :param labels: a list of labels
        :param max_subseq_len: length of the longest subsequence to compute
        :return: a list of LookupResult
        """
        subsequences_dict, subsequences_set = strings_subsequences(labels, max_subseq_len)
        lookup_results = dict(self.lookup(list(subsequences_set)))
        results = []
        for label in labels:  # aggregate subsequences results for each label
            label_results = []
            for subsequence in subsequences_dict[label]:
                label_results += lookup_results[subsequence]
            results.append(LookupResult(label, label_results))
        return results
