from typing import List, Any, Tuple, NamedTuple

from diskcache import Cache


class KVPair(NamedTuple):
    key: Any
    value: Any


class CacheWrapper:
    def __init__(self, path, size):
        self._cache = Cache(path, size_limit=size)

    def update_cache_entries(self, entries: List[KVPair]):
        """
        Update cache entries
        :param entries: a list of pairs key-value
        :return:
        """
        for entry in entries:
            self._cache.set(entry.key, entry.value)  # ALWAYS override!

    def set_entry(self, entry: KVPair):
        self._cache.set(entry.key, entry.value)

    def get_cached_entries(self, keys: List[Any]) -> Tuple[List[Any], List[Any]]:
        """
        Retrieve cached entries
        :param keys: a list of keys to retrieve
        :return: a tuple (<cached results>, <missing entries>)
        """
        to_compute = []
        cached_entries = []

        for key in keys:
            entry = self._cache.get(key)
            if entry is None:
                to_compute.append(key)
            else:
                cached_entries.append(entry)

        return cached_entries, to_compute

    def get_cached_entry(self, key: Any) -> Any:
        return self._cache.get(key)
