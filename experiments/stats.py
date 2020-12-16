import os

from diskcache import Cache
import numpy as np


class FactBaseStats:
    def __init__(self, generator):
        self._cache = Cache(
            os.path.join(
                os.path.dirname(__file__),
                'stats',
                self.__class__.__name__,
                generator),
            size_limit=int(2e9))
        self._dataset = None
        self._table = None

    def init(self, dataset, table):
        self._dataset = dataset
        self._table = table
        self._cache.set((self._dataset, self._table), {'exact': 0, 'strict': 0, 'loose': 0, 'none': 0})

    def _incr(self, field):
        entry = self._cache.get((self._dataset, self._table))
        entry[field] += 1
        self._cache.set((self._dataset, self._table), entry)

    def incr_exact(self):
        self._incr('exact')

    def incr_strict(self):
        self._incr('strict')

    def incr_loose(self):
        self._incr('loose')

    def incr_empty(self):
        self._incr('none')

    def get_dataset_stats(self, dataset):
        stats_keys = ['exact', 'strict', 'loose', 'none']
        stats = np.zeros(4, dtype=int)
        tables = 0
        for key in self._cache.iterkeys():
            if key[0] == dataset:
                tables += 1
                entry = self._cache.get(key)
                stats[0] += entry['exact']
                stats[1] += entry['strict']
                stats[2] += entry['loose']
                stats[3] += entry['none']
        cells = sum(stats)
        print('Dataset:', dataset)
        print('Tables:', tables)
        print('Cells:', cells)
        print('Stats:', dict(zip(stats_keys, zip(stats, np.round(stats/cells, 4)))))
