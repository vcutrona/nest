import csv
import os

import pandas as pd
from diskcache import Cache

from generators import Generator, SimpleGenerator, ContextGenerator
from gt import GTEnum

import swifter  # DO NOT REMOVE!

cache = Cache('cache')


class Evaluator:

    def __init__(self, generator: Generator):
        self._generator = generator

    def _get_candidates(self, row_tuple):
        raise NotImplementedError

    def _compute(self, gt):
        print('Processing %s on %s' % (self._generator.__class__.__name__, gt.name))

        filename = os.path.join(os.path.dirname(__file__),
                                '%s_%s_candidates.csv' % (self._generator.__class__.__name__, gt.name))

        if os.path.isfile(filename):  # already processed file -> return results
            print('Got results from %s' % filename)
            return pd.read_csv(filename,
                               dtype={'table': str, 'col_id': int, 'row_id': int, 'label': str,
                                      'context': str, 'entities': str},
                               keep_default_na=False)

        # init execution  - CODE BLOCK USEFUL FOR RESUMING OLD EXECUTIONS - KEEP IT
        # dataset = pd.DataFrame(columns=['table', 'col_id', 'row_id', 'label', 'context', 'entities'])
        # gt_df = gt.get_df()
        # gt_original_size = gt_df.shape[0]
        # tmp_filename = os.path.join(os.path.dirname(__file__),
        #                             '%s_%s_candidates.checkpoint.csv' % (self._generator.__class__.__name__, gt.name))
        #
        # if os.path.isfile(tmp_filename):  # but if checkpoint file -> resume
        #     logger.info('Resuming from %s ...' % tmp_filename)
        #     dataset = pd.read_csv(tmp_filename,
        #                           dtype={'table': str, 'col_id': int, 'row_id': int, 'label': str,
        #                                  'context': str, 'entities': str},
        #                           keep_default_na=False)
        #     gt_df = pd.merge(gt_df, dataset, on=['table', 'col_id', 'row_id'], how='left', suffixes=['', '_r'])
        #     gt_df = gt_df[gt_df.entities_r.isnull()][['table', 'col_id', 'row_id', 'label', 'context', 'entities']]
        #     assert gt_original_size - dataset.shape[0] == gt_df.shape[0]
        #
        # logger.info('Original GT size: %d' % gt_original_size)
        # logger.info('Already processed lines: %d' % dataset.shape[0])
        # logger.info('Lines to process: %d' % gt_df.shape[0])
        #
        # for tuple_row in gt_df.itertuples(index=False):
        #     new_row = {
        #         'table': tuple_row.table,
        #         'col_id': tuple_row.col_id,
        #         'row_id': tuple_row.row_id,
        #         'label': tuple_row.label,
        #         'context': tuple_row.context,
        #         'entities': tuple_row.entities,
        #         'candidates': " ".join(self._get_candidates(tuple_row))
        #     }
        #
        #     dataset = dataset.append(new_row, ignore_index=True)
        #
        #     if dataset.shape[0] % 10000 == 0:  # save a checkpoint every 10K rows
        #         dataset.to_csv(tmp_filename, quoting=csv.QUOTE_ALL, index=False)
        #         logger.info('Saving checkpoint... (%.2f%%)' % (dataset.shape[0] / gt_original_size * 100))

        dataset = gt.get_df()
        dataset['candidates'] = dataset.swifter.apply(self._get_candidates, axis=1)
        dataset['candidates'] = dataset['candidates'].swifter.apply(lambda x: ' '.join(map(str, x)))
        dataset.to_csv(filename, quoting=csv.QUOTE_ALL, index=False)
        print('Actual cache size: %.2f MB' % (cache.volume() / 1024 / 1024))
        return dataset

    def score_all(self, exclude=None):
        gts = []
        for gt in GTEnum:
            if exclude and gt in exclude:
                continue
            gts.append(gt)
        return self._score(gts)

    def score(self, gt):
        return self._score([gt])

    def _score(self, gts):
        results = {}
        for gt in gts:
            candidate_df = self._compute(gt)
            total = candidate_df.shape[0]

            correct = 0
            missing = 0

            for tuple_row in candidate_df.itertuples():
                if not tuple_row.candidates:
                    missing = missing + 1
                elif tuple_row.candidates.split()[0] in tuple_row.entities.split():
                    correct = correct + 1

            results[gt.name] = {'correct': correct,
                                'missing': missing,
                                'wrong': total - correct - missing,
                                'P': correct / total}

        return {self._generator.__class__.__name__: results}


class SimpleEvaluator(Evaluator):
    def __init__(self, generator: SimpleGenerator):
        super().__init__(generator)

    def _get_candidates(self, row_tuple):
        k = (row_tuple.label, self._generator.__class__.__name__)
        result = self._generator.search(label=row_tuple.label)
        cache.set(k, result)
        return cache[k]


class SimpleCachedEvaluator(SimpleEvaluator):
    def _get_candidates(self, row_tuple):
        k = (row_tuple.label, self._generator.__class__.__name__)
        result = cache.get(k)
        if result is None:
            cache.add(k, self._generator.search(label=row_tuple.label))
        return cache[k]


class ContextEvaluator(Evaluator):
    def __init__(self, generator: ContextGenerator):
        super().__init__(generator)

    def _get_candidates(self, row_tuple):
        k = (row_tuple.label, row_tuple.context, self._generator.__class__.__name__)
        result = self._generator.search(label=row_tuple.label, context=row_tuple.context)
        cache.set(k, result)
        return cache[k]


class ContextCachedEvaluator(ContextEvaluator):
    def _get_candidates(self, row_tuple):
        k = (row_tuple.label, row_tuple.context, self._generator.__class__.__name__)
        result = cache.get(k)
        if result is None:
            cache.add(k, self._generator.search(label=row_tuple.label, context=row_tuple.context))
        return cache[k]
