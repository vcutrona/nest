import csv
import os

import pandas as pd

from generators import Generator
from gt import GTEnum


class Evaluator:

    def __init__(self, generator: Generator):
        self._generator = generator

    def _get_candidates(self, labels, contexts):
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

        dataset = gt.get_df()
        results = self._get_candidates(dataset['label'].values.tolist(), dataset['context'].values.tolist())
        results = {label: " ".join(candidates) for label, candidates in results.items()}
        dataset['candidates'] = dataset['label'].map(results)  # TODO: it works only for simple generators!
        dataset.to_csv(filename, quoting=csv.QUOTE_ALL, index=False)
        return dataset

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

    def score_all(self, exclude=None):
        gts = []
        for gt in GTEnum:
            if exclude and gt in exclude:
                continue
            gts.append(gt)
        return self._score(gts)

    def score(self, gt):
        return self._score([gt])


class SimpleEvaluator(Evaluator):
    def _get_candidates(self, labels, _):
        return self._generator.multi_search(labels)


class ContextEvaluator(Evaluator):
    def _get_candidates(self, labels, contexts):
        return self._generator.multi_search(labels, contexts)

