import csv
import os
import urllib.parse

import pandas as pd

from generators import Generator, SimpleGenerator, ContextGenerator
from gt import GTEnum


class Evaluator:
    def __init__(self, generator: Generator):
        self._generator = generator

    @staticmethod
    def precision_score(correct_cells, annotated_cells):
        """
        Code from SemTab 2019
        Precision = (# correctly annotated cells) / (# annotated cells)
        :param correct_cells:
        :param annotated_cells:
        :return:
        """
        return float(len(correct_cells)) / len(annotated_cells) if len(annotated_cells) > 0 else 0.0

    @staticmethod
    def recall_score(correct_cells, gt_cell_ent):
        """
        Code from SemTab 2019
        Recall = (# correctly annotated cells) / (# target cells)
        :param correct_cells:
        :param gt_cell_ent:
        :return:
        """
        return float(len(correct_cells)) / len(gt_cell_ent.keys())

    @staticmethod
    def f1_score(precision, recall):
        """
        Code from SemTab 2019
        F1 Score = (2 * Precision * Recall) / (Precision + Recall)
        :param precision:
        :param recall:
        :return:
        """
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def _get_candidates_df(self, labels, contexts):
        raise NotImplementedError

    def _compute(self, gt):
        print('Processing %s on %s' % (self._generator.__class__.__name__, gt.name))

        filename = os.path.join(os.path.dirname(__file__),
                                '%s_%s_candidates.csv' % (self._generator.__class__.__name__, gt.name))

        if os.path.isfile(filename) and isinstance(gt, GTEnum):  # already processed file -> return results (skip tests)
            print('Getting results from %s' % filename)
            return pd.read_csv(filename,
                               dtype={'table': str, 'col_id': int, 'row_id': int, 'label': str,
                                      'context': str, 'entities': str},
                               keep_default_na=False)

        dataset = gt.get_df()
        candidates_df = self._get_candidates_df(dataset['label'].values.tolist(), dataset['context'].values.tolist())
        dataset = dataset.merge(candidates_df, how="left")
        if isinstance(gt, GTEnum):
            dataset.to_csv(filename, quoting=csv.QUOTE_ALL, index=False)
        return dataset

    def _get_scores(self, gt, sub):
        """
        Code from SemTab 2019
        Notes:
        6) Annotations for cells out of the target cells are ignored.
        1) # denotes the number.
        2) F1 Score is used as the primary score; Precision is used as the secondary score.
        3) An empty annotation of a cell will lead to an annotated cell;
           We suggest to exclude the cell with empty annotation in the submission file.
        :param gt: a Pandas Dataframe with the following cols: 'tab_id', 'col_id', 'row_id', 'entity'
        :param sub: a Pandas Dataframe with the following cols: 'tab_id', 'col_id', 'row_id', 'entity'
        :return:
        """
        gt_cell_ent = dict()
        gt_cell_ent_orig = dict()
        for row in gt.itertuples():
            cell = '%s %s %s' % (row.tab_id, row.col_id, row.row_id)
            gt_cell_ent[cell] = urllib.parse.unquote(row.entity).lower().split(' ')
            gt_cell_ent_orig[cell] = row.entity.split(' ')

        correct_cells, annotated_cells = set(), set()
        for row in sub.itertuples():
            cell = '%s %s %s' % (row.tab_id, row.col_id, row.row_id)
            if cell in gt_cell_ent:
                if cell in annotated_cells:
                    raise Exception("Duplicate cells in the submission file")
                else:
                    annotated_cells.add(cell)

                annotation = urllib.parse.unquote(row.entity).lower()
                if annotation in gt_cell_ent[cell]:
                    correct_cells.add(cell)

        precision = self.precision_score(correct_cells, annotated_cells)
        recall = self.recall_score(correct_cells, gt_cell_ent)
        f1 = self.f1_score(precision, recall)

        return {
            'total': gt.shape[0],
            'correct': len(correct_cells),
            'annotated': len(annotated_cells),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _score(self, gts):
        results = {}
        for gt in gts:
            candidate_df = self._compute(gt)
            candidate_df['candidates'] = candidate_df.candidates.str.split(" ", expand=True)[0]  # take the first candidate
            results[gt.name] = self._get_scores(
                candidate_df[['table', 'col_id', 'row_id', 'entities']].rename(columns={"entities": "entity",
                                                                                        'table': "tab_id"}),
                candidate_df[['table', 'col_id', 'row_id', 'candidates']].rename(columns={"candidates": "entity",
                                                                                          'table': "tab_id"}))

        return {self._generator.__class__.__name__: results}

    def score_all(self, exclude=None):
        if exclude is None:
            exclude = []
        return self._score(filter(lambda x: x not in exclude, GTEnum))

    def score(self, gt):
        return self._score([gt])


class SimpleEvaluator(Evaluator):
    def __init__(self, generator: SimpleGenerator):
        super().__init__(generator)

    def _get_candidates_df(self, labels, _):
        return pd.DataFrame([(label, " ".join(candidates))
                             for label, candidates in self._generator.multi_search(labels).items()],
                            columns=["label", "candidates"])


class ContextEvaluator(Evaluator):
    def __init__(self, generator: ContextGenerator):
        super().__init__(generator)

    def _get_candidates_df(self, labels, contexts):
        return pd.DataFrame([((*lc_pair), " ".join(candidates))
                             for lc_pair, candidates in self._generator.multi_search(labels, contexts).items()],
                            columns=["label", "context", "candidates"])
