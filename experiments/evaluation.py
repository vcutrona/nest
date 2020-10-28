import os
import pickle
from typing import List, Tuple

from data_model.dataset import Entity, GTTable, Table
from datasets import CEADatasetEnum
from generators import CandidateGenerator


class CEAEvaluator:
    """
    A class to test generator algorithms on a dataset.
    """

    def __init__(self, generator: CandidateGenerator):
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
        return float(len(correct_cells)) / len(gt_cell_ent)

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

    def annotate_dataset(self, dataset: CEADatasetEnum):
        print('Processing %s_%s (%s) on %s' % (*self._generator.id, dataset.name))
        targets = dataset.get_targets()
        # TODO parallelize
        for table in dataset.get_tables():
            if table.tab_id in targets:
                target_cells = targets[table.tab_id]
                filename = os.path.join(
                    os.path.dirname(__file__),
                    'candidates',
                    '%s_%s_%s_%s_%s.pkl' % (*self._generator.id, dataset.name, table.tab_id))
                # check existing results
                if not os.path.isfile(filename):
                    # keep the cell-search_key pair -> results may be shuffled!
                    search_key_cell_dict = {table.get_search_key(cell_): cell_ for cell_ in target_cells}
                    results = self._generator.multi_search(list(search_key_cell_dict.keys()))

                    for search_key, candidates in results:
                        if candidates:
                            table.annotate_cell(search_key_cell_dict[search_key], Entity(candidates[0]))

                    pickle.dump(table, open(filename, 'wb'))

                yield pickle.load(open(filename, 'rb'))

    # def _get_candidates_df(self, gt):
    #     """
    #     This methods executes the generator on the testing dataset.
    #     :param gt: the testing dataset
    #     :return: the testing dataset, with the new column `candidates` appended.
    #     """
    #     print('Processing %s_%s (%s) on %s' % (*self._generator.id, gt.name))
    #
    #     filename = os.path.join(os.path.dirname(__file__),
    #                             'candidates',
    #                             '%s_%s_%s_%s_candidates.csv' % (*self._generator.id, gt.name))
    #
    #     if os.path.isfile(filename) and isinstance(gt, GTEnum):  # already processed file -> return results (skip tests)
    #         print('Getting results from %s' % filename)
    #         return pd.read_csv(filename,
    #                            dtype={'table': str, 'col_id': int, 'row_id': int, 'label': str,
    #                                   'context': str, 'entities': str, 'candidates': str},
    #                            keep_default_na=False)
    #
    #     dataset = gt.get_df()
    #     generator_results = self._generator.multi_search(list(
    #         map(lambda x: SearchKey(*x), zip(dataset['label'].values.tolist(), dataset['context'].values.tolist()))))
    #     candidates_df = pd.DataFrame([" ".join(generator_result.candidates) for generator_result in generator_results],
    #                                  columns=["candidates"])
    #     assert dataset.shape[0] == candidates_df.shape[0]
    #     dataset = dataset.join(candidates_df, how="left")
    #     if isinstance(gt, GTEnum):
    #         dataset.to_csv(filename, quoting=csv.QUOTE_ALL, index=False)
    #     return dataset

    def _get_scores(self, tables: List[Tuple[GTTable, Table]]):
        """
        Code adapted from SemTab 2019
        Notes:
        6) Annotations for cells out of the target cells are ignored.
        1) # denotes the number.
        2) F1 Score is used as the primary score; Precision is used as the secondary score.
        3) An empty annotation of a cell will lead to an annotated cell;
           We suggest to exclude the cell with empty annotation in the submission file.
        :param tables: a list of pair <gt_table, table>
        :return:
        """

        gt_cell_ent, correct_cells, annotated_cells = list(), list(), list()

        for gt_table, table in tables:
            gt_table_ann = gt_table.cell_annotations
            table_ann = table.cell_annotations
            gt_cell_ent += list(gt_table_ann)
            annotated_cells += list(table_ann)
            for cell, cell_annotation in table_ann.items():
                if cell in gt_table_ann and cell_annotation == gt_table_ann[cell]:
                    correct_cells.append(cell)

        precision = self.precision_score(correct_cells, annotated_cells)
        recall = self.recall_score(correct_cells, gt_cell_ent)
        f1 = self.f1_score(precision, recall)

        return {
            'total': len(gt_cell_ent),
            'correct': len(correct_cells),
            'annotated': len(annotated_cells),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # def _get_scores(self, gt, sub):
    #     """
    #     Code from SemTab 2019
    #     Notes:
    #     6) Annotations for cells out of the target cells are ignored.
    #     1) # denotes the number.
    #     2) F1 Score is used as the primary score; Precision is used as the secondary score.
    #     3) An empty annotation of a cell will lead to an annotated cell;
    #        We suggest to exclude the cell with empty annotation in the submission file.
    #     :param gt: a Pandas Dataframe with the following cols: 'tab_id', 'col_id', 'row_id', 'entity'
    #     :param sub: a Pandas Dataframe with the following cols: 'tab_id', 'col_id', 'row_id', 'entity'
    #     :return:
    #     """
    #     gt_cell_ent = dict()
    #     # gt_cell_ent_orig = dict()
    #     for row in gt.itertuples():
    #         cell = '%s %s %s' % (row.tab_id, row.col_id, row.row_id)
    #         gt_cell_ent[cell] = urllib.parse.unquote(row.entity).lower().split(' ')
    #         # gt_cell_ent_orig[cell] = row.entity.split(' ')
    #
    #     correct_cells, annotated_cells = set(), set()
    #     for row in sub.itertuples():
    #         cell = '%s %s %s' % (row.tab_id, row.col_id, row.row_id)
    #         if cell in gt_cell_ent:
    #             if cell in annotated_cells:
    #                 raise Exception("Duplicate cells in the submission file")
    #             else:
    #                 annotated_cells.add(cell)
    #
    #             annotation = urllib.parse.unquote(row.entity).lower()
    #             if annotation in gt_cell_ent[cell]:
    #                 correct_cells.add(cell)
    #
    #     precision = self.precision_score(correct_cells, annotated_cells)
    #     recall = self.recall_score(correct_cells, gt_cell_ent)
    #     f1 = self.f1_score(precision, recall)
    #
    #     return {
    #         'total': gt.shape[0],
    #         'correct': len(correct_cells),
    #         'annotated': len(annotated_cells),
    #         'precision': precision,
    #         'recall': recall,
    #         'f1': f1
    #     }

    @staticmethod
    def _is_table_in_cat(table_id, include, exclude):
        """
        Helper method to filter out tables from a category.
        :param table_id: the id of a table
        :param include: the list of keywords that the table_id must contain
        :param exclude: the list of keywords that the table_id must not contain
        :return: True if the table_id contains all the ```include``` keywords and none of the ``exclude`` keywords,
                 False otherwise.
        """
        for i in include:
            if i not in table_id:
                return False
        for e in exclude:
            if e in table_id:
                return False
        return True

    # def _score(self, gts):
    #     """
    #     Compute Precision, Recall and F1 measures.
    #     :param gts: the list of testing datasets on which to test the generator
    #     :return: a dictionary Dict(generator, results), where results is a dict which contains scores grouped
    #              by tables categories.
    #     """
    #     results = {}
    #     for gt in gts:
    #         candidate_df = self._get_candidates_df(gt)
    #         candidate_df['candidates'] = candidate_df.candidates.str.split(" ", expand=True)[0]  # take first candidate
    #         gt_df = candidate_df[['table', 'col_id', 'row_id', 'entities']].rename(columns={"entities": "entity",
    #                                                                                         'table': "tab_id"})
    #         ann_df = candidate_df[['table', 'col_id', 'row_id', 'candidates']].rename(columns={"candidates": "entity",
    #                                                                                            'table': "tab_id"})
    #         ann_df = ann_df[ann_df['entity'].astype(bool)]
    #
    #         tables_categories = gt.get_table_categories()
    #         results[gt.name] = {}
    #         for cat in tables_categories:
    #             include, exclude = tables_categories[cat]
    #             results[gt.name][cat] = self._get_scores(
    #                 gt_df[gt_df['tab_id'].apply(lambda x: self._is_table_in_cat(x, include, exclude))],
    #                 ann_df[ann_df['tab_id'].apply(lambda x: self._is_table_in_cat(x, include, exclude))])
    #
    #     return {"%s_%s (%s)" % self._generator.id: results}

    def _score(self, datasets):
        """
        Compute Precision, Recall and F1 measures.
        :param datasets: the list of testing datasets on which to test the generator
        :return: a dictionary Dict(generator, results), where results is a dict which contains scores grouped
                 by tables categories.
        """
        results = {}
        for dataset in datasets:
            tables = {gt_table.tab_id: (gt_table, ) for gt_table in dataset.get_gt_tables()}
            for table in self.annotate_dataset(dataset):
                tables[table.tab_id] += (table, )

            tables_categories = dataset.get_table_categories()
            results[dataset.name] = {}
            for cat in tables_categories:
                include, exclude = tables_categories[cat]
                results[dataset.name][cat] = self._get_scores(
                    [tables[tab_id] for tab_id in tables if self._is_table_in_cat(tab_id, include, exclude)])

        return {"%s_%s (%s)" % self._generator.id: results}

    def score_all(self, exclude=None):
        """
        Helper method to test all the datasets available in the benchmark, excluding some of them.
        :param exclude: list of datasets to exclude from the evaluation
        :return: a dictionary Dict(generator, results), where results is a dict which contains scores grouped
                 by tables categories.
        """
        if exclude is None:
            exclude = []
        return self._score(filter(lambda x: x not in exclude, CEADatasetEnum))

    def score(self, gt):
        """
        Helper method to test a single dataset of the benchmark.
        :return: a dictionary Dict(generator, results), where results is a dict which contains scores grouped
                 by tables categories.
        """
        return self._score([gt])
