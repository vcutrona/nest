import os
from enum import Enum

import pandas as pd

from data_model.dataset import Table

TT_CATEGORIES = {'ALL': ([], []),
                 'CTRL_WIKI': (['WIKI'], ['NOISE2']),
                 'CTRL_DBP': (['CTRL', 'DBP'], ['NOISE2']),
                 'CTRL_NOISE2': (['CTRL', 'NOISE2'], []),
                 'TOUGH_T2D': (['T2D'], ['NOISE2']),
                 'TOUGH_HOMO': (['HOMO'], ['SORTED', 'NOISE2']),
                 'TOUGH_MISC': (['MISC'], ['NOISE2']),
                 'TOUGH_MISSP': (['MISSP'], ['NOISE1', 'NOISE2']),
                 'TOUGH_SORTED': (['SORTED'], ['NOISE2']),
                 'TOUGH_NOISE1': (['NOISE1'], []),
                 'TOUGH_NOISE2': (['TOUGH', 'NOISE2'], [])
                 }


class DatasetEnum(Enum):
    """
    Enumerate the datasets available in the benchmark
    """
    ST19_Round1 = 'Round1'
    ST19_Round2 = 'Round2'
    ST19_Round3 = 'Round3'
    ST19_Round4 = 'Round4'
    TT = '2T'
    T2D = 'T2D'
    T2D_sub = 'T2D_subset'

    def _target_path(self, task):
        return f"{os.path.dirname(__file__)}/{self.value}/targets/{task}_{self.value}_Targets.csv"

    def _gt_path(self, task):
        return f"{os.path.dirname(__file__)}/{self.value}/gt/{task}_{self.value}_gt.csv"

    def _table_path(self, tab_id):
        return f"{os.path.dirname(__file__)}/{self.value}/tables/{tab_id}.csv"

    def get_tables(self):
        cea = pd.read_csv(self._gt_path('CEA'),
                          names=['tab_id', 'col_id', 'row_id', 'entities'],
                          dtype={'tab_id': str, 'col_id': int, 'row_id': int, 'entities': str})
        cea['entities'] = cea['entities'].apply(str.split)
        cta_groups = None
        if os.path.exists(self._gt_path('CTA')):
            cta = pd.read_csv(self._gt_path('CTA'),
                              names=['tab_id', 'col_id', 'types'],
                              dtype={'tab_id': str, 'col_id': int, 'types': str})
            cta['types'] = cta['types'].apply(str.split)
            cta_groups = cta.groupby('tab_id')
        cpa_groups = None
        if os.path.exists(self._gt_path('CPA')):
            cpa = pd.read_csv(self._gt_path('CPA'),
                              names=['tab_id', 'source_id', 'target_id', 'properties'],
                              dtype={'tab_id': str, 'source_id': int, 'target_id': int, 'properties': str})
            cpa['properties'] = cpa['properties'].apply(str.split)
            cpa_groups = cpa.groupby('tab_id')

        cea_groups = cea.groupby('tab_id')
        for tab_id, cea_group in cea_groups:
            table = Table(tab_id, self.value, self._table_path(tab_id))
            table.set_gt_cell_annotations(zip(cea_group['row_id'], cea_group['col_id'], cea_group['entities']))
            if cta_groups and tab_id in cta_groups.groups:
                cta_group = cta_groups.get_group(tab_id)
                table.set_gt_column_annotations(zip(cta_group['col_id'], cta_group['types']))
            if cpa_groups and tab_id in cpa_groups.groups:
                cpa_group = cpa_groups.get_group(tab_id)
                table.set_gt_property_annotations(zip(cpa_group['source_id'],
                                                      cpa_group['target_id'],
                                                      cpa_group['properties']))

            yield table

    def get_table_categories(self):
        """
        Return the set of relevant categories for a specific dataset.
        For each category, a pair of keywords (to_include, to_exclude) is provided to
        filter the rows relevant to the category.
        :return: a dictionary Dict(category, Tuple(to_include, to_exclude))
        """
        if self == self.TT:
            return TT_CATEGORIES
        return {'ALL': ([], [])}

    @classmethod
    def get_test_dataset(cls, size, from_dataset=None, rand=False):
        """
        Helper method to generate a test dataset on-the-fly.
        :param size: dimension of the test dataset to create (# cells)
        :param from_dataset: dataset to sample rows from. Default: Round1
        :param rand: True if the rows should be sampled randomly; otherwise, the top ``size`` rows are returned.
        :return: a Pandas dataframe
        """
        if from_dataset is None:
            from_dataset = cls.ST19_Round1
        cea = pd.read_csv(from_dataset._gt_path('CEA'),
                          names=['tab_id', 'col_id', 'row_id', 'entities'],
                          dtype={'tab_id': str, 'col_id': int, 'row_id': int, 'entities': str})
        if rand:
            cea = cea.sample(size).reset_index()
        else:
            cea = cea[:size]

        cta_groups = None
        if os.path.exists(from_dataset._gt_path('CTA')):
            cta = pd.read_csv(from_dataset._gt_path('CTA'),
                              names=['tab_id', 'col_id', 'types'],
                              dtype={'tab_id': str, 'col_id': int, 'types': str})
            cta['types'] = cta['types'].apply(str.split)
            cta_groups = cta.groupby('tab_id')
        cpa_groups = None
        if os.path.exists(from_dataset._gt_path('CPA')):
            cpa = pd.read_csv(from_dataset._gt_path('CPA'),
                              names=['tab_id', 'source_id', 'target_id', 'properties'],
                              dtype={'tab_id': str, 'source_id': int, 'target_id': int, 'properties': str})
            cpa['properties'] = cpa['properties'].apply(str.split)
            cpa_groups = cpa.groupby('tab_id')

        cea_groups = cea.groupby('tab_id')
        tables = []
        for tab_id, cea_group in cea_groups:
            table = Table(tab_id, from_dataset.value, from_dataset._table_path(tab_id))
            table.set_gt_cell_annotations(zip(cea_group['row_id'], cea_group['col_id'], cea_group['entities']))
            if cta_groups and tab_id in cta_groups.groups:
                cta_group = cta_groups.get_group(tab_id)
                cta_group = cta_group[cta_group['col_id'].isin(cea_group['col_id'].unique())]
                table.set_gt_column_annotations(zip(cta_group['col_id'], cta_group['types']))
            if cpa_groups and tab_id in cpa_groups.groups:
                cpa_group = cpa_groups.get_group(tab_id)
                cpa_group = cpa_group[(cpa_group['source_id'].isin(cea_group['col_id'].unique()))
                                      & (cpa_group['target_id'].isin(cea_group['col_id'].unique()))]
                table.set_gt_property_annotations(zip(cpa_group['source_id'],
                                                      cpa_group['target_id'],
                                                      cpa_group['properties']))

            tables.append(table)

        tmp = Enum('GTTestEnum', {'%s_TEST_%d' % (from_dataset.name, size): tables})  # create a temp enum
        setattr(tmp, 'get_tables', lambda x: x.value)  # add the get_df function, that returns the tables
        setattr(tmp, 'get_table_categories', lambda x: from_dataset.get_table_categories())
        return list(tmp)[0]
