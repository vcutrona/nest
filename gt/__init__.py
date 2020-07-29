import os
from enum import Enum

import pandas as pd


class GTEnum(Enum):
    """
    Enumerate the datasets available in the benchmark
    """
    CEA_Round1 = 'CEA_Round1.csv'
    CEA_Round2 = 'CEA_Round2.csv'
    CEA_Round3 = 'CEA_Round3.csv'
    CEA_Round4 = 'CEA_Round4.csv'
    CEA_TT = 'CEA_TT.csv'

    def get_df(self):
        """
        Load the dataset in a Pandas dataframe
        :return: a dataframe
        """
        return pd.read_csv(os.path.join(os.path.dirname(__file__), self.value),
                           dtype={'table': str, 'col_id': int, 'row_id': int, 'label': str,
                                  'context': str, 'entities': str},
                           keep_default_na=False)

    def get_table_categories(self):
        """
        Return the set of relevant categories for a specific dataset.
        For each category, a pair of keywords (to_include, to_exclude) is provided to
        filter the rows relevant to the category.
        :return: a dictionary Dict(category, Tuple(to_include, to_exclude))
        """
        if self == self.CEA_TT:
            return {
                'ALL': ([], []),
                'CTRL_WIKI': (['WIKI'], ['NOISE2']),
                'CTRL_DBP': (['CTRL', 'DBP'], ['NOISE2']),
                'CTRL_NOISE2': (['CTRL', 'NOISE2'], []),
                'TOUGH_T2D': (['T2D'], ['NOISE2']),
                'TOUGH_HOMO': (['HOMO'], ['SORTED', 'NOISE2']),
                'TOUGH_OD': (['OD'], ['NOISE2']),
                'TOUGH_MISSP': (['MISSP'], ['NOISE1', 'NOISE2']),
                'TOUGH_SORTED': (['SORTED'], ['NOISE2']),
                'TOUGH_NOISE1': (['NOISE1'], []),
                'TOUGH_NOISE2': (['TOUGH', 'NOISE2'], [])
            }
        return {'ALL': ([], [])}

    @classmethod
    def get_test_gt(cls, size, from_gt=None, random=False):
        """
        Helper method to generate a test dataset on-the-fly.
        :param size: dimension of the test dataset to create
        :param from_gt: dataset to sample rows from
        :param random: True if the rows should be sampled randomly; otherwise, the first ``size`` are returned.
        :return: a Pandas dataframe
        """
        if from_gt is None:
            from_gt = cls.CEA_Round1
        if random:
            df = from_gt.get_df().sample(size)
        else:
            df = from_gt.get_df()[:size]
        tmp = Enum('GTTestEnum', {'%s_TEST_%drows' % (from_gt.name, size): df})  # create a temp enum (name: df)
        setattr(tmp, 'get_df', lambda x: x.value)  # add the get_df function, that returns the df
        setattr(tmp, 'get_table_categories', lambda x: {'ALL': ([], [])})
        return list(tmp)[0]
