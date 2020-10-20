import os
from enum import Enum

import pandas as pd

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
            return TT_CATEGORIES
        return {'ALL': ([], [])}

    @classmethod
    def get_test_gt(cls, size, from_gt=None, random=False):
        """
        Helper method to generate a test dataset on-the-fly.
        :param size: dimension of the test dataset to create
        :param from_gt: dataset to sample rows from
        :param random: True if the rows should be sampled randomly; otherwise, the top ``size`` rows are returned.
        :return: a Pandas dataframe
        """
        if from_gt is None:
            from_gt = cls.CEA_Round1
        if random:
            df = from_gt.get_df().sample(size).reset_index()
        else:
            df = from_gt.get_df()[:size]
        tmp = Enum('GTTestEnum', {'%s_TEST_%drows' % (from_gt.name, size): df})  # create a temp enum (name: df)
        setattr(tmp, 'get_df', lambda x: x.value)  # add the get_df function, that returns the df
        setattr(tmp, 'get_table_categories', lambda x: {'ALL': ([], [])})
        return list(tmp)[0]

    @classmethod
    def get_test_tt_by_type(cls, type_, size=0):
        """
        Helper method to generate a test dataset on-the-fly, from the 2T ground truth.
        :param type_: optionally filter entries by type (e.g., dbo:Person).
        :param size: dimension of the test dataset to create
        :return: a Pandas dataframe
        """
        df = cls.CEA_TT.get_df()
        df2 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'CTA_TT.csv'))
        df2 = df2[df2['type'] == type_]
        df = df.merge(df2, on=['table', 'col_id'], how='inner').drop(columns=['type'])
        if size > 0:
            df = df.sample(size).reset_index()

        tmp = Enum('GTTestEnum', {'CEA_TT_%s' % type_[type_.rindex("/") + 1:]: df})
        setattr(tmp, 'get_df', lambda x: x.value)
        setattr(tmp, 'get_table_categories', lambda x: TT_CATEGORIES)
        return list(tmp)[0]
