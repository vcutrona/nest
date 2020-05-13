import os
from enum import Enum

import pandas as pd


class GTEnum(Enum):
    CEA_Round1 = 'CEA_Round1.csv'
    CEA_Round2 = 'CEA_Round2.csv'
    CEA_Round3 = 'CEA_Round3.csv'
    CEA_Round4 = 'CEA_Round4.csv'

    def get_df(self):
        return pd.read_csv(os.path.join(os.path.dirname(__file__), self.value),
                           dtype={'table': str, 'col_id': int, 'row_id': int, 'label': str,
                                  'context': str, 'entities': str},
                           keep_default_na=False)

    @classmethod
    def get_test_gt(cls, size, from_gt=None, random=False):
        if from_gt is None:
            from_gt = cls.CEA_Round1
        if random:
            df = from_gt.get_df().sample(size)
        else:
            df = from_gt.get_df()[:size]
        tmp = Enum('GTTestEnum', {'%s_TEST_%drows' % (from_gt.name, size): df})  # create a temp enum (name: df)
        setattr(tmp, 'get_df', lambda x: x.value)  # add the get_df function, that returns the df
        return list(tmp)[0]
