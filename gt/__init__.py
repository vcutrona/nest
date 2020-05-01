import os
from enum import Enum

import pandas as pd


class GTEnum(Enum):
    TEST = 'TEST.csv'
    CEA_ROUND1 = 'CEA_Round1.csv'
    CEA_ROUND2 = 'CEA_Round2.csv'
    CEA_ROUND3 = 'CEA_Round3.csv'
    CEA_ROUND4 = 'CEA_Round4.csv'

    def get_df(self):
        return pd.read_csv(os.path.join(os.path.dirname(__file__), self.value), dtype=str, keep_default_na=False)
