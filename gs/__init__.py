import os
from enum import Enum

import pandas as pd


class GSEnum(Enum):
    CEA_ROUND1 = 'CEA_Round1.csv'

    def get_df(self):
        return pd.read_csv(os.path.join(os.path.dirname(__file__), self.value), dtype=str, keep_default_na=False)
