from os import listdir

import pandas as pd
import csv

folder_entities = "entities_instance/"
folder_tables = "t2d_tables_instance/"
target = pd.DataFrame(columns=['table', 'col_id', 'row_id', 'entity'])

for file in listdir(folder_entities):
    entity = pd.read_csv(folder_entities + file,
                         names=['entity', 'label', 'row_id'],
                         dtype={'entity': str, 'label': str, 'row_id': int},
                         header=None)
    table = pd.read_csv(folder_tables + file, header=None)

    candidates = []
    index = 0

    while len(candidates) != 1:
        candidates = []
        for row in table.itertuples():
            n_col = 0
            for x in row:
                if x == entity['label'][index]:
                    candidates.append(n_col)
                n_col += 1

        candidates = list(set(candidates))
        index += 1

    column = candidates[0] - 1

    for row in range(entity.shape[0]):
        target = target.append({'table': file.replace('.csv', ''),
                                'col_id': column,
                                'row_id': entity['row_id'][row],
                                'entity': entity['entity'][row]},
                               ignore_index=True)

target.to_csv('T2D_Targets.csv', quoting=csv.QUOTE_ALL, index=False)
