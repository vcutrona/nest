import os
import pickle

from tqdm.contrib.concurrent import process_map

from data_model.kgs import Entity
from datasets import DatasetEnum
from factbase_lookup import getTypes, getDescriptionTokens, containsFact, getMostFrequent, atLeast5, \
    getFirst, search_strict, search_loose
from lookup.services import DBLookup
from utils.kgs import DBpediaWrapper


# target = '/home/vincenzo/git-repositories/fast-candidate-selection/datasets/T2D/targets/CEA_T2D_Targets.csv'
# table = pd.read_csv(target, names=['table', 'col_id', 'row_id']).sort_values(by='table')
# annotated_table = table.copy()

# labelList = []
# acceptableTypes = []
# descriptionTokens = []
# labelColumn = getLabelColumn(table)
# referenceColumns = getReferenceColumns(table)

# annotationList = ["not annotated"] * table.shape[0]
# index, count = 0, 0


def annotate_table(table):
    # for table in tables:
    # count += index
    # index = 0
    table_filename = f"./gianluca/{table.tab_id}.pkl"
    if not os.path.exists(table_filename):
        abc = DBpediaWrapper()
        dblookup = DBLookup()

        allTypes, descTokens, candidateRelations, relations = [], [], [], []
        firstResult = {}

        search_key_cell_dict = {table.get_search_key(cell_): cell_ for cell_ in table.get_gt_cells()}

        for search_key, cell in search_key_cell_dict.items():
            label = search_key.label
            # labelList.append(label)

            results = dblookup._lookup(labels=[label])[0][1]  # all URIs of a given label

            if len(results) > 0:
                topResult = results[0]
                firstResult[cell] = results[0]
                allTypes += getTypes(topResult)
                descTokens += getDescriptionTokens(topResult)
                if len(results) == 1:
                    table.annotate_cell(cell, Entity(topResult))
                    for a, v in search_key.context:
                        # for a, v in referenceColumns[row][index].items():
                        candidateRelations += containsFact(topResult, a, v)
            # else:
            #     firstResult.append("not annotated")
            #
            # index += 1

        acceptableTypes = getMostFrequent(allTypes, n=5)
        descriptionTokens = getMostFrequent(descTokens)

        candidateRelations = atLeast5(candidateRelations)

        # for attr in referenceColumns[row][index - 1]:
        for attr in table.get_search_key(table.get_gt_cells()[0]):
            relations += getFirst(candidateRelations, attr)

        # index = 0

        # for label in labelColumn[row]:
        for search_key, cell in search_key_cell_dict.items():
            # if annotationList[count + index] != "not annotated":
            if search_key in table.cell_annotations:
                # index += 1
                continue
            label = search_key.label
            results = search_strict(label=label, types=acceptableTypes, description=descriptionTokens)

            if len(results) > 0:
                topResult = results[0]
                table.annotate_cell(cell, Entity(topResult))
                # annotationList[count + index] = topResult
                # index += 1
                continue

            for r in relations:
                # results = search_loose(label=label, relation=r[1], value=referenceColumns[row][index][r[0]])
                results = search_loose(label=label, relation=r[1], value=search_key.context[r[0]])
                if len(results) > 0:
                    topResult = results[0]
                    # annotationList[count + index] = topResult
                    table.annotate_cell(cell, Entity(topResult))
                    break

            # if annotationList[count + index] == "not annotated" and firstResult[index] != "not annotated":
            if search_key not in table.cell_annotations and cell in firstResult:
                label_ = abc.get_labels_for_uris([firstResult[cell]])[firstResult[cell]]

                if len(label_) > 0:
                    if label_[0] == label:
                        table.annotate_cell(cell, Entity(firstResult[cell]))

            # index += 1

        pickle.dump(table, open(table_filename, 'wb'))

    return pickle.load(open(table_filename, 'rb'))


ann_tables = process_map(annotate_table,
                         list(DatasetEnum.T2D.get_tables()),
                         max_workers=6)

# annotated_table["label"] = labelList
# annotated_table["entity"] = annotationList
# annotated_table.to_csv('./T2D_lookup.csv', quoting=csv.QUOTE_ALL, index=False)
pickle.dump(ann_tables, open('gianluca.pickle', 'wb'))
