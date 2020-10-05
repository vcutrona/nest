import pandas as pd

from lookup.services import DBLookup, ESLookup
from generators.utils import AbstractCollector
import factbase_lookup
from utils import functions

abc = AbstractCollector()
dblookup = DBLookup()
eslookup = ESLookup()

base_dir = 'Round 1'
target = base_dir + '/targets/CEA_Round1_Targets11.csv'
factbase_lookup.sortTable(target, 'table')
table = pd.read_csv(target)
annotated_table = table.copy()

labelList = []
acceptableTypes = []
descriptionTokens = []
labelColumn = factbase_lookup.getLabelColumn(table)
referenceColumns = factbase_lookup.getReferenceColumns(table)

annotationList = ["not annotated"] * table.shape[0]
index = 0
count = 0

for row in range(len(labelColumn)):  # n di file
    count += index
    index = 0

    allTypes = []
    descTokens = []
    candidateRelations = []
    relations = []
    for label in labelColumn[row]:
        labelList.append(label)

        results = dblookup._lookup(labels=[label])[0][1]  # all URIs of a given label

        if len(results) > 0:
            topResult = results[0]
            allTypes.append(factbase_lookup.getTypes(topResult))
            descTokens.append(factbase_lookup.getDescriptionTokens(topResult))
            if len(results) == 1:
                annotationList[count + index] = topResult
                for a, v in referenceColumns[row][index].items():
                    candidateRelations.append(factbase_lookup.containsFact(topResult, a, v))

        index += 1

    allTypes = functions.toList(allTypes)
    acceptableTypes = factbase_lookup.getMostFrequent(allTypes, n=5)
    descTokens = functions.toList(descTokens)
    descriptionTokens = factbase_lookup.getMostFrequent(descTokens)

    candidateRelations = functions.toList(candidateRelations)
    candidateRelations = factbase_lookup.atLeast5(candidateRelations)

    for attr in referenceColumns[row][index - 1]:
        relations.append(factbase_lookup.getFirst(candidateRelations, attr))
    relations = functions.toList(relations)

    index = 0

    for label in labelColumn[row]:
        if annotationList[count + index] != "not annotated":
            index += 1
            continue

        results = factbase_lookup.search_strict(label=label, types=acceptableTypes, description=descriptionTokens)

        if len(results) > 0:
            topResult = results[0]
            annotationList[count + index] = topResult
            index += 1
            continue

        for r in relations:
            results = factbase_lookup.search_loose(label=label, relation=r[1], value=referenceColumns[row][index][r[0]])
            results = factbase_lookup.sortByEditDistance(list_=results, label=label)
            if len(results) > 0:
                topResult = results[0]
                annotationList[count + index] = topResult
                break

        index += 1

annotated_table["label"] = labelList
annotated_table["annotation"] = annotationList
annotated_table.to_csv(base_dir + '/gt/CEA_Round1_gt_lookup.csv')
