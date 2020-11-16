import csv
import pandas as pd

from lookup.services import DBLookup
from generators.utils import AbstractCollector
from factbase_lookup import sortTable, getLabelColumn, getReferenceColumns, \
    getTypes, getDescriptionTokens, containsFact, getMostFrequent, atLeast5, \
    getFirst, search_strict, search_loose
from utils.functions import toList

abc = AbstractCollector()
dblookup = DBLookup()

base_dir = 'T2D_GoldStandard'
target = base_dir + '/T2D_Targets.csv'
table = pd.read_csv(sortTable(target, 'table'))
annotated_table = table.copy()

labelList = []
acceptableTypes = []
descriptionTokens = []
labelColumn = getLabelColumn(table)
referenceColumns = getReferenceColumns(table)

annotationList = ["not annotated"] * table.shape[0]
index, count = 0, 0

for row in range(len(labelColumn)):
    count += index
    index = 0
    allTypes, descTokens, candidateRelations, relations = [], [], [], []
    firstResult = []

    for label in labelColumn[row]:
        labelList.append(label)

        results = dblookup._lookup(labels=[label])[0][1]  # all URIs of a given label

        if len(results) > 0:
            topResult = results[0]
            firstResult.append(results[0])
            allTypes.append(getTypes(topResult))
            descTokens.append(getDescriptionTokens(topResult))
            if len(results) == 1:
                annotationList[count + index] = topResult
                for a, v in referenceColumns[row][index].items():
                    candidateRelations.append(containsFact(topResult, a, v))
        else:
            firstResult.append("not annotated")

        index += 1

    allTypes = toList(allTypes)
    acceptableTypes = getMostFrequent(allTypes, n=5)
    descTokens = toList(descTokens)
    descriptionTokens = getMostFrequent(descTokens)

    candidateRelations = toList(candidateRelations)
    candidateRelations = atLeast5(candidateRelations)

    for attr in referenceColumns[row][index - 1]:
        relations.append(getFirst(candidateRelations, attr))
    relations = toList(relations)

    index = 0

    for label in labelColumn[row]:
        if annotationList[count + index] != "not annotated":
            index += 1
            continue

        results = search_strict(label=label, types=acceptableTypes, description=descriptionTokens)

        if len(results) > 0:
            topResult = results[0]
            annotationList[count + index] = topResult
            index += 1
            continue

        for r in relations:
            results = search_loose(label=label, relation=r[1], value=referenceColumns[row][index][r[0]])
            if len(results) > 0:
                topResult = results[0]
                annotationList[count + index] = topResult
                break

        if annotationList[count + index] == "not annotated" and firstResult[index] != "not annotated":
            label_ = abc.get_label(firstResult[index])

            if len(label_) > 0:
                if label_[0] == label:
                    annotationList[count + index] = firstResult[index]

        index += 1

#annotated_table["label"] = labelList
annotated_table["entity"] = annotationList
annotated_table.to_csv(base_dir + '/lookup/T2D_lookup_6409.csv', quoting=csv.QUOTE_ALL, index=False)
