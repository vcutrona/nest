import pandas as pd

base_dir = '/content/drive/My Drive/Round 1'
table = pd.read_csv(base_dir + '/targets/CEA_Round1_Targets.csv')

annotated_table = table
allTypes = []
descriptionTokens = []
labelColumn = getLabelColumn(table)
referenceColumns = getReferenceColumns(table)