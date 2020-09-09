import pandas as pd

class Prova():

    def getLabelColumn(T):
      labelList = []
      for row in range(T.shape[0]):
        tab_id = T['table'][row]
        col_id = T['col_id'][row]
        row_id = T['row_id'][row]
        tab = pd.read_csv('/content/drive/My Drive/Round 1' + '/tables/' + tab_id + '.csv')
        labelList.append(tab.iloc[row_id-1][col_id])
      return labelList

    def getReferenceColumns(T):
        refList = []
        for row in range(T.shape[0]):
            tab_id = T['table'][row]
            col_id = T['col_id'][row]
            row_id = T['row_id'][row]
            tab = pd.read_csv('/content/drive/My Drive/Round 1' + '/tables/' + tab_id + '.csv')
            a = list(tab.iloc[row_id - 1])
            a.remove(tab.iloc[row_id - 1][col_id])
            # l = ""
            # for x in a:
            # l+=str(x)+' '
            # refList.append(l.strip())
            refList.append(a)
        return refList

    def getMostFrequent(List, n=1):
        counter = {}
        for x in List:
            if x in counter:
                counter[x] += 1
            else:
                counter[x] = 1
        return sorted(counter, key=counter.get, reverse=True)[:n]