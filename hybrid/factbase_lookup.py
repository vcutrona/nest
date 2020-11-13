import nltk
from nltk.corpus import stopwords

from data_model.lookup import ESLookupConfig
from lookup.services import DBLookup, ESLookup
from utils.functions import simplify_string, first_sentence
from utils.kgs import DBpediaWrapper


# def sortTable(table, label):
#     """
#     Sort a table by the given label
#
#     :param table: the path of the table to be ordered
#     :param label: the label used to sort
#     :return:
#     """
#     with open(table, 'r', newline='') as f_input:
#         csv_input = csv.DictReader(f_input)
#         data = sorted(csv_input, key=lambda row: (row[label]))
#
#     with open(table, 'w', newline='') as f_output:
#         csv_output = csv.DictWriter(f_output, fieldnames=csv_input.fieldnames)
#         csv_output.writeheader()
#         csv_output.writerows(data)
#
#     return table
#
#
# def getLabelColumn(table):
#     """
#     Get the label's values of many tables
#
#     :param table: the summary table containing the tables to retrieve from
#     :return: a list of list containing the label's values divided by tables
#     """
#
#     labelList = []
#     labelLoL = []
#     for row in range(table.shape[0]):
#         tab_id = table['table'][row]
#         col_id = table['col_id'][row]
#         row_id = table['row_id'][row]
#         tab = pd.read_csv('/home/vincenzo/git-repositories/fast-candidate-selection/datasets/T2D/tables/' + tab_id + '.csv')
#         if row == 0:
#             labelList.append(tab.iloc[row_id - 1][col_id])
#         else:
#             if table['table'][row - 1] == tab_id:
#                 labelList.append(tab.iloc[row_id - 1][col_id])
#             else:
#                 labelLoL.append(labelList)
#                 labelList = []
#                 labelList.append(tab.iloc[row_id - 1][col_id])
#     labelLoL.append(labelList)
#
#     return labelLoL
#
#
# def getReferenceColumns(table):
#     """
#     Get the reference columns' values of many tables
#
#     :param table: the summary table containing the tables to retrieve from
#     :return: a list of list containing a tuple (column name, value) divided by tables
#     """
#     refList = []
#     refLoL = []
#     for row in range(table.shape[0]):
#         tab_id = table['table'][row]
#         col_id = table['col_id'][row]
#         row_id = table['row_id'][row]
#         tab = pd.read_csv('/home/vincenzo/git-repositories/fast-candidate-selection/datasets/T2D/tables/' + tab_id + '.csv')
#         value = list(tab.iloc[row_id - 1])
#         value.remove(tab.iloc[row_id - 1][col_id])
#         key = []
#         for x in tab.columns:
#             key.append(x)
#         key.remove(tab.columns[col_id])
#         dictionary = zip(key, value)
#         if row == 0:
#             refList.append(dict(dictionary))
#         else:
#             if table['table'][row - 1] == tab_id:
#                 refList.append(dict(dictionary))
#             else:
#                 refLoL.append(refList)
#                 refList = []
#                 refList.append(dict(dictionary))
#     refLoL.append(refList)
#
#     return refLoL


def getTypes(uri):
    """
    Get the types' of the given URI

    :param uri: the URI to be read
    :return: a list of types
    """
    abc = DBpediaWrapper()
    toRemove = ['http://www.w3.org/2002/07/owl#Thing']
    result = abc._get_es_docs_by_ids(uri)
    types = []
    for _, doc in result:
        for x in doc['type']:
            if x not in toRemove:
                types.append(x)

    return types


def getDescriptionTokens(uri):
    """
    Get the description tokens' of the given URI

    :param uri: the URI to be read
    :return: a list of keywords
    """
    abc = DBpediaWrapper()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    result = abc._get_es_docs_by_ids(uri)
    stop_words = set(stopwords.words('english'))

    for _, doc in result:
        if len(doc['description']) > 0:
            word = simplify_string(doc['description'][0], dates=False, numbers=False, single_char=False, brackets=True)
            word = first_sentence(word)
            if word is not None:
                word_tokens = tokenizer.tokenize(word.lower())
                return [word for word in word_tokens if word not in stop_words]

    return []


def getMostFrequent(list_, n=1):
    """
    Get the most frequent value(s) in a list

    :param list_: the list to be read
    :param n: number of values to be returned
    :return: a list of the most frequent value(s)
    """
    counter = {}
    for x in list_:
        if x in counter:
            counter[x] += 1
        else:
            counter[x] = 1

    return sorted(counter, key=counter.get, reverse=True)[:n]


def containsFact(uri, a, v):
    """
    In a given URI, retrieve all the relations which its value it is v

    :param uri: the URI to be read
    :param a: the relation's name
    :param v: the value of the relation a
    :return: a list of tuples (a, relation found)
    """
    abc = DBpediaWrapper()
    toRemove = ['http://dbpedia.org/ontology/abstract', 'http://dbpedia.org/ontology/wikiPageWikiLink',
                'http://www.w3.org/2000/01/rdf-schema#comment', 'http://purl.org/dc/terms/subject',
                'http://www.w3.org/2000/01/rdf-schema#label', 'http://www.w3.org/2002/07/owl#Thing']
    relations = abc.get_relations([(uri, v)])
    candidateRelations = []
    for pair, col_relations in relations.items():
        for rel in col_relations:
            if rel not in toRemove:
                candidateRelations.append((a, rel))

    return candidateRelations


def atLeast5(list_):
    """
    Filter a list by removing an element if it is not present at least 5 times

    :param list_: the input list
    :return: a list
    """
    remove = []
    for x in list_:
        if list_.count(x) < 5:
            remove.append(x)
    for rem in remove:
        list_.remove(rem)

    return list_


def getFirst(list_, attr):
    """
    Get the most frequent tuple containing attr as first element in a list of tuples

    :param list_: the input list
    :param attr: the value to check
    :return: a list containing the most frequent tuple
    """
    list2 = list_.copy()
    remove = []
    for x in list_:
        if x[0] != attr:
            remove.append(x)
    for rem in remove:
        list2.remove(rem)

    return getMostFrequent(list2)


def search_strict(label, types, description):
    """
    Execute a search operation on a given label restricting the results to those of an acceptable
    type, having one of the most frequent tokens in their description values

    :param label: the label to look for
    :param types: a list of types
    :param description: a list of keywords
    :return: a list of results
    """
    dblookup = DBLookup()
    results = dblookup._lookup(labels=[label])[0][1]
    removeList = []
    for res in results:
        res_desc = getDescriptionTokens(res)
        res_types = getTypes(res)
        remove = True
        if res_desc is not None:
            for desc in res_desc:
                if desc in description:
                    for type_ in res_types:
                        if type_ in types:
                            remove = False
                            break
        if remove:
            removeList.append(res)

    for rem in removeList:
        results.remove(rem)

    return results


def search_loose(label, relation, value):
    """
    Execute a search operation on a given label allowing a big margin of edit distance (Levenshtein),
    restricting the results to have in their facts at least one of the relation:value

    :param label: the label to look for
    :param relation: a relation
    :param value: a value
    :return: a list of results
    """
    eslookup = ESLookup(ESLookupConfig('titan', 'dbpedia'))
    abc = DBpediaWrapper()

    results = eslookup._lookup(labels=[label])[0][1]

    removeList = []
    for res in results:
        relations = abc.get_relations([(res, value)])
        remove = True
        for pair, col_relations in relations.items():
            if relation in col_relations:
                remove = False

        if remove:
            removeList.append(res)

    for rem in removeList:
        results.remove(rem)

    return results
