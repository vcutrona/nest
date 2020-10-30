import csv
import ast
import pickle
from ngram import NGram
from utils.functions import chunk_list
import sys
import time
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import networkx as nx
from gensim.models import KeyedVectors
from collections import defaultdict

from factbase_lookup import sortTable, getLabelColumn
from embeddings import etp, thinOut

base_dir = 'T2D_GoldStandard/'

target = base_dir + 'targets/T2D_embedding.csv'
table = open(sortTable(target, 'table'), 'r', newline='')
# annotated_table = table.copy()
# w2v = KeyedVectors.load_word2vec_format(base_dir + 'dbpedia_embeddings-skipgram-300/embedding.txt', binary=False)

porter = PorterStemmer()
tokenizer = nltk.RegexpTokenizer(r"\w+")
stop_words = set(stopwords.words('english'))
tempDictCandidates = defaultdict(lambda: defaultdict(int))
entity_candidates = {}
annotation = {}

# converti surfaceFormIndex (csv) in labelToEntities (dict)
"""with open(base_dir + 'surfaceFormIndex.csv', 'r', newline='', encoding='utf-8') as surface_form_index:
    reader = csv.reader(surface_form_index, delimiter=' ')
    next(reader)

    tempCandidates = [[sf_form, entity, count] for entity, sf_forms, count in reader
                      for sf_form in ast.literal_eval(sf_forms)]
    print("1")
    surface_form_index.close()

for sf_form, entity, count in tempCandidates:
    tempDictCandidates[sf_form][entity] = count
tempCandidates = ['ne']  # void to release memory
print("2")
labelToEntities = {sf_form: {entity: count[entity] for entity in count}
                   for sf_form, count in tempDictCandidates.items()}
print("3")
tempDictCandidates = {'a': 2} # void to release memory
with open(base_dir + 'labelToEntities.pickle', 'wb') as file:
    pickle.dump(labelToEntities, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()"""



#carica labelToEntities
with open(base_dir + 'labelToEntities.pickle', 'rb') as file:
    labelToEntities = pickle.load(file)
    file.close()

#sottoinsieme di labelToEntities
"""labelToEnt = {}
count = 0
for k, x in labelToEntities.items():
    count += 1
    if count <= 1000000:
        labelToEnt[k] = x
    else:
        break
labelToEntities = labelToEnt"""

#recupera labels e crea modello ngram
labelColumn = getLabelColumn(table)
flat_list = [d for d in labelToEntities.keys()]
ng = NGram(flat_list)

for row in range(len(labelColumn)):
    disambiguation_graph = nx.Graph()
    # genera candidati
    with open('candidates.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=' ', quoting=csv.QUOTE_ALL)
        for label in labelColumn[row]:
            graph = nx.DiGraph()
            candidates = []
            inner = time.time()
            if label in labelToEntities:
                candidates = [entity for entity in labelToEntities[label]]
            """else: #se non c'è exact matching con label
                word_tokens = tokenizer.tokenize(label.lower())
                word_tokens = [porter.stem(word) for word in word_tokens] #stemming
                labelMod = [word for word in word_tokens if word not in stop_words] #stopwords
                labelMod = " ".join(labelMod)
                listSF = [word for (word, sim) in ng.search(labelMod, 0.82)] #trigram sim.
                candidates = [entity for sf_form in listSF for entity in labelToEntities[sf_form]]"""
            writer.writerow([{label: candidates}])
        file.close()

        graph.add_nodes_from(candidates)
        entity_candidates[label] = graph

    # crea grafo disambiguazione
    for surface_form in entity_candidates:
        disambiguation_graph = nx.compose(disambiguation_graph, entity_candidates.get(surface_form))
    # crea archi nel grafo
    for surface_form in entity_candidates:
        for v1 in entity_candidates.get(surface_form):
            for v2 in (set(disambiguation_graph.nodes()) - set(entity_candidates.get(surface_form))):
                disambiguation_graph.add_weighted_edges_from([(v1, v2, abs(w2v.similarity(v1, v2)))])

    denominator = {}
    for label in entity_candidates:
        denominator[label] = 0
        for (node, prop) in entity_candidates.get(label).nodes(data=True):
            denominator[label] += prop['weight']

    personalization = {node: prop['weight'] / denominator[label] for label in entity_candidates
                       for (node, prop) in entity_candidates.get(label).nodes(data=True)}
    # calcola denominatore
    """denominator = {}
    for v1 in disambiguation_graph.nodes():
        denominator[v1] = 0
        for v2 in disambiguation_graph.nodes():
            if disambiguation_graph.get_edge_data(v1, v2) is not None:
                denominator[v1] += disambiguation_graph.get_edge_data(v1, v2)['weight']"""

    # aggiorna pesi archi
    """for v1 in disambiguation_graph.nodes():
        for v2 in disambiguation_graph.nodes():
            if disambiguation_graph.get_edge_data(v1, v2) is not None:
                disambiguation_graph.add_weighted_edges_from(
                    [(v1, v2, etp(disambiguation_graph, v1, v2, denominator[v1]))])"""

    # thin out 25% of edges
    disambiguation_graph.remove_edges_from(thinOut(disambiguation_graph))

    #pagerank
    pageRank = nx.pagerank(disambiguation_graph, max_iter=50, alpha=0.9, personalization=personalization)

    # pick the best candidate for each surface form
    for surface_form in entity_candidates:
        best = -1
        for v in entity_candidates.get(surface_form):
            if pageRank[v] > best:
                best = pageRank[v]
                annotation[surface_form] = v
