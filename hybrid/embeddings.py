import ast
import csv
import pickle
import networkx as nx
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from ngram import NGram
from collections import defaultdict

nltk.download('stopwords')


def create_surfaceFormIndex():
    return


def surfaceFormIndex_to_surfaceToEntities(surface_form):
    # converti surfaceFormIndex (csv) in surfaceToEntities (dict)
    tempDictCandidates = defaultdict(lambda: defaultdict(int))
    with open(surface_form, 'r', newline='', encoding='utf-8') as surface_form_index:
        reader = csv.reader(surface_form_index, delimiter=' ')
        next(reader)

        tempCandidates = [[sf_form.lower(), entity, count] for entity, sf_forms, count in reader
                          for sf_form in ast.literal_eval(sf_forms)]

        surface_form_index.close()

    for sf_form, entity, count in tempCandidates:
        tempDictCandidates[sf_form][entity] = count

    return {sf_form: {entity: count[entity] for entity in count}
            for sf_form, count in tempDictCandidates.items()}


def save(obj, save_to):
    with open(save_to, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()


def load(load_from):
    with open(load_from, 'rb') as file:
        obj = pickle.load(file)
        file.close()
    return obj


def generate_candidates(labels, surface_to_entities, ng):
    entity_candidates = {}
    porter = PorterStemmer()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    stop_words = set(stopwords.words('english'))
    # ng = NGram([surfaceForm for surfaceForm in surface_to_entities.keys()])

    for label in labels:
        graph = nx.DiGraph()
        candidates = []
        if label.lower() in surface_to_entities:
            candidates = [(entity, {'weight': surface_to_entities[label.lower()].get(entity)})
                          for entity in surface_to_entities[label.lower()]]
            # candidates = [entity for entity in surfaceToEntities[label] if entity in w2v]
        # else:  # se non c'e exact matching con label
        #     word_tokens = [porter.stem(word) for word in tokenizer.tokenize(label.lower())]  # stemming
        #     labelMod = [word for word in word_tokens if word not in stop_words]  # stopwords
        #     labelMod = " ".join(labelMod)
        #     listSF = [word for (word, sim) in ng.search(labelMod, 0.82)]  # trigram sim.
        #     candidates = [(entity, {'weight': surface_to_entities[sf_form].get(entity)})
        #                   for sf_form in listSF for entity in surface_to_entities[sf_form]]
        #     # candidates = [entity for sf_form in listSF for entity in surfaceToEntities[sf_form] if entity in w2v]
        graph.add_nodes_from(candidates)
        entity_candidates[label] = graph

    return entity_candidates


def disambiguate_entities(entity_candidates, w2v):
    disambiguation_graph = nx.DiGraph()

    for surface_form in entity_candidates:
        graph = nx.DiGraph()
        # remove node if it is not in load_from(w2v)
        nodes = filterNodes(surface_form, entity_candidates, w2v)
        # take the best 8 candidates for each surface form
        nodes = topCandidates(nodes)

        graph.add_nodes_from(nodes)
        entity_candidates[surface_form] = graph
        disambiguation_graph = nx.compose(disambiguation_graph, entity_candidates.get(surface_form))

    return disambiguation_graph, entity_candidates


def create_subset(entity_candidates, load_from, save_to):
    list_ = []
    with open(load_from, 'r', newline='', encoding='utf-8') as file:
        with open(save_to, 'w', newline='', encoding='utf-8') as filew:
            writer = []
            for line in file:
                words = line.split()
                for sf in entity_candidates:
                    if words[0] in entity_candidates.get(sf) not in list_:
                        list_.append(words[0])
                        writer.append(line)
            filew.write(str(len(writer)) + " 300\n")
            for line in writer:
                filew.write(line)
            filew.close()
        file.close()
    return save_to


def filterNodes(surface_form, entity_candidates, w2v):
    nodes = []
    for (entity, weight) in entity_candidates.get(surface_form).nodes(data=True):
        if entity in w2v:
            nodes.append((entity, {'weight': int(weight['weight'])}))
    return nodes


def topCandidates(nodes, n=8):
    return [(entity, {'weight': int(weight['weight'])})
            for entity, weight in sorted(nodes, key=lambda item: item[1]['weight'], reverse=True)[:n]]


def addEdges(entity_candidates, disambiguation_graph, w2v):
    for surface_form in entity_candidates:
        for v1 in entity_candidates.get(surface_form):
            for v2 in (set(disambiguation_graph.nodes()) - set(entity_candidates.get(surface_form))):
                disambiguation_graph.add_weighted_edges_from([(v1, v2, abs(w2v.similarity(v1, v2)))])
    return disambiguation_graph


def normalizePriors(entity_candidates):
    denominator = {}
    for surface_form in entity_candidates:
        denominator[surface_form] = 0
        for (node, prop) in entity_candidates.get(surface_form).nodes(data=True):
            denominator[surface_form] += prop['weight']
    return denominator


def etp(graph, v1, v2, denominator):
    return graph.get_edge_data(v1, v2)['weight'] / denominator


def thinOut(graph):
    count = int(0.75 * len(graph.edges.data("weight")))
    return sorted(graph.edges.data("weight"), key=lambda tup: tup[2])[:count]


def bestAnnotation(entity_candidates, page_rank):
    annotations = {}
    for surface_form in entity_candidates:
        best = -1
        annotations[surface_form] = "not annotated"
        for entity in entity_candidates.get(surface_form):
            if page_rank[entity] > best:
                best = page_rank[entity]
                annotations[surface_form] = entity

    return annotations.values()


def annotate(table, save_to):
    with open(save_to, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ', quoting=csv.QUOTE_ALL)
        for tab, col, row, annotation in table.itertuples(index=False):
            writer.writerow([tab, col, row, annotation])
        file.close()
