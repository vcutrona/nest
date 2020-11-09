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


def sfi_to_ste(surface_form_index):
    """
    Convert a surface_form_index in a more readable and quickly accessible surface_to_entities.
    High speeding of candidates generation.

    e.g.:
        surfaceFormIndex = <http://dbpedia.org/resource/Italy> ['italy', 'italia'] 554
                            <http://dbpedia.org/resource/Roma> ['italy', 'italia', 'roma'] 266
        surfaceToEntities = {'italy': {'<http://dbpedia.org/resource/Italy>': 554,
                                        '<http://dbpedia.org/resource/Roma>': 266}
                            'italia': {'<http://dbpedia.org/resource/Italy>': 554,
                                        '<http://dbpedia.org/resource/Roma>': 266}
                            'roma':   {'<http://dbpedia.org/resource/Roma>': 266}}

    :param surface_form_index: a csv surface_form_index
    :return: a dict surface_to_entities
    """
    dict_candidates = defaultdict(lambda: defaultdict(int))
    with open(surface_form_index, 'r', newline='', encoding='utf-8') as surface_form_index_file:
        reader = csv.reader(surface_form_index_file, delimiter=' ')
        next(reader)

        candidates = [[sf_form.lower(), entity, count] for entity, sf_forms, count in reader
                      for sf_form in ast.literal_eval(sf_forms)]

        surface_form_index_file.close()

    for sf_form, entity, count in candidates:
        dict_candidates[sf_form][entity] = count

    return {sf_form: {entity: count[entity] for entity in count}
            for sf_form, count in dict_candidates.items()}


def save(instance, save_to):
    """
    Save an instance of an object on disk as save_to

    :param instance: the instance to save
    :param save_to: the path where save the instance
    :return:
    """
    with open(save_to, 'wb') as file:
        pickle.dump(instance, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()


def load(load_from):
    """
    Load an instance of an object from load_from

    :param load_from: the path from where to load
    :return: the instance
    """
    with open(load_from, 'rb') as file:
        instance = pickle.load(file)
        file.close()
    return instance


def generate_candidates(labels, surface_to_entities, ng):
    """
    Generate possible candidates for all the labels.
    It compares the labels to those stored in the surface_to_entities dict. All entities in the surface_to_entities
    that provide an exact matching with the label serve as entity candidate. If no candidates are retrieved it
    computes stemming and stopwords removal on the label and it compares the new label to those stored in the
    surface_to_entities dict by applying trigram similarity; all entities that overcome the threshold serve as entity
    candidates.
    entity_candidates is a dict where keys k are tuple (index, sf_form) and values are graph which nodes are candidates.
    Maintaining k different graph is very useful to easily create a disambiguation graph in the disambiguation step.

    :param labels: a list of labels
    :param surface_to_entities: the surface_to_entities dict
    :param ng: the ngram model
    :return: an entity_candidates dict
    """
    entity_candidates = {}
    porter = PorterStemmer()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    stop_words = set(stopwords.words('english'))

    index = 0  # index needed since label might be repeated
    for label in labels:
        graph = nx.DiGraph()
        if label.lower() in surface_to_entities:
            candidates = [(entity, {'weight': int(surface_to_entities[label.lower()].get(entity))})
                          for entity in surface_to_entities[label.lower()]]
        else:  # se non c'e exact matching con label
            word_tokens = [porter.stem(word) for word in tokenizer.tokenize(label.lower())]  # stemming
            label_mod = [word for word in word_tokens if word not in stop_words]  # stopwords
            label_mod = " ".join(label_mod)
            listSF = [word for (word, sim) in ng.search(label_mod, 0.82)]  # trigram sim.
            candidates = [(entity, {'weight': surface_to_entities[sf_form].get(entity)})
                          for sf_form in listSF for entity in surface_to_entities[sf_form]]

        graph.add_nodes_from(candidates)
        entity_candidates[(index, label)] = graph  # index needed since label might be repeated
        index += 1

    return entity_candidates


def create_subset(entity_candidates, load_from, save_to):
    """
    TO DELETE

    :param entity_candidates:
    :param load_from:
    :param save_to:
    :return:
    """
    list_ = []
    with open(load_from, 'r', newline='', encoding='utf-8') as file:
        with open(save_to, 'w', newline='', encoding='utf-8') as filew:
            writer = []
            for line in file:
                words = line.split()
                for (index, sf) in entity_candidates:
                    if words[0] in entity_candidates.get((index, sf)) not in list_:
                        list_.append(words[0])
                        writer.append(line)
            filew.write(str(len(writer)) + " 300\n")
            for line in writer:
                filew.write(line)
            filew.close()
        file.close()
    return save_to


def disambiguate_entities(entity_candidates, w2v):
    """
    Create a complete directed k-partite disambiguation graph where k is the number of surface forms.
    Since the number of entity candidates could be extremely high, a filtering is applied.

    :param entity_candidates: the entity_candidates dict
    :param w2v: an embeddings model
    :return: the disambiguation graph, the entity_candidates updated after the filtering
    """
    disambiguation_graph = nx.DiGraph()

    for (index, surface_form) in entity_candidates:
        graph = nx.DiGraph()
        nodes = filter_candidates((index, surface_form), entity_candidates, w2v)
        graph.add_nodes_from(nodes)
        entity_candidates[(index, surface_form)] = graph
        disambiguation_graph = nx.compose(disambiguation_graph, entity_candidates.get((index, surface_form)))

    return disambiguation_graph, entity_candidates


def filter_candidates(index_surface_form, entity_candidates, w2v, n=8):
    """
    Filter candidates by taking only entities contained in the embeddings model w2v and reducing the number
    of candidates to the n most relevant (highest priors probability) candidates.

    :param index_surface_form: a tuple (index, surface_form)
    :param entity_candidates: the entity_candidates dict
    :param w2v: the embeddings model
    :param n: the number of candidates to return
    :return: a list of weighted entity candidates
    """
    return [(entity, {'weight': weight['weight']})
            for entity, weight in
            sorted(entity_candidates.get(index_surface_form).nodes(data=True),
                   key=lambda item: item[1]['weight'], reverse=True)
            if entity in w2v][:n]


def add_edges(entity_candidates, disambiguation_graph, w2v):
    """
    Add weighted edges among the nodes in the disambiguation graph. Edges' weights are the cosine similarity between
    the nodes which it is connected. Only positive weights are considered.

    :param entity_candidates: the entity_candidates dict
    :param disambiguation_graph: the disambiguation graph
    :param w2v: the embeddings model
    :return: the updated disambiguation graph
    """
    for (index, surface_form) in entity_candidates:
        for v1 in entity_candidates.get((index, surface_form)):
            for v2 in (set(disambiguation_graph.nodes()) - set(entity_candidates.get((index, surface_form)))):
                if w2v.similarity(v1, v2) > 0:
                    disambiguation_graph.add_weighted_edges_from([(v1, v2, w2v.similarity(v1, v2))])
    return disambiguation_graph


def normalize_priors(entity_candidates):
    """
    Normalize priors probability of the graph for each surface_form.

    :param entity_candidates: the entity_candidates dict
    :return: a dict {node: weight} where the sum of the nodes' weights belonging to the same surface form is 1.
    """
    denominator = {}
    for (index, surface_form) in entity_candidates:
        denominator[(index, surface_form)] = 0
        for (entity, prop) in entity_candidates.get((index, surface_form)).nodes(data=True):
            denominator[(index, surface_form)] += prop['weight']

    return {entity: prop['weight'] / denominator[(index, surface_form)]
            for (index, surface_form) in entity_candidates
            for (entity, prop) in entity_candidates.get((index, surface_form)).nodes(data=True)
            if denominator[(index, surface_form)] != 0}


def thin_out(disambiguation_graph):
    """
    Thin out 25% of those edges which weights are the lowest

    :param disambiguation_graph: the disambiguation graph
    :return: a list of the lowest weights edges to remove
    """
    n = int(0.75 * len(disambiguation_graph.edges.data("weight")))
    return sorted(disambiguation_graph.edges.data("weight"), key=lambda item: item[2])[:n]


def best_annotation(entity_candidates, page_rank):
    """
    Take the best candidate for each surface form provided by the page_rank execution

    :param entity_candidates: the entity_candidates dict
    :param page_rank: the scored provided after the page rank execution
    :return: a list of annotations
    """
    annotations = {}
    for (index, surface_form) in entity_candidates:
        best = -1
        annotations[(index, surface_form)] = "not annotated"
        for entity in entity_candidates.get((index, surface_form)):
            if page_rank[entity] > best:
                best = page_rank[entity]
                annotations[(index, surface_form)] = entity

    return annotations.values()


def annotate(table, save_to):
    """
    Save the table in save_to

    :param table: the annotated table
    :param save_to: the path where save the table
    :return:
    """
    with open(save_to, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ', quoting=csv.QUOTE_ALL)
        for tab, col, row, annotation in table.itertuples(index=False):
            writer.writerow([tab, col, row, annotation])
        file.close()
