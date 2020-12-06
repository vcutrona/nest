import ast
import csv
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from gensim import matutils
from numpy import dot

from utils.functions import simplify_string


def sfi_to_ste(surface_form_index_complete, surface_form_index, w2v):
    """
    Convert a surface_form_index in a more readable and quickly accessible surface_to_entities.
    High speeding of candidates generation.

    e.g.:
        surface_form_index =    'http://dbpedia.org/resource/Italy' ['italy', 'italia']          554
                                'http://dbpedia.org/resource/Roma'  ['italy', 'italia', 'roma']  266
        surface_to_entities = { 'italy':    {   'http://dbpedia.org/resource/Italy':    554,
                                                'http://dbpedia.org/resource/Roma':     266}
                                'italia':   {   'http://dbpedia.org/resource/Italy':    554,
                                                'http://dbpedia.org/resource/Roma':     266}
                                'roma':     {   'http://dbpedia.org/resource/Roma':     266}    }

    :param surface_form_index_complete: the complete csv
    :param surface_form_index: the new filtered csv
    :param w2v: an embeddings wrapper
    :return: a dict surface_to_entities
    """
    dict_candidates = defaultdict(lambda: defaultdict(tuple))
    filter_sfi(surface_form_index_complete, surface_form_index, w2v)

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


def filter_sfi(surface_form_index_complete, surface_form_index, w2v):
    """

    :param surface_form_index_complete: the complete csv
    :param surface_form_index: the new filtered csv
    :param w2v: an embeddings wrapper
    :return:
    """
    with open(surface_form_index_complete, 'r', newline='', encoding='utf-8') as surface_form_index_file:
        reader = csv.reader(surface_form_index_file, delimiter=' ')
        next(reader)
        data = [[entity, sf_forms, count] for entity, sf_forms, count in reader]
        with open(surface_form_index, 'w', newline='', encoding='utf-8') as filew:
            writer = csv.writer(filew, delimiter=' ', quoting=csv.QUOTE_ALL)
            writer.writerow(['entity', 'surface_form', 'count'])

            embedding = w2v.get_vectors([entity for entity, sf_forms, count in data])
            writer.writerows([[entity, sf_forms, count]
                              for entity, sf_forms, count in data if embedding[entity]])

            filew.close()
        surface_form_index_file.close()


def get_labels(path, table):
    """
    Get labels from the table located at path

    :param path: the path where the table is located
    :param table: the table dataframe
    :return: a list of labels
    """
    labels = []
    for tab_id, col_id, row_id in table.itertuples(index=False):
        tab = pd.read_csv(path + tab_id + '.csv')
        labels.append(str(tab.iloc[row_id - 1][col_id]))

    return labels


def generate_candidates(labels, surface_to_entities, ngram):
    """
    Generate possible candidates for all the labels.
    It compares the labels to those stored in the surface_to_entities dict. All entities in the surface_to_entities
    that provide an exact matching with the label serve as entity candidate. If no candidates are retrieved it
    computes stemming and stopwords removal on the label and it compares the new label to those stored in the
    surface_to_entities dict by applying trigram similarity; all entities that overcome the threshold=0.82 serve as
    entity candidates.
    entity_candidates is a dict where keys k are tuple (index, sf_form) and values are graph which nodes are candidates.
    Maintaining k different graph is very useful to easily create a disambiguation graph in the disambiguation step.

    :param labels: a list of labels
    :param surface_to_entities: the surface_to_entities dict
    :param ngram: the ngram model
    :return: an entity_candidates dict
    """
    entity_candidates = {}

    index = 0  # index needed since label might be repeated
    for label in labels:
        graph = nx.DiGraph()
        if label.lower() in surface_to_entities:
            candidates = [(entity, {'weight': int(surface_to_entities[label.lower()].get(entity))})
                          for entity in surface_to_entities[label.lower()]]

        else:  # if no exact matching with label
            label_mod = simplify_string(label.lower(), dates=True, numbers=False, single_char=False, brackets=True)\
                .replace('&nbsp;', '')
            #label_mod = ' '.join(tokenize(label_mod, stemming=True))  # stemming and stopwords

            listSF = [word for (word, sim) in ngram.search(label_mod, 0.82)]  # trigram sim.
            candidates = [(entity, {'weight': int(surface_to_entities[sf_form].get(entity))})
                          for sf_form in listSF for entity in surface_to_entities[sf_form]]

        graph.add_nodes_from(candidates)
        entity_candidates[(index, label)] = graph  # index needed since label might be repeated
        index += 1

    return entity_candidates


def disambiguate_entities(entity_candidates, w2v):
    """
    Create a complete directed k-partite disambiguation graph where k is the number of surface forms.
    Since the number of entity candidates could be extremely high, a filtering is applied.

    :param entity_candidates: the entity_candidates dict
    :param w2v: an embeddings wrapper
    :return: the disambiguation graph, the entity_candidates updated after the filtering
    """
    disambiguation_graph = nx.DiGraph()
    embeddings = w2v.get_vectors([entity for (index, surface_form) in entity_candidates
                                  for entity in entity_candidates.get((index, surface_form))])

    for (index, surface_form) in entity_candidates:
        graph = nx.DiGraph()
        nodes = filter_candidates((index, surface_form), entity_candidates, embeddings)
        graph.add_nodes_from(nodes)
        entity_candidates[(index, surface_form)] = graph
        disambiguation_graph = nx.compose(disambiguation_graph, entity_candidates.get((index, surface_form)))

    return disambiguation_graph, entity_candidates


def filter_candidates(index_surface_form, entity_candidates, embeddings, n=3):
    """
    Filter candidates by taking only entities contained in the embeddings model w2v and reducing the number
    of candidates to the n most relevant (highest priors probability) candidates.

    :param index_surface_form: a tuple (index, surface_form)
    :param entity_candidates: the entity_candidates dict
    :param embeddings: the embeddings
    :param n: the number of candidates to return
    :return: a list of weighted entity candidates and their embeddings
    """
    return [(entity, {'weight': weight, 'embedding': np.array(embeddings[entity])})
            for entity, weight in
            sorted(entity_candidates.get(index_surface_form).nodes.data('weight'),
                   key=lambda item: item[1], reverse=True)
            ][:n]


def add_edges(entity_candidates, disambiguation_graph):
    """
    Add weighted edges among the nodes in the disambiguation graph. Weights of edges are the cosine similarity between
    the nodes which the edge is connected to. Only positive weights are considered.

    :param entity_candidates: the entity_candidates dict
    :param disambiguation_graph: the disambiguation graph
    :return: an updated disambiguation graph
    """
    for (index, surface_form) in entity_candidates:
        for v1, v1_emb in entity_candidates.get((index, surface_form)).nodes.data('embedding'):
            for v2 in (set(disambiguation_graph.nodes()) - set(entity_candidates.get((index, surface_form)))):
                v2_emb = disambiguation_graph.nodes.data('embedding')[v2]
                cos_sim = similarity(v1_emb, v2_emb)
                if cos_sim > 0:
                    disambiguation_graph.add_weighted_edges_from([(v1, v2, cos_sim)])
    return disambiguation_graph


def similarity(v1, v2):
    """
    Compute cosine-similarity between vector v1 and vector v2.

    :param v1: vector v1
    :param v2: vector v2
    :return: cosine-similarity value between v1 and v2
    """
    return dot(matutils.unitvec(v1), matutils.unitvec(v2))


def normalize_priors(entity_candidates):
    """
    Normalize priors probability of the graph for each surface_form.

    :param entity_candidates: the entity_candidates dict
    :return: a dict {node: weight} where the sum of weights of the nodes which belong to the same surface form is 1.
    """
    denominator = {}
    for (index, surface_form) in entity_candidates:
        denominator[(index, surface_form)] = 0
        for entity, weight in entity_candidates.get((index, surface_form)).nodes.data('weight'):
            denominator[(index, surface_form)] += weight

    return {entity: weight / denominator[(index, surface_form)]
            for (index, surface_form) in entity_candidates
            for (entity, weight) in entity_candidates.get((index, surface_form)).nodes.data('weight')
            if denominator[(index, surface_form)] != 0}


def thin_out(disambiguation_graph):
    """
    Thin out 25% of those edges which weights are the lowest

    :param disambiguation_graph: the disambiguation graph
    :return: a list of the lowest weights edges to remove
    """
    n = int(0.75 * len(disambiguation_graph.edges.data("weight")))
    return sorted(disambiguation_graph.edges.data("weight"), key=lambda item: item[2])[:n]


def page_rank(disambiguation_graph, priors):
    """
    Compute page rank algorithm where the epsilon used for convergence is increased by a factor 2 until convergence.

    :param disambiguation_graph: the disambiguation graph
    :param priors: the priors probabilities
    :return: a list of ranked entities
    """
    pr = None
    epsilon = 1e-6
    while pr is None:
        try:
            pr = nx.pagerank(disambiguation_graph, tol=epsilon, max_iter=50, alpha=0.9, personalization=priors)
        except Exception:
            epsilon *= 2  # lower factor can be used too since pagerank is extremely fast
    return pr


def best_annotation(entity_candidates, ranks):
    """
    Take the best candidate for each surface form provided by the page_rank execution

    :param entity_candidates: the entity_candidates dict
    :param ranks: the scored provided after the page rank execution
    :return: a list of annotations
    """
    annotations = {}
    for (index, surface_form) in entity_candidates:
        best = -1
        annotations[(index, surface_form)] = "not annotated"
        for entity in entity_candidates.get((index, surface_form)):
            if ranks[entity] > best:
                best = ranks[entity]
                annotations[(index, surface_form)] = entity

    return annotations.values()


def annotate(table, save_to):
    """
    Save the table in save_to

    :param table: the annotated table
    :param save_to: the path where save the table
    :return:
    """
    with open(save_to, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=' ', quoting=csv.QUOTE_ALL)
        for tab, col, row, annotation in table.itertuples(index=False):
            writer.writerow([tab, col, row, annotation])
        file.close()
