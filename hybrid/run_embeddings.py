import time
import pandas as pd
import networkx as nx
from gensim.models import KeyedVectors
from ngram import NGram

from factbase_lookup import sortTable, getLabelColumn
# from hybrid.factbase_lookup import sortTable, getLabelColumn

from embeddings import etp, thinOut, surfaceFormIndex_to_surfaceToEntities, save, load, \
    generate_candidates, create_subset, disambiguate_entities, addEdges, normalizePriors, bestAnnotation, annotate
"""from hybrid.embeddings import etp, thinOut, surfaceFormIndex_to_surfaceToEntities, save, load, \
    generate_candidates, create_subset, disambiguate_entities, addEdges, normalizePriors, bestAnnotation, annotate"""

# base_dir = '../hybrid/'
base_dir = '/datahdd/gpuleri/'
# datasets = '../datasets/T2D/targets/'
datasets = '/datahdd/gpuleri/T2D/targets/'
target = datasets + 'T2D_embedding.csv'

start = time.time()
table = pd.read_csv(sortTable(target, 'table'))  # get targets
labelColumn = getLabelColumn(table)  # get labels
annotated_table = table.copy()
annotations = []
#surfaceToEntities = surfaceFormIndex_to_surfaceToEntities(base_dir + 'surfaceFormIndex.csv')  # convert

#save(surfaceToEntities, base_dir + 'surfaceToEntities.pickle')
surfaceToEntities = load(base_dir + 'surfaceToEntities.pickle')
print("LOAD")
print(time.time() - start)
#ng = NGram([surfaceForm for surfaceForm in surfaceToEntities.keys()])
for labels in labelColumn:
    # generate candidates
    entity_candidates = generate_candidates(labels, surfaceToEntities, 0)  # ng
    print("candidates")
    print(time.time() - start)

    #save(entity_candidates, base_dir + 'candidates.pickle')
    #entity_candidates = load(base_dir + 'candidates.pickle')

    # load embeddings and create disambiguation graph
    subset = create_subset(entity_candidates,
                           base_dir + 'dbpedia-embeddings-skipgram-300-filtered.txt',
                           base_dir + 'dbpedia-embeddings-skipgram-300-subset.txt')
    w2v = KeyedVectors.load_word2vec_format(subset, binary=False)
    disambiguation_graph, entity_candidates = disambiguate_entities(entity_candidates, w2v)

    # add edges on the graph
    disambiguation_graph = addEdges(entity_candidates, disambiguation_graph, w2v)

    # normalize priors
    denominator = normalizePriors(entity_candidates)

    # normalizza prob a priori
    personalization = {node: prop['weight'] / denominator[surface_form] for surface_form in entity_candidates
                       for (node, prop) in entity_candidates.get(surface_form).nodes(data=True)
                       if denominator[surface_form] != 0}

    # calcola denominatore per cos similarity
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

    # pagerank
    pageRank = nx.pagerank(disambiguation_graph, tol=1e-05, max_iter=50, alpha=0.9, personalization=personalization)

    # pick the best candidate for each surface form and save all of them
    annotations += list(bestAnnotation(entity_candidates, pageRank))
    print("ANNOTATE")
    print(time.time() - start)

annotated_table["entity"] = annotations
annotate(annotated_table, base_dir + 'annotations.csv')
