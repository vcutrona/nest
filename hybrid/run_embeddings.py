import time
import pandas as pd
import networkx as nx
from gensim.models import KeyedVectors
from ngram import NGram

from factbase_lookup import sortTable, getLabelColumn

from embeddings import sfi_to_ste, save, load, generate_candidates, create_subset, \
    disambiguate_entities, add_edges, normalize_priors, thin_out, best_annotation, annotate

base_dir = '/datahdd/gpuleri/'
datasets = '/datahdd/gpuleri/T2D/targets/'
target = datasets + 'T2D_embedding.csv'  # CEA_T2D_Targets.csv

start = time.time()
table = pd.read_csv(sortTable(target, 'table'))
label_column = getLabelColumn(table)
annotated_table = table.copy()
annotations = []
#surface_to_entities = sfi_to_ste(base_dir + 'surfaceFormIndex.csv')  # convert
#save(surface_to_entities, base_dir + 'surface_to_entities.pickle')

surface_to_entities = load(base_dir + 'surface_to_entities.pickle')
print("LOAD", time.time() - start)
ng = NGram([surface_form for surface_form in surface_to_entities.keys()])
print("LOAD NGRAM", time.time() - start)
for labels in label_column:
    begin = time.time()
    entity_candidates = generate_candidates(labels, surface_to_entities, ng)
    print("CANDIDATES", len(entity_candidates))

    subset = create_subset(entity_candidates,
                           base_dir + 'dbpedia-embeddings-skipgram-300-filtered.txt',
                           base_dir + 'dbpedia-embeddings-skipgram-300-subset.txt')
    w2v = KeyedVectors.load_word2vec_format(subset, binary=False)
    disambiguation_graph, entity_candidates = disambiguate_entities(entity_candidates, w2v)
    disambiguation_graph = add_edges(entity_candidates, disambiguation_graph, w2v)
    personalization = normalize_priors(entity_candidates)
    disambiguation_graph.remove_edges_from(thin_out(disambiguation_graph))

    #
    # NO ETP NEEDED SINCE NX.PAGERANK AUTOMATICALLY NORMALIZE USING THE SAME PAPER'S FORMULA
    #
    page_rank = nx.pagerank(disambiguation_graph, tol=1e-04, max_iter=50, alpha=0.9, personalization=personalization)
    annotations += list(best_annotation(entity_candidates, page_rank))
    done = time.time()
    print("ANNOTATE", len(annotations), done - begin)
    print("TOTAL TIME", time.time() - start)


annotated_table["entity"] = annotations
annotate(annotated_table, base_dir + 'annotations.csv')
