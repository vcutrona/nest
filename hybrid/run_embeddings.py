import pandas as pd
import pickle
import time
from ngram import NGram
from tqdm.contrib.concurrent import process_map

from utils.functions import simplify_string, tokenize

from embeddings import sfi_to_ste, get_labels, generate_candidates, \
    disambiguate_entities, add_edges, normalize_priors, thin_out, page_rank, best_annotation, annotate
from utils.embeddings import WORD2Vec


def embeddings_efthy(table):
    start = time.time()

    labels = get_labels(datasets_dir + 'tables/', table)
    entity_candidates = generate_candidates(labels, surface_to_entities, ngram)
    disambiguation_graph, entity_candidates = disambiguate_entities(entity_candidates, w2v)
    disambiguation_graph = add_edges(entity_candidates, disambiguation_graph)
    priors = normalize_priors(entity_candidates)
    disambiguation_graph.remove_edges_from(thin_out(disambiguation_graph))
    pr = page_rank(disambiguation_graph, priors)

    print(table.shape[0], time.time() - start)
    return list(best_annotation(entity_candidates, pr))


base_dir = '/datahdd/gpuleri/'
datasets_dir = '/datahdd/gpuleri/T2D/'
target = datasets_dir + 'targets/CEA_T2D_Targets.csv'

target_table = pd.read_csv(target)
annotated_table = target_table.copy()
tables = [table for tab_id, table in target_table.groupby('tab_id')]
w2v = WORD2Vec()

# surface_to_entities = sfi_to_ste(base_dir + 'surfaceFormIndexComplete.csv',
#                                  base_dir + 'surfaceFormIndex.csv', w2v)  # convert
# with open(base_dir + 'surface_to_entities.pickle', 'wb') as file:  # save
#     pickle.dump(surface_to_entities, file, protocol=pickle.HIGHEST_PROTOCOL)
#     file.close()
with open(base_dir + 'surface_to_entities.pickle', 'rb') as file:  # load
    surface_to_entities = pickle.load(file)
    file.close()
ngram = NGram([surface_form for surface_form in surface_to_entities.keys()])
# with open(base_dir + 'ngram.pickle', 'wb') as file:  # save
#     pickle.dump(ngram, file, protocol=pickle.HIGHEST_PROTOCOL)
#     file.close()
# with open(base_dir + 'ngram.pickle', 'rb') as file:  # load
#     ngram = pickle.load(file)
#     file.close()
annotations = process_map(embeddings_efthy,
                          tables,
                          max_workers=10)
annotations = [element for sublist in annotations for element in sublist]
annotated_table["entity"] = annotations
annotate(annotated_table, base_dir + 'T2D_Targets_sub/T2D_Embeddings_ngram_3S_bis_new.csv')
