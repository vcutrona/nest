import json
import os

from datetime import datetime

from annotators import CEAAnnotator
from data_model.lookup import ESLookupFuzzyConfig, ESLookupConfig, ESLookupTrigramConfig, ESLookupExactConfig
from datasets import DatasetEnum
from experiments.evaluation import CEAEvaluator
from generators import HybridGenerator
from generators.baselines import LookupGenerator, FactBase, EmbeddingOnGraph
from generators.ours import FastBert
from lookup.services import WikipediaSearch, ESLookupFuzzy, DBLookup, ESLookupTrigram, ESLookupExact

res = []
dblookup = DBLookup()
wikisearch = WikipediaSearch()
es_fuzzy = ESLookupFuzzy(ESLookupFuzzyConfig('titan', 'dbpedia'))
es_trigram = ESLookupTrigram(ESLookupTrigramConfig('titan', 'dbpedia'))
es_exact = ESLookupExact(ESLookupExactConfig('titan', 'dbpedia', 25))

dataset = DatasetEnum.T2D

# for generator in [LookupGenerator, FactBase, FastBert]:
#     for lookup in [es_fuzzy, es_trigram, dblookup, wikisearch]:
#         res.append(CEAEvaluator(CEAAnnotator(generator(lookup))).score(dataset))

res.append(CEAEvaluator(CEAAnnotator(FactBase(dblookup))).score(dataset))

# EmbeddingOnGraph
res.append(CEAEvaluator(CEAAnnotator(EmbeddingOnGraph(es_exact, es_trigram))).score(dataset))
#
# # HybridI
# res.append(CEAEvaluator(CEAAnnotator(HybridGenerator(FactBase(dblookup), EmbeddingOnGraph(es_trigram)))).score(dataset))
# # HybridII
# res.append(CEAEvaluator(CEAAnnotator(HybridGenerator(EmbeddingOnGraph(es_trigram), FactBase(dblookup)))).score(dataset))


with open(f"results_{datetime.now().strftime('%d%m%Y_%H%M%S')}.json", 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
