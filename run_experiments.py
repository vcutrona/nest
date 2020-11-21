import json
from datetime import datetime

from annotators import CEAAnnotator
from data_model.lookup import ESLookupFuzzyConfig, ESLookupTrigramConfig, ESLookupConfig, ESLookupExactConfig
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
es_exact = ESLookupExact(ESLookupExactConfig('titan', 'dbpedia'))

dataset = DatasetEnum.T2D

# Lookup, FactBase, FastBert
for generator in [LookupGenerator, FactBase, FastBert]:
    for lookup in [es_fuzzy, es_trigram]:
        res.append(CEAEvaluator(CEAAnnotator(generator(es_exact, lookup))).score(dataset))
        print(res[-1])
    for lookup in [dblookup, wikisearch]:
        print(res[-1])

# EmbeddingOnGraph
res.append(CEAEvaluator(CEAAnnotator(EmbeddingOnGraph(es_exact, es_trigram))).score(dataset))
print(res[-1])

# HybridI
res.append(CEAEvaluator(CEAAnnotator(HybridGenerator(FactBase(dblookup),
                                                     EmbeddingOnGraph(es_exact, es_trigram)))).score(dataset))
# HybridII
res.append(CEAEvaluator(CEAAnnotator(HybridGenerator(EmbeddingOnGraph(es_exact, es_trigram),
                                                     FactBase(dblookup)))).score(dataset))
print(res[-1])

with open(f"results_{datetime.now().strftime('%d%m%Y_%H%M%S')}.json", 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
