import json

from annotators import CEAAnnotator
from data_model.generator import FactBaseConfig, EmbeddingOnGraphConfig
from data_model.lookup import ESLookupFuzzyConfig, ESLookupTrigramConfig, ESLookupExactConfig
from datasets import DatasetEnum
from experiments.evaluation import CEAEvaluator
from generators import HybridGenerator
from generators.baselines import FactBase, EmbeddingOnGraph
from generators.ours import FactBaseSTA2V, FactBaseSTR2V, EmbeddingOnGraphST
from lookup.services import ESLookupFuzzy, ESLookupTrigram, ESLookupExact

res = {}
ELASTIC_INDEX_HOST = 'titan'
DATASETS = [DatasetEnum.T2D, DatasetEnum.ST19_Round4, DatasetEnum.TT]

# EmbeddingOnGraph
es_trigram = ESLookupTrigram(ESLookupTrigramConfig(ELASTIC_INDEX_HOST, 'dbpedia', 100))
es_exact = ESLookupExact(ESLookupExactConfig(ELASTIC_INDEX_HOST, 'dbpedia', 100))
eog_cfg = EmbeddingOnGraphConfig(max_subseq_len=0, max_candidates=8, thin_out_frac=0.25)

for eog in [EmbeddingOnGraph, EmbeddingOnGraphST]:
    result = CEAEvaluator(CEAAnnotator(eog(es_exact, es_trigram, config=eog_cfg), max_workers=4)).score(DATASETS)
    for k, v in result.items():
        if k not in res:
            res[k] = {}
        res[k].update(v)

# FactBase
es_fuzzy = ESLookupFuzzy(ESLookupFuzzyConfig(ELASTIC_INDEX_HOST, 'dbpedia', 100))
fb_cfg = FactBaseConfig(max_subseq_len=0, max_workers=2)
for fb in [FactBase, FactBaseSTR2V, FactBaseSTA2V]:
    result = CEAEvaluator(CEAAnnotator(fb(es_fuzzy, config=fb_cfg), max_workers=4)).score(DATASETS)
    for k, v in result.items():
        if k not in res:
            res[k] = {}
        res[k].update(v)

# Hybrid
fb = FactBase(es_fuzzy, config=fb_cfg)
fbr2v = FactBaseSTR2V(es_fuzzy, config=fb_cfg)
fba2v = FactBaseSTA2V(es_fuzzy, config=fb_cfg)
eog = EmbeddingOnGraph(es_exact, es_trigram, config=eog_cfg)
eogst = EmbeddingOnGraphST(es_exact, es_trigram, config=eog_cfg)

hybrid_models = [HybridGenerator(fb, eog),
                 HybridGenerator(eog, fb),
                 HybridGenerator(fbr2v, eogst),
                 HybridGenerator(fba2v, eogst),
                 HybridGenerator(eogst, fbr2v),
                 HybridGenerator(eogst, fba2v)]

for hybrid in hybrid_models:
    result = CEAEvaluator(CEAAnnotator(hybrid, max_workers=8)).score(DATASETS)
    for k, v in result.items():
        if k not in res:
            res[k] = {}
        res[k].update(v)

with open(f"eswc_experiments.json", 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
