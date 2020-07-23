import json

from experiments.evaluate import SimpleEvaluator, ContextEvaluator
from generators.baselines import ESLookup, WikipediaSearch, DBLookup
from generators.ours import FastElmo, FastTransformers

simple_models = {
    WikipediaSearch: {},
    ESLookup: {'threads': 6},
    DBLookup: {}
}

ctx_models = {
    FastTransformers: {},
    FastElmo: {}
}

res = []
for model, args in simple_models.items():
    evaluator = SimpleEvaluator(model(**args))
    res.append(evaluator.score_all())

for modelCls, args in ctx_models.items():
    evaluator = ContextEvaluator(modelCls(**args))
    res.append(evaluator.score_all())

with open('experiments_results3.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)
