from experiments.evaluate import SimpleEvaluator, ContextEvaluator
from generators.baselines import ESLookup, WikipediaSearch, DBLookup
from generators.ours import FastElmo, FastTransformers
# from gt import GTEnum

evaluators = [
    SimpleEvaluator(WikipediaSearch()),
    SimpleEvaluator(ESLookup(threads=6)),
    SimpleEvaluator(DBLookup()),
    ContextEvaluator(FastElmo()),
    ContextEvaluator(FastTransformers())
]

res = []
for evaluator in evaluators:
    # print(evaluator.score(GTEnum.get_test_gt(size=100, from_gt=GTEnum.CEA_Round1, random=False)))
    res.append(evaluator.score_all())

with open('experiments_results.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)

