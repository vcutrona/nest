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

for evaluator in evaluators:
    # print(evaluator.score(GTEnum.get_test_gt(size=100, from_gt=GTEnum.CEA_Round1, random=False)))
    print(evaluator.score_all())

