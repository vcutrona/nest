from experiments.evaluate import SimpleEvaluator  # ContextEvaluator
from generators.baselines import ESLookup, WikipediaSearch, DBLookup
# from generators.ours import FastElmo
from gt import GTEnum

evaluators = [
    SimpleEvaluator(WikipediaSearch()),
    SimpleEvaluator(ESLookup(threads=6)),
    SimpleEvaluator(DBLookup()),
    # ContextEvaluator(FastElmo())
]

for evaluator in evaluators:
    print(evaluator.score_all())
    # print(evaluator.score(GTEnum.get_test_gt(size=100, from_gt=GTEnum.CEA_ROUND1, random=False)))

