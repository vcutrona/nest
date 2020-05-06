from experiments.evaluate import SimpleEvaluator  # ContextEvaluator
from generators.baselines import ESLookup, WikipediaSearch, DBLookup
# from generators.ours import FastElmo
from gt import GTEnum

evaluators = [
    SimpleEvaluator(ESLookup(threads=4)),
    SimpleEvaluator(WikipediaSearch()),
    SimpleEvaluator(DBLookup()),
    # ContextEvaluator(FastElmo())
]

for evaluator in evaluators:
    print(evaluator.score_all(exclude=[GTEnum.TEST]))
    # print(evaluator.score(GTEnum.TEST))
