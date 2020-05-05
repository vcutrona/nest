from experiments.evaluate import SimpleEvaluator
from generators.baselines import ESLookup, WikipediaSearch, DBLookup
# from generators.ours import FastElmo
from gt import GTEnum

evaluators = [
    SimpleEvaluator(ESLookup(), threads=4, chunk_size=1000),
    SimpleEvaluator(WikipediaSearch(), threads=4),
    SimpleEvaluator(DBLookup(), threads=4)
]
# context_generators = [FastElmo]


for evaluator in evaluators:
    print(evaluator.score_all(exclude=[GTEnum.TEST]))
    # print(evaluator.score(GTEnum.TEST))

# print(WikipediaSearch().search("Secrets of the Heart âu0080u0093 Gilles Ortion, Alfonso Pino and Bela MarÃ­a da Costa El Tiempo de la Felicidad âu0080u0093 Daniel Goldstein, Ricardo Steinberg and Eduardo FernÃ¡ndez MartÃ­n (hache) âu0080u0093 Daniel Goldstein, Ricardo Steinberg, Carlos Garrido and Ãu0081ngel Gallardo"))
