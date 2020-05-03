from experiments.evaluate import SimpleCachedEvaluator
from generators.baselines import ESLookup, WikipediaSearch, DBLookup
# from generators.ours import FastElmo
from gt import GTEnum

simple_generators = [ESLookup, WikipediaSearch, DBLookup]
# context_generators = [FastElmo]


for generator in simple_generators:
    print(SimpleCachedEvaluator(generator()).score_all(exclude=[GTEnum.TEST]))

# print(WikipediaSearch().search("Secrets of the Heart âu0080u0093 Gilles Ortion, Alfonso Pino and Bela MarÃ­a da Costa El Tiempo de la Felicidad âu0080u0093 Daniel Goldstein, Ricardo Steinberg and Eduardo FernÃ¡ndez MartÃ­n (hache) âu0080u0093 Daniel Goldstein, Ricardo Steinberg, Carlos Garrido and Ãu0081ngel Gallardo"))

