from experiments.evaluate import ContextEvaluator, SimpleEvaluator
from generators.baselines import ESLookup, WikipediaSearch, DBLookup
from generators.ours import FastElmo
from gs import GSEnum

print(ContextEvaluator(GSEnum.CEA_ROUND1).score(FastElmo()))
print(SimpleEvaluator(GSEnum.CEA_ROUND1).score(ESLookup()))
print(SimpleEvaluator(GSEnum.CEA_ROUND1).score(WikipediaSearch()))
print(SimpleEvaluator(GSEnum.CEA_ROUND1).score(DBLookup()))
