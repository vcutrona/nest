from tqdm import tqdm

from generators import SimpleGenerator, ContextGenerator
from generators.baselines import ESLookup
from gs import GSEnum


class Evaluator:
    def __init__(self, gs: GSEnum):
        self._gs = gs.get_df()

    def _call_search(self, generator, row):
        raise NotImplementedError

    def score(self, generator):
        correct = 0
        total = 0
        empty = 0
        for index, row in tqdm(self._gs.iterrows(), total=self._gs.shape[0]):
            result = self._call_search(generator, row)

            total = total + 1

            if not result:
                empty = empty + 1
            elif result[0] in row["entities"].split():
                correct = correct + 1

        return correct, empty, total, correct / total


class SimpleEvaluator(Evaluator):
    def _call_search(self, generator: SimpleGenerator, row):
        return generator.search(label=row['label'])


class ContextEvaluator(Evaluator):
    def _call_search(self, generator: ContextGenerator, row):
        return generator.search_context(label=row['label'], context=row['context'])

