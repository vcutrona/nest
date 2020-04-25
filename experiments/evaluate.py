from tqdm import tqdm

from generators import SimpleGenerator, ContextGenerator
from gs import GSEnum


class Evaluator:
    def __init__(self, gs: GSEnum):
        self._gs = gs.get_df()

    @staticmethod
    def _get_candidates(row, generator):
        raise NotImplementedError

    def score(self, generator):
        total = self._gs.shape[0]

        tqdm.pandas()
        self._gs['candidates'] = self._gs.progress_apply(self._get_candidates, args=(generator,), axis=1)

        correct = 0
        missing = 0

        for row in self._gs.itertuples():
            if not row.candidates:
                missing = missing + 1
            elif row.candidates[0] in row.entities:
                correct = correct + 1

        return {'correct': correct,
                'missing': missing,
                'wrong': total - correct - missing,
                'P': correct / total}


class SimpleEvaluator(Evaluator):
    @staticmethod
    def _get_candidates(row, generator: SimpleGenerator):
        return generator.search(label=row['label'])


class ContextEvaluator(Evaluator):
    @staticmethod
    def _get_candidates(row, generator: ContextGenerator):
        return generator.search(label=row['label'], context=row['context'])
