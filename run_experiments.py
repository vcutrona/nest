import json
from datetime import datetime

from annotators import CEAAnnotator
from data_model.lookup import ESLookupConfig
from datasets import DatasetEnum
from experiments.evaluation import CEAEvaluator
from generators.baselines import LookupGenerator, FactBase
from generators.ours import FastBert
from lookup.services import WikipediaSearch, ESLookup, DBLookup

generators = {
    LookupGenerator: [
        {
            'lookup': (ESLookup, {'config': ESLookupConfig('titan', 'dbpedia')}),
            'args': {}
        },
        {
            'lookup': (DBLookup, {}),
            'args': {}
        },
        {
            'lookup': (WikipediaSearch, {}),
            'args': {}
        },
    ],
    FactBase: [
        {
            'lookup': (ESLookup, {'config': ESLookupConfig('titan', 'dbpedia')}),
            'args': {}
        },
    ],
    FastBert: [
        {
            'lookup': (ESLookup, {'config': ESLookupConfig('titan', 'dbpedia')}),
            'args': {}
        },
        {
            'lookup': (DBLookup, {}),
            'args': {}
        },
        {
            'lookup': (WikipediaSearch, {}),
            'args': {}
        }
    ]
}

res = []

for generator, configs in generators.items():
    for config in configs:
        lookup_ = config['lookup'][0](**config['lookup'][1])
        generator_ = generator(lookup_, **config['args'])
        annotator_ = CEAAnnotator(generator_)
        evaluator = CEAEvaluator(annotator_)
        # res.append(evaluator.score(DatasetEnum.T2D))
        res.append(evaluator.score_all())

with open(f"results_{datetime.now().strftime('%d/%m/%Y_%H:%M:%S')}", 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
