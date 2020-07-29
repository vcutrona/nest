import json

from experiments.evaluation import Evaluator
from generators.baselines import LookupGenerator
from generators.ours import FastElmo, FastTransformer
from lookup.services import WikipediaSearch, ESLookup, DBLookup

res = []

generators = {
    LookupGenerator: [
        {
            'lookup': (ESLookup, {}),
            'args': {
                'threads': 6
            }
        },
        {
            'lookup': (ESLookup, {}),
            'args': {
                'threads': 6,
                'config': 'BaseSingle'
            }
        },
        {
            'lookup': (DBLookup, {}),
            'args': {
                'threads': 2,
                'chunk_size': 1000
            }
        },
        {
            'lookup': (DBLookup, {}),
            'args': {
                'threads': 2,
                'chunk_size': 1000,
                'config': 'BaseSingle'
            }
        },
        {
            'lookup': (WikipediaSearch, {}),
            'args': {
                'threads': 3,
                'chunk_size': 1000
            }
        },
        {
            'lookup': (WikipediaSearch, {}),
            'args': {
                'threads': 3,
                'chunk_size': 1000,
                'config': 'BaseSingle'
            }
        }
    ],
    FastElmo: [
        {
            'lookup': (ESLookup, {}),
            'args': {}
        },
        {
            'lookup': (ESLookup, {}),
            'args': {'config': 'FastElmoSingle'}
        },
        {
            'lookup': (DBLookup, {}),
            'args': {}
        },
        {
            'lookup': (DBLookup, {}),
            'args': {'config': 'FastElmoSingle'}
        },
        {
            'lookup': (WikipediaSearch, {}),
            'args': {}
        },
        {
            'lookup': (WikipediaSearch, {}),
            'args': {'config': 'FastElmoSingle'}
        },
    ],
    FastTransformer: [
        {
            'lookup': (ESLookup, {}),
            'args': {}
        },
        {
            'lookup': (ESLookup, {}),
            'args': {'config': 'FastTransformerSingle'}
        },
        {
            'lookup': (DBLookup, {}),
            'args': {}
        },
        {
            'lookup': (DBLookup, {}),
            'args': {'config': 'FastTransformerSingle'}
        },
        {
            'lookup': (WikipediaSearch, {}),
            'args': {}
        },
        {
            'lookup': (WikipediaSearch, {}),
            'args': {'config': 'FastTransformerSingle'}
        }
    ]
}

for generator, configs in generators.items():
    for config in configs:
        evaluator = Evaluator(generator(config['lookup'][0](**config['lookup'][1]), **config['args']))
        res.append(evaluator.score_all())

with open('experiments_results.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)
