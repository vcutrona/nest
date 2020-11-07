import json

from annotators import CEAAnnotator
from data_model.generator import CandidateGeneratorConfig, EmbeddingCandidateGeneratorConfig, FastBertConfig, \
    FactBaseConfig
from data_model.lookup import ESLookupConfig
from experiments.evaluation import CEAEvaluator
from generators.baselines import LookupGenerator, FactBase
from generators.ours import FastBert
from datasets import DatasetEnum
from lookup.services import WikipediaSearch, ESLookup, DBLookup

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
                'config': CandidateGeneratorConfig(max_subseq_len=None)
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
                'config': CandidateGeneratorConfig(max_subseq_len=None)
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
                'config': CandidateGeneratorConfig(max_subseq_len=None)
            }
        }
    ],
    FastBert: [
        {
            'lookup': (ESLookup, {}),
            'args': {}
        },
        {
            'lookup': (ESLookup, {}),
            'args': {'config': FastBertConfig(max_subseq_len=None,
                                              abstract='short',
                                              abstract_max_tokens=512)}
        },
        {
            'lookup': (DBLookup, {}),
            'args': {}
        },
        {
            'lookup': (DBLookup, {}),
            'args': {'config': FastBertConfig(max_subseq_len=None,
                                              abstract='short',
                                              abstract_max_tokens=512)}
        },
        {
            'lookup': (WikipediaSearch, {}),
            'args': {}
        },
        {
            'lookup': (WikipediaSearch, {}),
            'args': {'config': FastBertConfig(max_subseq_len=None,
                                              abstract='short',
                                              abstract_max_tokens=512)}
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
        res.append(evaluator.score_all())

with open('experiments_results.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)
