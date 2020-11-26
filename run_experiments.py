import numpy as np

from annotators import CEAAnnotator
from data_model.generator import LookupGeneratorConfig
from data_model.lookup import ESLookupFuzzyConfig, ESLookupTrigramConfig, ESLookupExactConfig
from datasets import DatasetEnum
from experiments.evaluation import CEAEvaluator
from generators.baselines import LookupGenerator
from lookup.services import WikipediaSearch, ESLookupFuzzy, DBLookup, ESLookupTrigram, ESLookupExact

res = []
dblookup = DBLookup()
wikisearch = WikipediaSearch()
es_fuzzy = ESLookupFuzzy(ESLookupFuzzyConfig('titan', 'dbpedia'))
es_trigram = ESLookupTrigram(ESLookupTrigramConfig('titan', 'dbpedia'))
es_exact = ESLookupExact(ESLookupExactConfig('titan', 'dbpedia', 25))

dataset = [DatasetEnum.TT, DatasetEnum.T2D]

# Lookup (ES)
cfg = LookupGeneratorConfig(max_subseq_len=0, max_workers=4, chunk_size=250)
for lookup in [es_fuzzy, es_trigram]:
    res.append(CEAEvaluator(CEAAnnotator(LookupGenerator(es_exact, lookup, config=cfg), max_workers=2)).score_all())
    print(res[-1])


# Lookup (Online DBpedia and WikipediaSearch -> reduce the amount of parallel requests)
for lookup in [dblookup, wikisearch]:
    print(lookup)
    for dataset in DatasetEnum:
        print(dataset)
        len_median = np.median([len(table.get_gt_cells()) for table in dataset.get_tables()])
        if len_median > 25:
            dataset_workers = 2
            table_workers = 2
            chunk_size = int(len_median / table_workers)
        else:
            dataset_workers = 4
            table_workers = 1
            chunk_size = None
        cfg = LookupGeneratorConfig(max_subseq_len=0, max_workers=table_workers, chunk_size=chunk_size)
        print(dataset_workers, cfg)

        result = CEAEvaluator(CEAAnnotator(LookupGenerator(lookup, config=cfg),
                                           max_workers=dataset_workers)).score_one(dataset)
        print(result)
        for k, v in result.items():
            if k not in res:
                res[k] = {}
            res[k].update(v)

print(res)

# with open(f"results_{datetime.now().strftime('%d%m%Y_%H%M%S')}.json", 'w', encoding='utf-8') as f:
#     json.dump(res, f, ensure_ascii=False, indent=2)
