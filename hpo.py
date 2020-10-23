import os
import pickle

import numpy as np
from skopt import dump, load
from skopt import forest_minimize
from skopt.space import Real, Integer, Categorical

from data_model.generator import FastBertConfig
from experiments.evaluation import CEAEvaluator
from generators.ours import FastBert
from lookup.services import WikipediaSearch

# [FastBertConfig space]
space = [Categorical([0, 3, 4, 5], name="max_subseq_len"),
         # Categorical(['short', 'long'], name="abstract"),
         Categorical([8, 16, 32, 64, 128, 256, 512], name="abstract_max_tokens"),
         Real(0.5, 2.0, name='default_score'),
         Real(0.0, 1.0, name='alpha'),
         Categorical(['context', 'sentence', 'cls'], name="strategy"),
         Integer(0, 1, name="dummy_default_score")]


def config_values_from_sugg(suggestion):
    cfg_values = [suggestion[0], 'short'] + suggestion[1:-1]
    if not suggestion[-1]:
        cfg_values[3] = None
    return cfg_values


def ml_algorithm(suggestion):
    if tuple(suggestion) in eval_dict:
        return -np.median(eval_dict[tuple(suggestion)])

    cfg = FastBertConfig(*config_values_from_sugg(suggestion))
    print(suggestion)
    print(cfg)
    evaluator = CEAEvaluator(FastBert(WikipediaSearch(), cfg))
    # score_dict = evaluator.score(GTEnum.get_test_gt(10))
    score_dict = evaluator.score_all()
    metrics = []
    for key, value in list(score_dict.values())[0].items():
        metrics.append(value['ALL']['f1'])
    eval_dict[tuple(suggestion)] = metrics
    pickle.dump(eval_dict, open('eval_dict.pkl', 'wb'))

    metric = np.median(metrics)
    print(metric, metrics)
    return -metric


# HPO TASK
eval_dict = {}

if os.path.isfile('eval_dict.pkl'):
    eval_dict = pickle.load(open('eval_dict.pkl', 'rb'))

for i in range(3):
    hpo_results = forest_minimize(ml_algorithm, space, n_calls=100, random_state=42 + i, base_estimator="RF")

    # Optimal solution
    optimal_config = hpo_results.x
    optimal_config_evaluation = hpo_results.fun
    # History solutions
    history_config = hpo_results.x_iters
    history_config_evaluation = hpo_results.func_vals

    # Save and load results of HPO experiments
    # Save results
    dump(hpo_results, f'hpo_task_data_{i}.pkl')
# Load results
# hpo_loaded_results = load('hpo_task_data.pkl')
