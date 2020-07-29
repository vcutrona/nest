from dataclasses import dataclass
from typing import NamedTuple, List, Optional

import numpy as np

from data_model.lookup import SearchKey


@dataclass
class CandidateGeneratorConfig:
    max_subseq_len: Optional[int]

    def config_str(self) -> str:
        return "max_subseq_len:%s" % str(self.max_subseq_len)


@dataclass
class EmbeddingCandidateGeneratorConfig(CandidateGeneratorConfig):
    abstract: str
    abstract_max_tokens: int
    default_score: Optional[float] = None
    alpha: float = 0.5

    def config_str(self) -> str:
        return "max_subseq_len:%s|abstract:%s|abstract_max_tokens:%d|default_score:%s|alpha:%.2f" % \
               (str(self.max_subseq_len), self.abstract, self.abstract_max_tokens, str(self.default_score), self.alpha)


@dataclass
class AbstractCollectorConfig:
    es_host: str = 'localhost'
    index: str = 'dbpedia'
    sparql_endpoint: str = 'http://dbpedia.org/sparql'


class GeneratorResult(NamedTuple):
    search_key: SearchKey
    candidates: List[str] = []


class CandidateEmbeddings(NamedTuple):
    candidate: str
    context_emb: np.ndarray = None
    abstract_emb: np.ndarray = None


class ScoredCandidate(NamedTuple):
    candidate: str
    original_rank: int
    distance: Optional[float]
    score: Optional[float]
