from dataclasses import dataclass
from typing import NamedTuple, List, Optional, Union

import numpy as np

from data_model.lookup import SearchKey


@dataclass
class CandidateGeneratorConfig:
    max_subseq_len: Optional[int]

    def config_str(self) -> str:
        return "max_subseq_len=%s" % str(self.max_subseq_len)


@dataclass
class EmbeddingCandidateGeneratorConfig(CandidateGeneratorConfig):
    abstract: str
    abstract_max_tokens: int
    default_score: float = np.nan
    alpha: float = 0.5

    def config_str(self) -> str:
        return "__".join([super().config_str(),
                          "abstract=%s__abstract_max_tokens=%d__default_score=%s__alpha=%.2f"
                          % (self.abstract, self.abstract_max_tokens, str(self.default_score), self.alpha)])

    def cache_dir(self):
        raise NotImplementedError


@dataclass
class FastBertConfig(EmbeddingCandidateGeneratorConfig):
    strategy: str = 'context'

    def config_str(self) -> str:
        return "__".join([super().config_str(), "strategy=%s" % self.strategy])

    def cache_dir(self):
        return "abstract=%s__abstract_max_tokens=%d__strategy=%s" % (self.abstract,
                                                                     self.abstract_max_tokens,
                                                                     self.strategy)


@dataclass
class AbstractCollectorConfig:
    es_host: str = 'localhost'
    index: str = 'dbpedia'
    sparql_endpoint: str = 'http://dbpedia.org/sparql'


class GeneratorResult(NamedTuple):
    search_key: SearchKey
    candidates: List[str] = []


class Embedding(NamedTuple):
    key: Union[str, SearchKey]
    embedding: np.ndarray


class CandidateEmbeddings(NamedTuple):
    candidate: str
    context_emb: np.ndarray = np.nan
    abstract_emb: np.ndarray = np.nan


class ScoredCandidate(NamedTuple):
    candidate: str
    original_rank: int
    distance: Optional[float]
    score: Optional[float]
