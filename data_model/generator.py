from dataclasses import dataclass
from typing import NamedTuple, List, Optional, Union

import numpy as np

from data_model.lookup import SearchKey


@dataclass
class CandidateGeneratorConfig:
    max_subseq_len: int

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
class FactBaseConfig(CandidateGeneratorConfig):
    pass  # TODO put parameters


@dataclass
class EmbeddingOnGraphConfig(CandidateGeneratorConfig):
    max_candidates: int
    thin_out_frac: float

    def config_str(self) -> str:
        return "__".join([super().config_str(),
                          "max_candidates=%d" % self.max_candidates,
                          "thin_out_frac=%.2f" % self.thin_out_frac])


@dataclass
class HybridConfig(FactBaseConfig, EmbeddingOnGraphConfig):
    pass


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


@dataclass
class ScoredCandidate:
    candidate: str
    score: float

    def __lt__(self, other):
        return self.score < other.score

    def __le__(self, other):
        return self.score <= other.score

    def __gt__(self, other):
        return self.score > other.score

    def __ge__(self, other):
        return self.score >= other.score


@dataclass
class ScoredCandidateEmbeddings(ScoredCandidate):
    original_rank: Optional[int]
    distance: Optional[float]
