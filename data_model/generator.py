from typing import NamedTuple, List, Optional

import numpy as np

from data_model.lookup import SearchKey


# GeneratorResult = namedtuple('GeneratorResult', ['search_key', 'candidates'])
class GeneratorResult(NamedTuple):
    search_key: SearchKey
    candidates: List[str] = []


# CandidateEmbeddings = namedtuple('CandidateEmbeddings', ['candidate', 'context_emb', 'abstract_emb'])
class CandidateEmbeddings(NamedTuple):
    candidate: str
    context_emb: np.ndarray = None
    abstract_emb: np.ndarray = None


# ScoredCandidate = namedtuple('ScoredCandidate', ['candidate', 'original_rank', 'distance', 'score'])
class ScoredCandidate(NamedTuple):
    candidate: str
    original_rank: int
    distance: Optional[float]
    score: Optional[float]
