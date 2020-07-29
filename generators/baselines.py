import multiprocessing as mp
from typing import List

from data_model.lookup import SearchKey
from data_model.generator import GeneratorResult
from generators import CandidateGenerator
from lookup import LookupService


class LookupGenerator(CandidateGenerator):
    """
    A generator that just forwards lookup results.
    """
    def __init__(self, lookup_service: LookupService, config='Base', threads=mp.cpu_count(), chunk_size=5000):
        super().__init__(lookup_service, config, threads, chunk_size)

    def _select_candidates(self, search_keys: List[SearchKey]) -> List[GeneratorResult]:
        """
        Candidate selection method. This implementation just forwards the LookupService results.
        :param search_keys: a list of SearchKeys to use for the candidate retrieval
        :return: a list of GeneratorResult
        """
        return self._lookup_candidates(search_keys)
