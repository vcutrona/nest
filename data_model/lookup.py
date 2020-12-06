from dataclasses import dataclass
from typing import NamedTuple, List, Tuple
from typing import Optional


class SearchKey(NamedTuple):
    label: str
    context: Tuple[Tuple[int, str]]  # Dict and List are not hashable...


class LookupResult(NamedTuple):
    label: str
    candidates: List[str] = []


@dataclass
class LookupServiceConfig:
    @property
    def enable_cache(self):
        return self.enable_cache

    @property
    def max_hits(self):
        raise NotImplementedError

    def cache_dir(self):
        raise NotImplementedError


@dataclass
class ESLookupConfig(LookupServiceConfig):
    host: str
    index: str
    size: int = 25
    enable_cache: bool = True

    @property
    def max_hits(self):
        return self.size

    def cache_dir(self):
        return "index=%s__size=%d" % (self.index, self.size)


@dataclass
class ESLookupExactConfig(ESLookupConfig):
    size: int = 10000


@dataclass
class ESLookupFuzzyConfig(ESLookupConfig):
    fuzziness: Optional[int] = 2
    prefix_length: Optional[int] = 0
    max_expansions: Optional[int] = 100

    def cache_dir(self):
        return "index=%s__size=%d__fuzziness=%s__prefix_length=%s__max_expansions=%s" % \
               (self.index, self.size, str(self.fuzziness), str(self.prefix_length), str(self.max_expansions))


@dataclass
class ESLookupTrigramConfig(ESLookupConfig):
    min_match: str = '82%'

    def cache_dir(self):
        return "index=%s__size=%d__min_match=%s" % (self.index, self.size, self.min_match)


@dataclass
class WikipediaSearchConfig(LookupServiceConfig):
    url: str
    limit: int = 10
    profile: str = 'engine_autoselect'
    enable_cache: bool = True

    @property
    def max_hits(self):
        return self.limit

    def cache_dir(self):
        return "limit=%d__profile=%s" % (self.limit, self.profile)


@dataclass
class DBLookupConfig(LookupServiceConfig):
    url: str
    max_hits: int = 10
    enable_cache: bool = True

    def cache_dir(self):
        return "max_hits=%d" % self.max_hits
