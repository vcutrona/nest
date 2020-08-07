from typing import NamedTuple, List
from dataclasses import dataclass
from typing import Optional


class SearchKey(NamedTuple):
    label: str
    context: str

    def to_str(self):
        return "%s %s" % (self.label, self.context)


class LookupResult(NamedTuple):
    label: str
    candidates: List[str] = []


@dataclass
class LookupServiceConfig:
    @property
    def enable_cache(self):
        return self.enable_cache

    def cache_dir(self):
        raise NotImplementedError


@dataclass
class ESLookupConfig(LookupServiceConfig):
    host: str
    index: str
    size: int = 10
    fuzziness: Optional[int] = 2
    prefix_length: Optional[int] = 0
    max_expansions: Optional[int] = 100
    enable_cache: bool = True

    def cache_dir(self):
        return "index:%s|size:%d|fuzziness:%s|prefix_length:%s|max_expansions:%s" % \
               (self.index, self.size, str(self.fuzziness), str(self.prefix_length), str(self.max_expansions))


@dataclass
class WikipediaSearchConfig:
    url: str
    limit: int = 10
    profile: str = 'engine_autoselect'
    enable_cache: bool = True

    def cache_dir(self):
        return "limit:%d|profile:%s" % (self.limit, self.profile)


@dataclass
class DBLookupConfig:
    url: str
    max_hits: int = 5
    enable_cache: bool = True

    def cache_dir(self):
        return "max_hits:%d" % self.max_hits
