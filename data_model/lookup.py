from typing import NamedTuple, List


# SearchKey = namedtuple('SearchKey', ['label', 'context'])
class SearchKey(NamedTuple):
    label: str
    context: str


# LookupResult = namedtuple('LookupResult', ['label', 'candidates'])
class LookupResult(NamedTuple):
    label: str
    candidates: List[str] = []
