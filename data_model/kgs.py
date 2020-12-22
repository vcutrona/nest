from urllib.parse import unquote
from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class DBpediaWrapperConfig:
    es_host: str = 'titan'
    index: str = 'dbpedia'
    sparql_endpoint: str = 'http://dbpedia.org/sparql'
    default_graph: str = 'http://dbpedia.org'


class Entity(NamedTuple):
    uri: str

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Entity):
            return unquote(self.uri).lower() == unquote(o.uri).lower()
        return False

    def __hash__(self) -> int:
        return hash(unquote(self.uri).lower())


class Class(NamedTuple):
    uri: str


class Property(NamedTuple):
    uri: str
