from dataclasses import dataclass


@dataclass
class DBpediaWrapperConfig:
    es_host: str = 'localhost'
    index: str = 'dbpedia'
    sparql_endpoint: str = 'http://dbpedia.org/sparql'
