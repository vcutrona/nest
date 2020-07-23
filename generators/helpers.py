import os
from configparser import ConfigParser

from SPARQLWrapper import SPARQLWrapper, JSON
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q


class AbstractHelper:
    def __init__(self, config='AbstractHelper'):
        cfg = ConfigParser()
        cfg.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        self._config = cfg[config]
        assert Elasticsearch(self._config['es_host']).ping()

        self._sparql = SPARQLWrapper(self._config['sparql_endpoint'])
        self._sparql.setReturnFormat(JSON)

    def _get_es_docs_by_ids(self, docs_ids):
        elastic = Elasticsearch(self._config['es_host'])
        hits = []
        for i in range(0, len(docs_ids), 10000):  # max result window size
            s = Search(using=elastic, index=self._config['index']).query(Q('ids', values=docs_ids[i:i + 10000]))
            hits = hits + [(hit.meta.id, hit.to_dict()) for hit in s.execute()]
        return hits

    def _get_abstracts_by_ids(self, docs_ids):
        return {doc_id: doc['description'] for doc_id, doc in self._get_es_docs_by_ids(docs_ids)}

    @staticmethod
    def _cut_abstract(abstract, size) -> str:
        return " ".join(abstract.split(" ")[:size]).strip()

    def fetch_long_abstracts(self, uris, max_tokens=None):  # TODO - re-implement
        pass
        # self._sparql.setQuery("""
        #                         SELECT DISTINCT ?abstract
        #                         WHERE {
        #                             <%s> dbo:abstract ?abstract.
        #                             FILTER (LANG(?abstract) = 'en' || LANG(?abstract) = '')
        #                         }
        #                     """ % uri)
        # results = self._sparql.query().convert()
        # return [result["abstract"]["value"] for result in results["results"]["bindings"]]

    def fetch_short_abstracts(self, uris, max_tokens=None):
        return {entity_uri: self._cut_abstract(entity_abstracts[0], max_tokens) if entity_abstracts else ''
                for entity_uri, entity_abstracts in self._get_abstracts_by_ids(uris).items()}
