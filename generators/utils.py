import urllib

import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

from data_model.generator import AbstractCollectorConfig
from utils import functions


class AbstractCollector:
    """
    Helper class to retrieve abstracts for DBpedia entities.
    """

    def __init__(self, config: AbstractCollectorConfig = AbstractCollectorConfig()):
        self._config = config
        assert Elasticsearch(self._config.es_host).ping()

        self._sparql = SPARQLWrapper(self._config.sparql_endpoint)
        self._sparql.setReturnFormat(JSON)

    def _get_es_docs_by_ids(self, docs_ids):
        """
        Retrieve documents from an ElasticSearch index.
        :param docs_ids: ids of the documents to retrieve
        :return: a dictionary Dict(doc_id: document)
        """
        elastic = Elasticsearch(self._config.es_host)
        hits = []
        for i in range(0, len(docs_ids), 10000):  # max result window size
            s = Search(using=elastic, index=self._config.index).query(Q('ids', values=docs_ids[i:i + 10000]))[0:10000]
            hits = hits + [(hit.meta.id, hit.to_dict()) for hit in s.execute()]
        return hits

    def _get_abstracts_by_ids(self, docs_ids):
        """
        Helper methods to extract abstracts from a list of ElasticSearch documents.
        :param docs_ids: ids of the documents to retrieve
        :return: a dictionary Dict(doc_id: List(abstracts))
        """
        return {doc_id: doc['description'] for doc_id, doc in self._get_es_docs_by_ids(docs_ids)}

    def fetch_long_abstracts(self, uris):
        """
        Retrieve long abstracts (dbo:abstract) of DBpedia entities, from a SPARQL endpoint.
        If more than one abstract is found, only the first will be returned.
        :param uris: list of URIs
        :return: a dictionary Dict(uri, abstract)
        """

        results = []

        for i in range(0, len(uris), 25):
            uris_list = " ".join(map(lambda x: "<%s>" % x, uris[i:i + 25]))
            try:
                self._sparql.setQuery("""
                SELECT distinct ?uri ?abstract {
                  VALUES ?uri { %s }
                  ?uri dbo:abstract ?abstract . 
                  FILTER langMatches( lang(?abstract), "EN" ) 
                }
                """ % uris_list)
            except requests.HTTPError as exception:
                print(exception)
            results += [(result["uri"]["value"], result["abstract"]["value"])
                        for result in self._sparql.query().convert()["results"]["bindings"]]

        return dict(results)

    def fetch_short_abstracts(self, uris):
        """
        Retrieve short abstracts of DBpedia entities, from an ElasticSearch index.
        If more than one abstract is found, only the first will be returned.
        :param uris: list of URIs
        :return: a dictionary Dict(uri, abstract)
        """
        return {entity_uri: entity_abstracts[0] if entity_abstracts else ''
                for entity_uri, entity_abstracts in self._get_abstracts_by_ids(uris).items()}

    def get_relation(self, uri, value):
        """
        Retrieve the relation of a specified value (e.g. dbo:"relation") of DBpedia entities, from a SPARQL endpoint.
        :param uri: list of URI
        :param value: name of the relation
        :return: a list of tuple (uri, name of the relation)
        """

        results = []
        uri = [uri]
        """exceptions = ["-", "--", "?", "."]
        if str(value) in exceptions:
            value = None

        if value is not None:"""
        if isinstance(value, str):
            for i in range(0, len(uri), 25):
                uri = " ".join(map(lambda x: "<%s>" % x, uri[i:i + 25]))
                self._sparql.setQuery("""
                SELECT distinct ?rel 
                  WHERE {{ 
                    %s ?rel ?value . ?value bif:contains '"%s"'} UNION {
                    %s ?rel [rdfs:label ?label] . ?label bif:contains '"%s"' } }
                """ % (uri, value, uri, value))
                results += [result["rel"]["value"]
                            for result in self._sparql.query().convert()["results"]["bindings"]]
                return results
        elif str(value).isnumeric():
            xsd = "^^xsd:integer"
        elif functions.is_float(str(value)):
            xsd = "^^xsd:double"
        elif functions.is_date(str(value)):
            xsd = "^^xsd:date"

        for i in range(0, len(uri), 25):
            uri = " ".join(map(lambda x: "<%s>" % x, uri[i:i + 25]))
            self._sparql.setQuery("""
            SELECT distinct ?rel 
              WHERE {{ 
                %s ?rel '%s'%s} UNION {
                %s ?rel [rdfs:label '%s'%s] } }
            """ % (uri, value, xsd, uri, value, xsd))
            results += [result["rel"]["value"]
                        for result in self._sparql.query().convert()["results"]["bindings"]]

        return results

# class Scorer(Enum):
#     DISTANCE = 0
#     WEIGHTED_RANK = 1
#
#     def score(self, candidates):
