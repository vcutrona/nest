from SPARQLWrapper import SPARQLWrapper, JSON
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

from data_model.generator import AbstractCollectorConfig


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

    @staticmethod
    def _cut_abstract(abstract, size) -> str:
        """
        Shorten an abstract to size ``size``.
        :param abstract: the abstract to shorten
        :param size: the new desired length
        :return: the shortened abstract
        """
        return " ".join(abstract.split(" ")[:size]).strip()

    def fetch_long_abstracts(self, uris, max_tokens=None):
        """
        Retrieve long abstracts (dbo:abstract) of DBpedia entities, from a SPARQL endpoint.
        If more than one abstract is found, only the first will be returned.
        :param uris: list of URIs
        :param max_tokens: max length of the abstracts; if longer, they will be cut to fit the desired length.
               If None, the full abstract will be returned.
        :return: a dictionary Dict(uri, abstract)
        """

        results = []

        for i in range(0, len(uris), 25):
            uris_list = " ".join(map(lambda x: "<%s>" % x, uris[i:i + 25]))
            self._sparql.setQuery("""
            SELECT distinct ?uri ?abstract {
              VALUES ?uri { %s }
              ?uri dbo:abstract ?abstract . 
              FILTER langMatches( lang(?abstract), "EN" ) 
            }
            """ % uris_list)

            results += [(result["uri"]["value"], self._cut_abstract(result["abstract"]["value"], max_tokens))
                        for result in self._sparql.query().convert()["results"]["bindings"]]

        return dict(results)

    def fetch_short_abstracts(self, uris, max_tokens=None):
        """
        Retrieve short abstracts of DBpedia entities, from an ElasticSearch index.
        If more than one abstract is found, only the first will be returned.
        :param uris: list of URIs
        :param max_tokens: max length of the abstracts; if longer, they will be cut to fit the desired length.
               If None, the full abstract will be returned.
        :return: a dictionary Dict(uri, abstract)
        """
        return {entity_uri: self._cut_abstract(entity_abstracts[0], max_tokens) if entity_abstracts else ''
                for entity_uri, entity_abstracts in self._get_abstracts_by_ids(uris).items()}

# class Scorer(Enum):
#     DISTANCE = 0
#     WEIGHTED_RANK = 1
#
#     def score(self, candidates):
