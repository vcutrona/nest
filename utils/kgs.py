from SPARQLWrapper import SPARQLWrapper, JSON
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

from data_model.kgs import DBpediaWrapperConfig


class DBpediaWrapper:
    """
    Wrapper class to retrieve data from a DBpedia SPARQL endpoint.
    """

    def __init__(self, config: DBpediaWrapperConfig = DBpediaWrapperConfig()):
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

    def get_label(self, uri):
        """
        Retrieve the label of DBpedia entities, from a SPARQL endpoint.

        :param uri: list of URI
        :return: a list of labels
        """

        results = []
        uri = [uri]

        for i in range(0, len(uri), 25):
            uri = " ".join(map(lambda x: "<%s>" % x, uri[i:i + 25]))

            self._sparql.setQuery("""
            SELECT distinct ?value 
              WHERE {
                %s rdfs:label ?value . 
              FILTER langMatches( lang(?value), "EN" ) } """ % uri)

            results += [result["value"]["value"].lower()
                        for result in self._sparql.query().convert()["results"]["bindings"]]

        return results

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
            self._sparql.setQuery("""
            SELECT distinct ?uri ?abstract {
              VALUES ?uri { %s }
              ?uri dbo:abstract ?abstract . 
              FILTER langMatches( lang(?abstract), "EN" ) 
            }
            """ % uris_list)

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

    def get_relations(self, subj_obj_pairs):
        """
        Retrieve the relations existing between subject-value pairs.
        :param subj_obj_pairs: list of subject-value (URI-literal) pairs
        :return: a dict subj_obj_pair: [properties]
        """

        query = """
        SELECT distinct ?entity ?value ?rel
            WHERE {
              VALUES (?entity ?value) {
                  %s
              }
              { ?entity ?rel ?aValue . }
              UNION
              { ?entity ?rel [rdfs:label ?aValue] . }
              FILTER(lcase(str(?aValue))=?value)
            }
        """
        results = {}
        for i in range(0, len(subj_obj_pairs), 25):
            query_values = " ".join(map(lambda x: '(<%s> "%s")' % x, subj_obj_pairs[i:i + 25]))
            self._sparql.setQuery(query % query_values)
            for result in self._sparql.query().convert()["results"]["bindings"]:
                key, value = (result['entity']['value'], result["value"]['value']), result["rel"]['value']
                if key not in results:
                    results[key] = []
                results[key].append(value)

        return results

