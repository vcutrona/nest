import os
from enum import Enum
from typing import Dict, Tuple, List

import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

from data_model.kgs import DBpediaWrapperConfig
from utils.caching import CacheWrapper, KVPair

PROPERTIES_BLACKLIST = ['http://dbpedia.org/ontology/abstract',
                        'http://dbpedia.org/ontology/wikiPageWikiLink',
                        'http://www.w3.org/2000/01/rdf-schema#comment',
                        'http://purl.org/dc/terms/subject',
                        ]

TYPES_BLACKLIST = ['http://www.w3.org/2002/07/owl#Thing']


class DBpediaWrapper:
    """
    Wrapper class to retrieve data from a DBpedia SPARQL endpoint.
    """

    def __init__(self, config: DBpediaWrapperConfig = DBpediaWrapperConfig()):
        self._config = config
        assert Elasticsearch(self._config.es_host).ping()

        self._sparql = SPARQLWrapper(self._config.sparql_endpoint, defaultGraph=self._config.default_graph)
        self._sparql.setReturnFormat(JSON)

        self._subj_cache = CacheWrapper(os.path.join(os.path.dirname(__file__), '.cache', 'DBpediaWrapper',
                                                     'subj'), int(4e9))
        self._rels_cache = CacheWrapper(os.path.join(os.path.dirname(__file__), '.cache', 'DBpediaWrapper',
                                                     'rels'), int(4e9))
        self._type_cache = CacheWrapper(os.path.join(os.path.dirname(__file__), '.cache', 'DBpediaWrapper',
                                                     'type'), int(4e9))
        self._label_cache = CacheWrapper(os.path.join(os.path.dirname(__file__), '.cache', 'DBpediaWrapper',
                                                      'label'), int(4e9))

    def _get_es_doc_by_id(self, doc_id):
        docs = self._get_es_docs_by_ids([doc_id])
        if docs:
            return docs[0][1]
        return {}

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
        Returns the labels of a DBpedia entity.

        :param uri: an entity URI
        :return: a list of labels
        """
        doc = self._get_es_doc_by_id(uri)
        if 'surface_form_keyword' in doc:
            return doc['surface_form_keyword']
        else:
            cached = self._label_cache.get_cached_entry(uri)
            if cached:
                return cached
            self._sparql.setQuery("""
                    SELECT distinct ?label
                    WHERE {
                      <%s> rdfs:label ?label .
                    FILTER (langMatches(lang(?label), "EN") || langMatches(lang(?label), "")) }
                    """ % uri)

            result = [result["label"]["value"]
                      for result in self._sparql.query().convert()["results"]["bindings"]]
            self._label_cache.set_entry(KVPair(uri, result))
            return result

    def fetch_long_abstracts(self, uris):
        """
        Retrieve long abstracts (dbo:abstract) of DBpedia entities, from a SPARQL endpoint.
        If more than one abstract is found, only the first will be returned.
        :param uris: list of URIs
        :return: a dictionary Dict(uri, abstract)
        """
        # TODO: cache
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
        return {doc_id: doc['description'][0] if 'description' in doc and doc['description'] else ''
                for doc_id, doc in self._get_es_docs_by_ids(uris)}

    def get_relations(self, subj_obj_pairs: List[Tuple[str, str]],
                      filter_blacklisted=True) -> Dict[Tuple[str, str], List[str]]:
        """
        Retrieve the relations existing between subject-value pairs.
        :param subj_obj_pairs: list of subject-value (URI-literal) pairs
        :param filter_blacklisted: remove blacklisted properties from results
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

        cached_entries, to_compute = self._rels_cache.get_cached_entries(subj_obj_pairs)
        results = {key: [] for key in to_compute}
        for i in range(0, len(to_compute), 25):
            query_values = " ".join(map(lambda x: '(<%s> "%s")' % x, to_compute[i:i + 25]))
            self._sparql.setQuery(query % query_values)
            for result in self._sparql.query().convert()["results"]["bindings"]:
                key, value = (result['entity']['value'], result["value"]['value']), result["rel"]['value']
                if not (filter_blacklisted and value in PROPERTIES_BLACKLIST):
                    results[key].append(value)

        self._rels_cache.update_cache_entries([KVPair(sub_obj_pair, (sub_obj_pair, results[sub_obj_pair]))
                                               for sub_obj_pair in to_compute])
        if cached_entries:
            results.update(dict(cached_entries))
        return results

    def get_types(self, uri):
        """
        Get the types of the given entity

        :param uri: the entity URI
        :return: a list of types (URIs)
        """
        doc = self._get_es_doc_by_id(uri)
        if 'type' in doc:
            types = [t for t in doc['type'] if t not in TYPES_BLACKLIST]
            # if types:
            return types

        # no type in index -> check online
        # cached = self._type_cache.get_cached_entry(uri)
        # if cached:
        #     return cached
        # self._sparql.setQuery("""
        #                     SELECT distinct ?type
        #                     WHERE {
        #                       <%s> a ?type .
        #                     }
        #                     """ % uri)
        #
        # result = [result["type"]["value"]
        #           for result in self._sparql.query().convert()["results"]["bindings"]]
        # self._label_cache.set_entry(KVPair(uri, result))
        # return result
        return []

    def get_descriptions(self, uri):
        """
        Get the descriptions of the given entity

        :param uri: the entity URI
        :return: a list of descriptions
        """
        doc = self._get_es_doc_by_id(uri)
        if 'description' in doc:
            return doc['description']
        return []

    def get_uri_count(self, uri):
        """
        Get the uri_count field of a ES document (from Spotlight lexicalizations)

        :param uri: the entity URI
        :return: the uri_count value
        """
        doc = self._get_es_doc_by_id(uri)
        if 'uri_count' in doc:
            return doc['uri_count']
        return 0

    def get_in_degree(self, uri):
        """
        Get the in_degree field of a ES document (from DBpedia Page Links)

        :param uri: the entity URI
        :return: the in_degree value
        """
        doc = self._get_es_doc_by_id(uri)
        if 'in_degree' in doc:
            return doc['in_degree']
        return 0

    def get_out_degree(self, uri):
        """
        Get the out_degree field of a ES document (from DBpedia Page Links)

        :param uri: the entity URI
        :return: the out_degree value
        """
        doc = self._get_es_doc_by_id(uri)
        if 'out_degree' in doc:
            return doc['out_degree']
        return 0

    def get_degree(self, uri):
        """
        Get the degree (in+out) of a ES document (from DBpedia Page Links)

        :param uri: the entity URI
        :return: the degree value
        """
        doc = self._get_es_doc_by_id(uri)
        if 'out_degree' in doc and 'in_degree' in doc:
            return doc['out_degree'] + doc['in_degree']
        return 0

    def get_subjects(self, prop: str, value: str) -> Dict[str, List[str]]:
        """
        Retrieve all the subject of triples <subject, prop, value>, along with their labels.
        :param prop: the property URI
        :param value: the value of the property
        :return: a dict {uri: [labels]}
        """
        """
        THE FOLLOWING QUERY CAUSES A TIMEOUT
           "SELECT distinct ?subject (str(?label) as ?label)
             WHERE {
             { ?subject <%s> ?value . }
             UNION
             { ?subject <%s> [rdfs:label ?value] . }
             ?subject rdfs:label ?label .
             FILTER(lcase(str(?value))=lcase("%s"))
             FILTER (langMatches(lang(?label), "EN") || langMatches(lang(?label), ""))
           }" % (prop, prop, value)
         REDUCE THE FULL SEARCH TO A SUBSET OF PREDEFINED VALUES
         - value
         - value.capitalize()
         - value.lower()
         - value.upper()
         - value.title()
        """

        cached = self._subj_cache.get_cached_entry((prop, value))
        if cached:
            return cached

        words_set = {'"%s"' % v for v in {value, value.lower(), value.upper(), value.capitalize(), value.title()}}
        words_set.update(["%s@en" % w for w in words_set])
        query = """
        SELECT distinct ?subject (str(?label) as ?label)
        WHERE {
          VALUES ?value {%s}
          { ?subject <%s> ?value . }
          UNION
          { ?subject <%s> [rdfs:label ?value] . }
          ?subject rdfs:label ?label .
          FILTER (langMatches(lang(?label), "EN") || langMatches(lang(?label), ""))
        }""" % (" ".join(words_set), prop, prop)
        self._sparql.setQuery(query)
        results = {}
        for result in self._sparql.query().convert()["results"]["bindings"]:
            subject, label = result["subject"]["value"], result["label"]["value"]
            if subject not in results:
                results[subject] = []
            results[subject].append(label)

        self._subj_cache.set_entry(KVPair((prop, value), results))
        return results


class KGEmbedding(Enum):
    RDF2VEC = 'http://localhost:5999/r2v/uniform'
    WORD2VEC = 'http://localhost:5998/w2v/dbp-300'

    def get_vectors(self, uris):
        """
        Get vectors for the given URIs
        :param uris: a DBpedia resource URI, or a list of DBpedia resource URIs
        :return: a dict {<uri>: <vec>}. <vec> is None if it does not exist a vector for <uri>.
        """
        data = {'uri': uris}
        response = requests.get(self.value, params=data)
        return response.json()
