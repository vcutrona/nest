import os
from typing import Dict, Tuple, List

from SPARQLWrapper import SPARQLWrapper, JSON
from elasticsearch import Elasticsearch

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

    # def _get_es_doc_by_id(self, doc_id):
    #     """
    #     Retrieve a document from an ElasticSearch index.
    #     :param doc_id: id of the document to retrieve
    #     :return: a dictionary
    #     """
    #     elastic = Elasticsearch(self._config.es_host)
    #     try:
    #         return elastic.get(id=doc_id, index=self._config.index)['_source']
    #     except NotFoundError:
    #         return {}

    def _get_es_docs_by_ids(self, docs_ids: List[str]):
        """
        Retrieve several documents from an ElasticSearch index.
        :param docs_ids: ids of the documents to retrieve
        :return: a list of tuples <doc_uri: document_dict>
        """
        if not docs_ids:
            return []
        elastic = Elasticsearch(self._config.es_host)
        return [(doc['_id'], doc['_source'])
                for doc in elastic.mget(body={'ids': docs_ids}, index='dbpedia')['docs'] if '_source' in doc]

    def _get_abstracts_by_ids(self, docs_ids):
        """
        Helper methods to extract abstracts from a list of ElasticSearch documents.
        :param docs_ids: ids of the documents to retrieve
        :return: a dictionary Dict(doc_id: List(abstracts))
        """
        return {doc_id: doc['description'] for doc_id, doc in self._get_es_docs_by_ids(docs_ids)}

    def _get_attribute_for_uris(self, uris, attribute, default_value):
        """
        Helper method that gets an attribute from a list of documents.
        :param uris: a list of URIs
        :param attribute: an attribute of the indexed document
        :param default_value: value to set if the selected attribute is missing
        :return: a dict uri: values
        """
        docs = self._get_es_docs_by_ids(uris)
        attributes = {}
        for uri, doc in docs:
            if attribute in doc:
                attributes[uri] = doc[attribute]
        for uri in uris:
            if uri not in attributes:
                attributes[uri] = default_value
        return attributes

    def get_labels_for_uris(self, uris):
        """
        Returns the labels of DBpedia entities.

        :param uris: a list of URIs
        :return: a dict uri: labels
        """
        labels = self._get_attribute_for_uris(uris, 'surface_form_keyword', [])
        missing = [uri for uri in labels if not labels[uri]]
        for uri in missing:  # TODO bulk query with VALUES
            cached = self._label_cache.get_cached_entry(uri)
            if cached:
                labels[uri] = cached
            self._sparql.setQuery("""
                    SELECT distinct ?label
                    WHERE {
                      <%s> rdfs:label ?label .
                    FILTER (langMatches(lang(?label), "EN") || langMatches(lang(?label), "")) }
                    """ % uri)

            result = [result["label"]["value"]
                      for result in self._sparql.query().convert()["results"]["bindings"]]
            self._label_cache.set_entry(KVPair(uri, result))
            labels[uri] = result
        return labels

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
        return {uri: abstracts[0] if abstracts else ''
                for uri, abstracts in self._get_attribute_for_uris(uris, 'description', '').items()}

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

        chunk_sizes = [x ** 2 for x in range(5, 0, -1)]  # handle too long queries
        for chunk_size in chunk_sizes:
            retry = False
            for i in range(0, len(to_compute), chunk_size):
                query_values = " ".join(map(lambda x: '(<%s> "%s")' % (x[0], x[1].replace('"', '\\"')),
                                            to_compute[i:i + chunk_size]))
                self._sparql.setQuery(query % query_values)
                try:
                    for result in self._sparql.query().convert()["results"]["bindings"]:
                        key, value = (result['entity']['value'], result["value"]['value']), result["rel"]['value']
                        if not (filter_blacklisted and value in PROPERTIES_BLACKLIST):
                            results[key].append(value)
                except:  # if an error occurs, retry with smaller chunks
                    retry = True
                    break
            if not retry:
                break

        self._rels_cache.update_cache_entries([KVPair(sub_obj_pair, (sub_obj_pair, results[sub_obj_pair]))
                                               for sub_obj_pair in to_compute])
        if cached_entries:
            results.update(dict(cached_entries))
        return results

    def get_types_for_uris(self, uris):
        """
        Get the types of a given list of entities

        :param uris: a list of URIs
        :return: a dict uri: types
        """
        # TODO filter out extra types
        return {uri: [t for t in types if t not in TYPES_BLACKLIST]
                for uri, types in self._get_attribute_for_uris(uris, 'type', []).items()}

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

    def get_descriptions_for_uris(self, uris):
        """
        Get the descriptions of a given list of entities

        :param uris: a list of URIs
        :return: a dict uri: descriptions
        """
        return self._get_attribute_for_uris(uris, 'description', [])

    def get_uri_count_for_uris(self, uris):
        """
        Get the uri_count field of a list of documents (from Spotlight lexicalizations)

        :param uris: a list of URIs
        :return: the dict uri: uri_count
        """
        return self._get_attribute_for_uris(uris, 'uri_count', 0)

    def get_in_degree_for_uris(self, uris):
        """
        Get the in_degree field a list of documents (from DBpedia Page Links)

        :param uris: a list of URIs
        :return: a dict uri: in_degree
        """
        return self._get_attribute_for_uris(uris, 'in_degree', 0)

    def get_out_degree_for_uris(self, uris):
        """
        Get the out_degree field a list of documents (from DBpedia Page Links)

        :param uris: a list of URIs
        :return: a dict uri: out_degree
        """
        return self._get_attribute_for_uris(uris, 'out_degree', 0)

    def get_degree_for_uris(self, uris):
        """
        Get the degree (in+out) a list of documents (from DBpedia Page Links)

        :param uris: a list of URIs
        :return: a dict uri: degree
        """
        in_degrees = self.get_in_degree_for_uris(uris)
        out_degrees = self.get_out_degree_for_uris(uris)
        return {uri: in_degrees[uri] + out_degrees[uri] for uri in uris}

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
        val = value.replace('"', '\\"')
        words_set = {'"%s"' % v for v in {val, val.lower(), val.upper(), val.capitalize(), val.title()}}
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
