import requests
from elasticsearch import Elasticsearch, TransportError
from elasticsearch_dsl import Q, Search

from data_model.config import ESLookupConfig, WikipediaSearchConfig, DBLookupConfig
from data_model.lookup import LookupResult
from lookup import LookupService


class ESLookup(LookupService):
    """
    Lookup service for ElasticSearch indexes.
    """

    def __init__(self, config: ESLookupConfig = ESLookupConfig('localhost', 'dbpedia')):
        super().__init__(config)

        assert Elasticsearch(self._config.host).ping()  # check if the server is up and running

    def _get_es_docs(self, labels):
        """
        Execute queries against an ElasticSearch index
        :param labels: list of labels
        :return: a generator of tuples <label, list[results]>
        """
        # CRITICAL: DO NOT set the ES client instance as a class member: it is not picklable! -> no parallel execution
        elastic = Elasticsearch(self._config.host)
        for label in labels:
            s = Search(using=elastic, index=self._config.index)
            q = {'value': label.lower()}

            if self._config.fuzziness:
                q['fuzziness'] = self._config.fuzziness
            if self._config.prefix_length:
                q['prefix_length'] = self._config.prefix_length
            if self._config.max_expansions:
                q['max_expansions'] = self._config.max_expansions

            s.query = Q('bool',
                        must=[Q('multi_match', query=label.lower(), fields=['surface_form_keyword'], boost=5),
                              Q({"fuzzy": {"surface_form_keyword": q}})
                              ],
                        should=[Q('match', description=label.lower())])

            s = s[0:self._config.size]

            try:
                yield label, [hit for hit in s.execute()]
            except TransportError:
                yield label, []

    def _lookup(self, labels) -> [LookupResult]:
        """
        Wrap query results into LookupResult
        :param labels: a list of labels
        :return: a list of LookupResult
        """
        return [LookupResult(short_label, [hit['uri'] for hit in candidates])
                for short_label, candidates in self._get_es_docs(labels)]


class WikipediaSearch(LookupService):
    """
    WikipediaSearch lookup service.
    """

    def __init__(self, config: WikipediaSearchConfig = WikipediaSearchConfig('https://en.wikipedia.org/w/api.php')):
        super().__init__(config)
        self._session = requests.Session()

    def _get_wiki_docs(self, labels):
        """
        Execute queries against the Wikipedia Search web service
        :param labels: a list of labels
        :return: a generator of tuples <label, list[results]>
        """
        for label in labels:
            params = {
                "action": "opensearch",
                "search": label,
                "format": "json",
                "limit": self._config.limit,
                "profile": self._config.profile
            }
            yield label, self._session.get(url=self._config.url, params=params).json()

    def _lookup(self, labels) -> [LookupResult]:
        """
        Wrap query results into LookupResult
        :param labels: a list of labels
        :return: a list of LookupResult
        """
        return [LookupResult(short_label, [x.replace("https://en.wikipedia.org/wiki/",
                                                     "http://dbpedia.org/resource/") for x in result[3]])
                if isinstance(result, list) else LookupResult(short_label, [])
                for short_label, result in self._get_wiki_docs(labels)]


class DBLookup(LookupService):
    """
    DBpediaLookup lookup service.
    """

    def __init__(self, config: DBLookupConfig = DBLookupConfig('http://lookup.dbpedia.org/api/search/KeywordSearch')):
        super().__init__(config)
        self._session = requests.Session()
        self._session.headers.update({'Accept': 'application/json'})

    def _get_db_docs(self, labels):
        """
        Execute queries against the DBpedia Lookup web service
        :param labels: a list of labels
        :return: a generator of tuples <label, list[results]>
        """
        for label in labels:
            params = {
                "QueryString": label,
                "MaxHits": self._config.max_hits
            }
            yield label, self._session.get(url=self._config.url, params=params).json()

    def _lookup(self, labels) -> [LookupResult]:
        """
        Wrap query results into LookupResult
        :param labels: a list of labels
        :return: a list of LookupResult
        """
        return [LookupResult(short_label, [x['uri'] for x in result['results']])
                for short_label, result in self._get_db_docs(labels)]