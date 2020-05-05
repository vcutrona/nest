import requests
from elasticsearch import Elasticsearch, TransportError
from elasticsearch_dsl import Search, Q

from generators import SimpleGenerator, ContextGenerator


class ESLookup(SimpleGenerator):
    def __init__(self, config='ES'):
        super().__init__(config)
        self._es = Elasticsearch([self._config['host']])

    def search_docs(self, label):
        """
        :param label:
        :return: a list of ES documents
        """
        s = Search(using=self._es)
        q = {'value': label.lower()}

        if self._config['fuzziness']:
            if self._config['fuzziness'] != 'AUTO':
                q['fuzziness'] = int(self._config['fuzziness'])
            else:
                q['fuzziness'] = self._config['fuzziness']
        if self._config['prefix_length']:
            q['prefix_length'] = int(self._config['prefix_length'])
        if self._config['max_expansions']:
            q['max_expansions'] = int(self._config['max_expansions'])

        s.query = Q('bool',
                    must=[Q('multi_match', query=label.lower(), fields=['surface_form_keyword'], boost=5),
                          Q({"fuzzy": {"surface_form_keyword": q}})
                          ],
                    should=[Q('match', description=label.lower())])
        return [hit for hit in s.execute()]

    def _get_es_docs(self, labels):
        for label in labels:
            yield label, self.search_docs(label)

    def _multi_search(self, labels):
        for label, result in self._get_es_docs(self._get_short_labels_set(labels)):
            try:
                yield label, [hit['uri'] for hit in result]
            except TransportError:
                continue


class WikipediaSearch(SimpleGenerator):
    def __init__(self, config='WikipediaSearch'):
        super().__init__(config)
        self._session = requests.Session()

    def _get_wiki_docs(self, labels):
        for label in labels:
            params = {
                "action": "opensearch",
                "search": label,
                "format": "json"
            }

            yield label, self._session.get(url=self._config['url'], params=params).json()

    def _multi_search(self, labels):
        for label, result in self._get_wiki_docs(self._get_short_labels_set(labels)):
            try:
                yield label, [x.replace("https://en.wikipedia.org/wiki/", "http://dbpedia.org/resource/") for x in result[3]]
            except KeyError:
                if 'error' in result and result['error']['code'] == 'request_too_long':
                    continue
                raise Exception  # something different happened -> inspect it


class DBLookup(SimpleGenerator):
    def __init__(self, config='DBLookup'):
        super().__init__(config)
        self._session = requests.Session()
        self._session.headers.update({'Accept': 'application/json'})

    def _get_db_docs(self, labels):
        for label in labels:
            params = {
                "QueryString": label
            }
            yield label, self._session.get(url=self._config['url'], params=params).json()

    def _multi_search(self, labels):
        for label, result in self._get_db_docs(self._get_short_labels_set(labels)):
            yield label, [x['uri'] for x in result['results']]


class Mantis(ContextGenerator):
    def __init__(self, config='Mantis'):
        super().__init__(config)

    def search(self, label, context):
        pass
