import requests
from elasticsearch import Elasticsearch
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

    def search(self, label):
        """
        :param label:
        :return: a list of URIs
        """
        return [hit['uri'] for hit in self.search_docs(label)]


class WikipediaSearch(SimpleGenerator):

    def __init__(self, config='WikipediaSearch'):
        super().__init__(config)
        self._session = requests.Session()

    def search(self, label):

        params = {
            "action": "opensearch",
            "search": label,
            "format": "json"
        }

        response = self._session.get(url=self._config['url'], params=params).json()
        return [x.replace("https://en.wikipedia.org/wiki/", "http://dbpedia.org/resource/") for x in response[3]]


class DBLookup(SimpleGenerator):

    def __init__(self, config='DBLookup'):
        super().__init__(config)
        self._session = requests.Session()

    def search(self, label):
        params = {
            "QueryString": label
        }

        self._session.headers.update({'Accept': 'application/json'})
        response = self._session.get(url=self._config['url'], params=params).json()
        return [x['uri'] for x in response['results']]


class Mantis(ContextGenerator):
    def __init__(self, config='Mantis'):
        super().__init__(config)

    def search(self, label, context):
        pass


print(DBLookup().search("Barack"))
