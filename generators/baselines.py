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

    def search(self, label):
        """
        :param label:
        :return: a list of URIs
        """
        results = []
        for short_label in self._get_short_labels(label):
            try:
                results = results + [hit['uri'] for hit in self.search_docs(short_label)]
            except TransportError:  # it happens on very long labels
                continue  # labels are sorted by len (reverse) -> skip this iteration

        return list(dict.fromkeys(results))


class WikipediaSearch(SimpleGenerator):
    def __init__(self, config='WikipediaSearch'):
        super().__init__(config)
        self._session = requests.Session()

    def search(self, label):
        results = []
        for short_label in self._get_short_labels(label):

            params = {
                "action": "opensearch",
                "search": short_label,
                "format": "json"
            }

            response = self._session.get(url=self._config['url'], params=params).json()

            try:
                results = results + [x.replace("https://en.wikipedia.org/wiki/", "http://dbpedia.org/resource/")
                                     for x in response[3]]
            except KeyError:  # the result is a dict, not a list -> error
                if 'error' in response and response['error']['code'] == 'request_too_long':
                    continue
                raise Exception  # something different happened -> inspect it

        return list(dict.fromkeys(results))


class DBLookup(SimpleGenerator):
    def __init__(self, config='DBLookup'):
        super().__init__(config)
        self._session = requests.Session()

    def search(self, label):
        results = []
        for short_label in self._get_short_labels(label):
            params = {
                "QueryString": short_label
            }

            self._session.headers.update({'Accept': 'application/json'})
            response = self._session.get(url=self._config['url'], params=params).json()
            results = results + [x['uri'] for x in response['results']]

        return list(dict.fromkeys(results))


class Mantis(ContextGenerator):
    def __init__(self, config='Mantis'):
        super().__init__(config)

    def search(self, label, context):
        pass
