import multiprocessing as mp

import requests
from elasticsearch import Elasticsearch, TransportError
from elasticsearch_dsl import Search, Q

from generators import SimpleGenerator, ContextGenerator


class ESLookup(SimpleGenerator):
    def __init__(self, config='ES', threads=mp.cpu_count(), chunk_size=10000):
        super().__init__(config, threads, chunk_size)
        if 'size' not in self._config.keys():
            self._config['size'] = '10'  # it is the default value set by ES

        d = dict(self._config)
        self._cache_key_suffix = "%s_%s" % (self.__class__.__name__,
                                            "|".join(sorted(["%s:%s" % (k, d[k])
                                                             for k in d if k in ['size', 'fuzziness',
                                                                                 'prefix_length',   'max_expansions']])))
        assert Elasticsearch(self._config['host']).ping()  # check if the server is up and running

    def _get_cache_key(self, label):
        return label, self._cache_key_suffix

    def _get_es_docs(self, labels):
        # CRITICAL: DO NOT set the ES client instance as a class member: it is not picklable! -> no parallel execution
        elastic = Elasticsearch(self._config['host'])
        config_keys = self._config.keys()
        for label in labels:
            s = Search(using=elastic, index=self._config['index'])
            q = {'value': label.lower()}

            if 'fuzziness' in config_keys:
                if self._config['fuzziness'] != 'AUTO':
                    q['fuzziness'] = int(self._config['fuzziness'])
                else:
                    q['fuzziness'] = self._config['fuzziness']
            if 'prefix_length' in config_keys:
                q['prefix_length'] = int(self._config['prefix_length'])
            if 'max_expansions' in config_keys:
                q['max_expansions'] = int(self._config['max_expansions'])

            s.query = Q('bool',
                        must=[Q('multi_match', query=label.lower(), fields=['surface_form_keyword'], boost=5),
                              Q({"fuzzy": {"surface_form_keyword": q}})
                              ],
                        should=[Q('match', description=label.lower())])
            
            s = s[0:int(self._config['size'])]

            try:
                yield label, [hit for hit in s.execute()]
            except TransportError:
                yield label, []

    def _get_candidates(self, labels):
        return [(short_label, [hit['uri'] for hit in candidates])
                for short_label, candidates in self._get_es_docs(labels)]


class WikipediaSearch(SimpleGenerator):
    def __init__(self, config='WikipediaSearch', threads=3, chunk_size=10000):
        super().__init__(config, threads, chunk_size)
        self._session = requests.Session()

    def _get_wiki_docs(self, labels):
        for label in labels:
            params = {
                "action": "opensearch",
                "search": label,
                "format": "json"
            }
            yield label, self._session.get(url=self._config['url'], params=params).json()

    def _get_candidates(self, labels):
        return [(short_label, [x.replace("https://en.wikipedia.org/wiki/",
                                         "http://dbpedia.org/resource/") for x in result[3]])
                if isinstance(result, list) else (short_label, [])
                for short_label, result in self._get_wiki_docs(labels)]


class DBLookup(SimpleGenerator):
    def __init__(self, config='DBLookup', threads=3, chunk_size=10000):
        super().__init__(config, threads, chunk_size)
        self._session = requests.Session()
        self._session.headers.update({'Accept': 'application/json'})

    def _get_db_docs(self, labels):
        for label in labels:
            params = {
                "QueryString": label
            }
            yield label, self._session.get(url=self._config['url'], params=params).json()

    def _get_candidates(self, labels):
        return [(short_label, [x['uri'] for x in result['results']])
                for short_label, result in self._get_db_docs(labels)]


class Mantis(ContextGenerator):
    def __init__(self, config='Mantis', threads=mp.cpu_count(), chunk_size=1000):
        super().__init__(config, threads, chunk_size)

    def search(self, label, context):
        pass
