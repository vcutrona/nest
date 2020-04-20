from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

from generators import SimpleGenerator, ContextGenerator


class ESLookup(SimpleGenerator):
    def __init__(self, es_host='titan'):
        super().__init__()
        self._es = Elasticsearch([es_host])
        self._cfg = self._config['ES']

    def search(self, label):
        """
        :param label: keyword to search
        """
        s = Search(using=self._es)
        q = {'value': label.lower()}

        if self._cfg['fuzziness']:
            if self._cfg['fuzziness'] != 'AUTO':
                q['fuzziness'] = int(self._cfg['fuzziness'])
            else:
                q['fuzziness'] = self._cfg['fuzziness']
        if self._cfg['prefix_length']:
            q['prefix_length'] = int(self._cfg['prefix_length'])
        if self._cfg['max_expansions']:
            q['max_expansions'] = int(self._cfg['max_expansions'])

        s.query = Q('bool',
                    must=[Q('multi_match', query=label.lower(), fields=['surface_form_keyword'], boost=5),
                          Q({"fuzzy": {"surface_form_keyword": q}})
                          ],
                    should=[Q('match', description=label.lower())])
        return [hit['uri'] for hit in s.execute()]


class Mantis(ContextGenerator):
    def __init__(self):
        super().__init__()

    def search(self, label):
        pass

    def search_context(self, label, context):
        pass
