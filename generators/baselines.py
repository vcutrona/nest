from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q


class ESLookup:
    def __init__(self):
        self._es = Elasticsearch(['titan'])

    def search(self, label, fuzziness=2, prefix_length=0, max_expansions=100):
        """
        :param label: keyword to search
        :param fuzziness: edit distance - a positive number, or "AUTO"
        :param prefix_length: chars that must be equal (fuzziness not applied to the prefix)
        :param max_expansions: max variations to compute
        """
        s = Search(using=self._es)
        q = {
            'value': label.lower(),
            'fuzziness': fuzziness,
            'prefix_length': prefix_length,
            'max_expansions': max_expansions
        }
        s.query = Q('bool',
                    must=[Q('multi_match', query=label.lower(), fields=['surface_form_keyword'], boost=5),
                          Q({"fuzzy": {"surface_form_keyword": q}})
                          ],
                    should=[Q('match', description=label.lower())])
        return [hit['uri'] for hit in s.execute()]


class Mantis:
    def __init__(self):
        pass

    def search(self, label):
        pass

    def search_context(self, label, context):
        pass
