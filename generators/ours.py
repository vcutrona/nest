from generators.baselines import ESLookup
from SPARQLWrapper import SPARQLWrapper, JSON


class FastBERT:
    def __init__(self):
        self._lookup = ESLookup()
        self._sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self._sparql.setReturnFormat(JSON)
        # TODO init BERT model

    def _fetch_abstract(self, uri):
        self._sparql.setQuery("""
                                SELECT ?abstract
                                WHERE {
                                    <%s> dbo:abstract ?abstract.
                                    FILTER (LANG(?abstract) = 'en' || LANG(?abstract) = '')
                                }
                            """ % uri)
        results = self._sparql.query().convert()
        return {result["abstract"]["value"] for result in results["results"]["bindings"]}

    def search(self, label):
        candidate_uris = self._lookup.search(label)
        candidates = {}
        for uri in candidate_uris:
            candidates[uri] = self._fetch_abstract(uri)

        # TODO refinement/re-ranking

        return candidates
