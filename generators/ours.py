from generators.baselines import ESLookup
from SPARQLWrapper import SPARQLWrapper, JSON
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np

class FastElmo:
    def __init__(self):
        self._lookup = ESLookup()
        self.elmo_model = ElmoEmbedder()
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

    def get_embeddings_from_sentence(self, text, mode="layer_2", aggregate= False):
        """
        given a string generates the embeddings from ELMO for each token. if aggregate == True, returns the average of the embedding
        i.e., sentence embedding
        :param text: the text  you want to embed
        :param mode: from which layer of ELMO you want the embedding. "mean" gets the embedding of the three elmo layers for each token
        :param aggregate: True => averages the embedding to get a sentence embedding
        :return:
        """
        model_output = self.elmo_model.embed_sentence(text.split())
        embeds = None

        if mode == "layer_2":
            embeds = model_output[2]

        if mode == "layer_1":
            embeds = model_output[1]

        if mode == "layer_0":
            embeds = model_output[0]

        if mode == "mean":
            embeds = (model_output[0] + model_output[1] + model_output[2]) / 3

        if aggregate:
            return np.mean(embeds, axis=0)
        else:
            return embeds


