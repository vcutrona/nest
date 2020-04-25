import numpy
from SPARQLWrapper import SPARQLWrapper, JSON
from allennlp.commands.elmo import ElmoEmbedder
from scipy.spatial.distance import cosine

from generators import ContextGenerator
from generators.baselines import ESLookup


class FastElmo(ContextGenerator):
    def __init__(self, config='FastElmo'):
        super().__init__(config)
        self._lookup = ESLookup()
        self.elmo_model = ElmoEmbedder(cuda_device=0)
        self._sparql = SPARQLWrapper(self._config['sparql_endpoint'])
        self._sparql.setReturnFormat(JSON)
        # TODO init BERT model

    def _fetch_long_abstract(self, uri):
        self._sparql.setQuery("""
                                SELECT DISTINCT ?abstract
                                WHERE {
                                    <%s> dbo:abstract ?abstract.
                                    FILTER (LANG(?abstract) = 'en' || LANG(?abstract) = '')
                                }
                            """ % uri)
        results = self._sparql.query().convert()
        return [result["abstract"]["value"] for result in results["results"]["bindings"]]

    def _fetch_short_abstract(self, uri):
        self._sparql.setQuery("""
                                SELECT ?abstract
                                WHERE {
                                    <%s> dbo:abstract ?abstract.
                                    FILTER (LANG(?abstract) = 'en' || LANG(?abstract) = '')
                                }
                            """ % uri)
        results = self._sparql.query().convert()
        return {result["abstract"]["value"] for result in results["results"]["bindings"]}

    def _cut_abstract(self, abstract) -> str:
        return " ".join(abstract.split(" ")[:int(self._config['abstract_max_tokens'])]).strip()

    def _get_embeddings_from_sentences(self, sentences, mode="layer_2"):
        """
        Generates the sentence embeddings from ELMO for each sentence in a list of strings.
        :param sentences: the sentences you want to embed
        :param mode: from which layer of ELMO you want the embedding.
                     "mean" gets the embedding of the three elmo layers for each token
        :return:
        """
        model_outputs = self.elmo_model.embed_sentences([sentence.split() for sentence in sentences])
        embeds = []

        if mode == "layer_2":
            embeds = [model_output[2] for model_output in model_outputs]

        if mode == "layer_1":
            embeds = [model_output[1] for model_output in model_outputs]

        if mode == "layer_0":
            embeds = [model_output[0] for model_output in model_outputs]

        if mode == "mean":
            embeds = [(model_output[0] + model_output[1] + model_output[2]) / 3 for model_output in model_outputs]

        embeds = [numpy.mean(embed, axis=0) if embed.size else embed for embed in embeds]

        return embeds

    def search(self, label, context):

        if context:  # no context -> nothing to compare with -> return basic lookup
            context_emb = self._get_embeddings_from_sentences([context])[0]

            if context_emb.size:  # guard: sometimes you get an empty array -> same as no context
                if self._config['abstract'] == 'short':  # from the index
                    candidates = [{'uri': doc['uri'], 'abstracts': doc['description']} for doc in
                                  self._lookup.search_docs(label)]
                else:  # from dbpedia (SPARQL query)
                    candidate_uris = self._lookup.search(label)
                    candidates = []
                    for uri in candidate_uris:
                        candidates.append({'uri': uri, 'abstracts': self._fetch_long_abstract(uri)})

                abstracts = [self._cut_abstract(candidate['abstracts'][0]) if candidate['abstracts'] else ''
                             for candidate in candidates]
                if any(abstracts):  # no abstracts -> nothing to compare with -> return basic lookup
                    abstract_embs = self._get_embeddings_from_sentences(abstracts)

                    for idx, candidate in enumerate(candidates):
                        candidate['distance'] = 2  # init value -> safe for cosine only!
                        if abstract_embs[idx].size:  # guard: sometimes you get an empty array
                            candidate['distance'] = cosine(abstract_embs[idx], context_emb)
                    return [doc['uri'] for doc in sorted(candidates, key=lambda k: k['distance'])]

        return self._lookup.search(label)  # no context -> return basic lookup


# fe = FastElmo()
# print(fe.search("Bobtail", "Cat Female 7 10 Red"))  # breed, species, sex, age, weight, colour
# print(fe.search("Bobtail", ""))  # breed, species, sex, age, weight, colour
# print(fe.search("Beagle", "Dog Male 4 11 Black, tan and white"))  # breed, species, sex, age, weight, colour
# print(fe.search("Boys Don't Cry",
#                 "84 4-May-02 59 USA Peirce 1999 B-    2.7  "))  # CEA_ROUND1 row 5138 -> all abstracts are empty
