# NEST
This repository contains Python code to replicate the experiments for NEST. We propose to use neural models 
for type prediction and type representation to improve the type enrichment strategies that can be used in
existing matching pipelines in a modular fashion. In particular:
- type enrichment for type-based filtering: neural type prediction algorithms to enrich the types of candidate entities
  with types predicted by a neural network.
- type enrichment for entity similarity with distributed representations: distributed type representations to enrich
  entity embeddings and make their similarity more aware of their types.

## Reference
This work is under review (ESWC 2021):
> Cutrona, V., Puleri, G., Bianchi, F., and Palmonari, M. (2020). NEST: Neural Soft Type Constraints to 
> Improve Entity Linking in Tables. ESWC 2021 (under review).

## How to use
### Installing dependencies
The code is developed for Python 3.8.
Install all the required packages listed in the `requirements.txt` file.
```shell
virtualenv -p python3.8 venv # we suggest to create a virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

### Prepare utils data
Neural networks and type embeddings are available in the utils_data.zip file. The following files must be extracted under
the `utils/data` directory:
- `abs2vec_pred.keras` and `abs2vec_pred_classes.pkl`: the neural network based on BERT embeddings, and the list of
  its predictable classes
- `rdf2vec_pred.keras` and `rdf2vec_pred_classes.pkl`: the neural network based on RDF2Vec embeddings, and the list of
  its predictable classes
- `dbpedia_owl2vec`: typed embedding for DBpedia 2016-10 generated using OWL2Vec
- `tee.wv`: typed embedding for DBpedia 2016-10 generated using TEE

We release a set of Docker images to run the above predictors as a service; also, some other embedding models (e.g.,
RDF2Vec) have been exposed as a service.
Download abs2vec embeddings from
[GDrive](https://drive.google.com/uc?id=10kb6iPbbm_o8jZlvyZYxxne8LbHyknOR&export=download) and set its path in 
the `docker-compose.yml` file.
Finally, start the containers:
```shell
docker-compose up -d
```

### Benchmark datasets
Benchmark datasets can be downloaded from
[GDrive](https://drive.google.com/uc?id=1iPa2MCg8NRGJHCdsRacBFRXNMA1VHCg5&export=download). Unzip the file under
the `datasets` folder.

### Create the index
Replicating our experiments requires to initialize an index that contains DBpedia 2016-10. We created it by using
[ElasticPedia](https://pypi.org/project/elasticpedia/), then manually adding the
[Wikipedia anchor texts](http://downloads.dbpedia.org/2016-10/core-i18n/en/anchor_text_en.ttl.bz2), labels from the
[Lexicalization dataset](https://sourceforge.net/projects/dbpedia-spotlight/files/archiving/lexicalizations.tgz/download),
and the in- and out-degree from the
[Page Link dataset](http://downloads.dbpedia.org/2016-10/core-i18n/en/page_links_en.ttl.bz2).
Lastly, we re-indexed the index with the following mappings:
```json
{
  "dbpedia": {
    "mappings": {
      "properties": {
        "category": {
          "type": "text",
          "fields": {
            "keyword": {
              "type": "keyword"
            }
          }
        },
        "description": {
          "type": "text",
          "fields": {
            "keyword": {
              "type": "keyword"
            }
          }
        },
        "direct_type": {
          "type": "text",
          "fields": {
            "keyword": {
              "type": "keyword"
            }
          }
        },
        "in_degree": {
          "type": "integer"
        },
        "nested_surface_form": {
          "type": "nested",
          "properties": {
            "surface_form_keyword": {
              "type": "text",
              "fields": {
                "keyword": {
                  "type": "keyword"
                },
                "ngram": {
                  "type": "text",
                  "analyzer": "my_analyzer"
                }
              }
            }
          }
        },
        "out_degree": {
          "type": "integer"
        },
        "surface_form_keyword": {
          "type": "text",
          "fields": {
            "keyword": {
              "type": "keyword"
            },
            "ngram": {
              "type": "text",
              "analyzer": "my_analyzer"
            }
          }
        },
        "type": {
          "type": "text",
          "fields": {
            "keyword": {
              "type": "keyword"
            }
          }
        },
        "uri": {
          "type": "text",
          "fields": {
            "keyword": {
              "type": "keyword"
            }
          }
        },
        "uri_count": {
          "type": "integer"
        },
        "uri_prob": {
          "type": "float"
        }
      }
    },
    "settings": {
      "index": {
        "analysis": {
          "analyzer": {
            "my_analyzer": {
              "tokenizer": "my_tokenizer"
            }
          },
          "tokenizer": {
            "my_tokenizer": {
              "token_chars": [
                "letter",
                "digit"
              ],
              "min_gram": "3",
              "type": "ngram",
              "max_gram": "3"
            }
          }
        }
      }
    }
  }
}
```
We are planning to release a dump of our index.
Replace the host name `titan` with the endpoint of your Elasticsearch index in the following files:
- `utils/nn.py`
- `utils/embeddings.py`
- `data_model/kgs.py`
- `run_experiments.py`

### Run the experiments
Run the script as follows to initialize and run the models described in our paper:
```shell
python run_experiments.py
```
Results are printed in the `eswc_experiments.json` file.

### People
- Vincenzo Cutrona, University of Milano - Bicocca ([vincenzo.cutrona@unimib.it](mailto:vincenzo.cutrona@unimib.it))
- Gianluca Puleri, University of Milano - Bicocca ([gianluca.puleri@unimib.it](mailto:gianluca.puleri@unimib.it))
- Federico Bianchi, Bocconi University ([f.bianchi@unibocconi.it](mailto:f.bianchi@unibocconi.it))
- Matteo Palmonari, University of Milano - Bicocca ([matteo.palmonari@unimib.it](mailto:matteo.palmonari@unimib.it))
