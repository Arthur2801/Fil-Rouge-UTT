"""
Logique métier pour le système RAG (Retrieval Augmented Generation).
Gère la connexion à Elasticsearch et la recherche vectorielle.
"""

from elasticsearch import Elasticsearch
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_elasticsearch_client():
    """
    Initialise et retourne le client Elasticsearch.
    À modifier avec l'URL Cloud fournie par Yasin si nécessaire.
    """
    # Remplacer 'localhost' par l'URL cloud si Yasin a déployé la base
    return Elasticsearch("http://localhost:9200")


def search_deals(query):
    """
    Recherche les deals les plus pertinents dans la base vectorielle.
    """
    es = get_elasticsearch_client()
    
    # Initialisation du modèle d'embedding (doit être le même que celui de Yasin)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Transformation de la question utilisateur en vecteur numérique
    query_vector = embeddings.embed_query(query)
    
    # Requête de recherche par similarité cosinus dans l'index de Yasin
    # On ajoute +1.0 pour éviter les scores négatifs
    search_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector_field') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    
    response = es.search(
        index="dealabs_index",  # Nom de l'index créé par Yasin
        query=search_query
    )
    
    return response['hits']['hits']


def format_context(results):
    """
    Extrait et nettoie les informations des résultats pour le LLM.
    """
    context_text = ""
    for hit in results:
        source = hit['_source']
        context_text += f"\n- Produit: {source.get('title')} | Prix: {source.get('price')}€"
    return context_text