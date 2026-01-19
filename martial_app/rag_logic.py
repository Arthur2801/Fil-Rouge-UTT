"""
Logique RAG utilisant MongoDB Atlas Vector Search.
Utilise le modèle all-MiniLM-L6-v2 pour la cohérence avec l'indexation.
"""

import os
from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_deals_rag(query, category_filter="Toutes", max_price=1000):
    """
    Exécute une recherche vectorielle filtrée sur MongoDB Atlas.
    """
    # Récupération sécurisée de l'URI (Chaîne de connexion de Yassine)
    mongo_uri = os.getenv(
        "MONGO_URI", 
        "mongodb+srv://dev_team:filrougeutt@cluster0.ou16sxf.mongodb.net/?appName=Cluster0"
    )
    
    client = MongoClient(mongo_uri)
    db = client["deals_db"]
    collection = db["deals"]

    # Initialisation du modèle d'embedding (Identique à celui de Yassine)
    model_path = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    query_vector = embeddings.embed_query(query)

    # Configuration des Filter Fields (Pré-filtrage Atlas)
    search_filter = {"price": {"$lte": max_price}}
    if category_filter != "Toutes":
        search_filter["category"] = category_filter

    # Pipeline de recherche vectorielle
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "vector_field",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 5,
                "filter": search_filter
            }
        },
        {
            "$project": {
                "_id": 0,
                "title": 1,
                "price": 1,
                "category": 1,
                "url": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    try:
        return list(collection.aggregate(pipeline))
    except Exception as error:
        print(f"Erreur lors de la requête MongoDB : {error}")
        return []