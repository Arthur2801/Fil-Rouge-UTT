import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def get_db_collection():
    """Définition de la fonction de connexion pour corriger la NameError."""
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri)
    # Paramètres confirmés par les captures d'Arthur
    db = client["deals_db"]
    return db["deals"]

def get_unique_categories():
    """Récupère les catégories via le champ indexé group_display_summary."""
    try:
        collection = get_db_collection()
        # On utilise le 'Filter Path' exact configuré sur Atlas
        categories = collection.distinct("group_display_summary")
        return ["Toutes"] + sorted(categories)
    except Exception:
        return ["Toutes", "High-Tech", "Consoles", "Jeux Vidéo"]

def get_deals_rag(query, category_filter="Toutes", max_price=1200):
    """Recherche vectorielle hybride corrigée."""
    collection = get_db_collection()
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_vector = embeddings.embed_query(query)

    # Filtrage basé sur les Filter Fields d'Atlas
    search_filter = {"price": {"$lte": max_price}}
    if category_filter != "Toutes":
        search_filter["group_display_summary"] = category_filter

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index", #
                "path": "embedding",
                #"path": "vector_field",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 5,
                "filter": search_filter
            }
        },
        {
            "$project": {
                "title": 1, "price": 1, "group_display_summary": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    return list(collection.aggregate(pipeline))