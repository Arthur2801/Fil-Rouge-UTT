"""Logique RAG pour la recherche vectorielle de deals Dealabs."""

# Imports de la bibliothèque standard Python
import os

# Imports de bibliothèques tierces
from pymongo import MongoClient  # Client MongoDB pour Atlas
from dotenv import load_dotenv  # Chargement des variables d'environnement
from langchain_huggingface import HuggingFaceEmbeddings  # Modèles d'embedding

# Chargement des variables d'environnement depuis le fichier .env
# (notamment MONGO_URI pour la connexion à MongoDB Atlas)
load_dotenv()


def get_db_collection():
    """Établit la connexion à la collection MongoDB Atlas.
    
    Returns:
        Collection: La collection 'deals' de la base de données.
    """
    # Récupération de l'URI de connexion depuis les variables d'environnement
    uri = os.getenv("MONGO_URI")
    
    # Création du client MongoDB avec l'URI (connexion à Atlas)
    client = MongoClient(uri)
    
    # Sélection de la base de données 'deals_db'
    db = client["deals_db"]
    
    # Retourne la collection 'deals' contenant les deals Dealabs
    return db["deals"]


def get_unique_categories():
    """Récupère les catégories disponibles.
    
    Returns:
        list: Liste des catégories disponibles avec 'Toutes' en premier.
    """
    try:
        # Connexion à la collection MongoDB
        collection = get_db_collection()
        
        # Extraction des valeurs uniques du champ 'group_display_summary'
        # Ce champ contient les catégories des deals (High-Tech, etc.)
        categories = collection.distinct("group_display_summary")
        
        # Retourne la liste avec "Toutes" en premier, puis les catégories triées
        return ["Toutes"] + sorted(categories)
    except Exception:
        # En cas d'erreur de connexion, retourne des catégories par défaut
        return ["Toutes", "High-Tech", "Consoles", "Jeux Vidéo"]

def get_deals_rag(query, category_filter="Toutes", max_price=1200):
    """Effectue une recherche vectorielle hybride.
    
    Args:
        query (str): La requête de recherche de l'utilisateur.
        category_filter (str): Filtre de catégorie (défaut: 'Toutes').
        max_price (int): Prix maximum en euros (défaut: 1200).
    
    Returns:
        list: Liste des deals correspondants avec leurs scores.
    """
    # Connexion à la collection MongoDB contenant les deals
    collection = get_db_collection()
    
    # --- PHASE 1 : GÉNÉRATION DE L'EMBEDDING DE LA REQUÊTE ---
    # Nom du modèle d'embedding pré-entraîné (sentence-transformers)
    # Ce modèle convertit du texte en vecteurs de 384 dimensions
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Initialisation du modèle HuggingFace pour les embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Conversion de la requête utilisateur en vecteur numérique
    # Ce vecteur sera comparé aux vecteurs des deals en base
    query_vector = embeddings.embed_query(query)

    # --- PHASE 2 : CONSTRUCTION DES FILTRES ---
    # Filtre de base : prix inférieur ou égal au budget maximum
    search_filter = {"price": {"$lte": max_price}}
    
    # Si une catégorie spécifique est sélectionnée (pas "Toutes")
    # On ajoute un filtre sur le champ group_display_summary
    if category_filter != "Toutes":
        search_filter["group_display_summary"] = category_filter

    # --- PHASE 3 : PIPELINE D'AGRÉGATION MONGODB ---
    pipeline = [
        {
            # Étape 1 : Recherche vectorielle MongoDB Atlas Search
            "$vectorSearch": {
                # Nom de l'index vectoriel configuré dans Atlas
                "index": "vector_index",
                
                # Champ contenant les vecteurs d'embedding des deals
                "path": "embedding",
                
                # Vecteur de la requête utilisateur à comparer
                "queryVector": query_vector,
                
                # Nombre de candidats à évaluer (plus = meilleur mais plus lent)
                "numCandidates": 100,
                
                # Nombre maximum de résultats à retourner
                "limit": 5,
                
                # Filtres appliqués (prix et catégorie)
                "filter": search_filter
            }
        },
        {
            # Étape 2 : Projection des champs à retourner
            "$project": {
                "title": 1,  # Titre du deal
                "price": 1,  # Prix en euros
                "group_display_summary": 1,  # Catégorie
                "url": 1,  # Lien vers le deal
                "text": 1,  # Description complète
                # Score de similarité vectorielle (0 à 1)
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    # Exécution du pipeline et conversion des résultats en liste Python
    return list(collection.aggregate(pipeline))