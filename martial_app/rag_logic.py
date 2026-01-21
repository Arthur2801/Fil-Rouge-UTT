"""
Logique RAG (Retrieval-Augmented Generation) pour Dealabs.

Ce module implémente la logique de recherche vectorielle et d'interaction avec
le LLM pour l'application de recherche de deals Dealabs.

Composants principaux:
    - Connexion à MongoDB Atlas pour la recherche vectorielle
    - Génération d'embeddings avec HuggingFace (sentence-transformers)
    - Intégration avec GROQ LLM pour la génération de réponses intelligentes
    - Gestion des filtres (catégorie, prix)

Architecture RAG:
    1. Vectorisation de la requête utilisateur
    2. Recherche de similarité dans MongoDB Atlas
    3. Récupération des deals pertinents
    4. Construction du prompt avec contexte
    5. Génération de réponse par le LLM

Auteur: Projet Master Big Data
Date: Janvier 2026
"""

# Imports de la bibliothèque standard Python
import os

# Imports de bibliothèques tierces pour MongoDB et embeddings
from pymongo import MongoClient  # Client MongoDB pour Atlas
from dotenv import load_dotenv  # Chargement des variables d'environnement
from langchain_huggingface import HuggingFaceEmbeddings  # Embeddings

# Imports pour l'intégration LLM
from langchain_groq import ChatGroq  # Client GROQ pour le LLM
from langchain_core.prompts import ChatPromptTemplate  # Templates de prompts
from langchain_core.runnables import RunnablePassthrough  # Chaînes LangChain

# --- CHARGEMENT DES VARIABLES D'ENVIRONNEMENT ---
# Charge les variables depuis le fichier .env (MONGO_URI, GROQ_API_KEY)
load_dotenv()


def get_llm_answer(query, context_documents):
    """
    Génère une réponse intelligente en utilisant un LLM (GROQ).
    
    Cette fonction prend une requête utilisateur et des documents de contexte
    (deals trouvés), puis utilise un modèle de langage pour générer une
    réponse pertinente et comparative.
    
    Processus:
        1. Récupération de la clé API GROQ depuis l'environnement
        2. Initialisation du modèle LLM (llama-3.3-70b-versatile)
        3. Construction du prompt avec instructions et contexte
        4. Préparation du contexte (fusion des textes des deals)
        5. Exécution de la chaîne LangChain
        6. Retour de la réponse générée
    
    Args:
        query (str): La question/requête de l'utilisateur.
        context_documents (list): Liste de dictionnaires contenant les deals
                                 trouvés par la recherche vectorielle.
    
    Returns:
        str: La réponse générée par le LLM, analysant et comparant les deals.
    
    Raises:
        Exception: Si la clé API est manquante ou si l'appel au LLM échoue.
    
    Example:
        >>> docs = [{"text": "iPhone 13 à 599€"}, {"text": "Samsung S21..."}]
        >>> answer = get_llm_answer("meilleur smartphone", docs)
        >>> print(answer)
        "Le iPhone 13 offre le meilleur rapport qualité-prix..."
    """
    # --- RÉCUPÉRATION DE LA CLÉ API ---
    # Récupère la clé gsk_... depuis les variables d'environnement
    api_key = os.getenv("GROQ_API_KEY")

    # --- INITIALISATION DU LLM ---
    # Configuration du client GROQ avec:
    # - groq_api_key: authentification
    # - model_name: llama-3.3-70b-versatile (modèle puissant et rapide)
    # - temperature: 0 (réponses déterministes, pas de créativité)
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    # --- CONSTRUCTION DU PROMPT ---
    # Template de prompt avec instructions détaillées pour le LLM
    # Utilise des placeholders {context} et {question}
    template = """
    Tu es un assistant expert en bons plans Dealabs. 
    Analyse les deals fournis ci-dessous et aide l'utilisateur à faire le 
    meilleur choix.
    
    Instructions :
    - Si la question porte directement sur les deals fournis, réponds de 
      manière précise et comparative.
    - Si la question ne peut pas être répondue avec les deals fournis, 
      explique brièvement pourquoi, puis présente quand même les deals 
      trouvés en expliquant en quoi ils peuvent être pertinents.
    - Compare les prix, caractéristiques et avantages des différents deals.
    - Sois concis mais informatif (maximum 150 mots).

    CONTEXTE (Deals trouvés) :
    {context}

    QUESTION :
    {question}

    RÉPONSE :
    """
    
    # Conversion du template string en objet ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(template)

    # --- PRÉPARATION DU CONTEXTE ---
    # Fusion de tous les textes des deals en un seul string
    # Séparés par deux retours à la ligne pour la lisibilité
    # Utilisation de .get() pour éviter les KeyError si 'text' manque
    context_text = "\n\n".join([
        doc.get('text', '') for doc in context_documents
    ])

    # --- GÉNÉRATION DE LA RÉPONSE ---
    # Construction de la chaîne LangChain : prompt | llm
    # L'opérateur | connecte les composants de manière fluide
    chain = prompt | llm
    
    # Invocation de la chaîne avec les paramètres
    # .invoke() exécute le prompt avec les valeurs fournies
    response = chain.invoke({
        "context": context_text,
        "question": query
    })
    
    # Retour du contenu textuel de la réponse
    # Retour du contenu textuel de la réponse
    return response.content


def get_db_collection():
    """
    Établit la connexion à la collection MongoDB Atlas.
    
    Cette fonction crée une connexion à MongoDB Atlas en utilisant l'URI
    stockée dans les variables d'environnement, puis retourne la collection
    'deals' de la base de données 'deals_db'.
    
    Architecture MongoDB:
        - Base de données: deals_db
        - Collection: deals (contient tous les deals Dealabs)
        - Index vectoriel: vector_index (pour la recherche sémantique)
    
    Returns:
        pymongo.collection.Collection: La collection 'deals' prête à être
                                       utilisée pour les requêtes.
    
    Raises:
        pymongo.errors.ConfigurationError: Si l'URI est invalide.
        pymongo.errors.ConnectionFailure: Si la connexion échoue.
    
    Example:
        >>> collection = get_db_collection()
        >>> count = collection.count_documents({})
        >>> print(f"Nombre de deals: {count}")
    """
    # --- RÉCUPÉRATION DE L'URI ---
    # L'URI contient toutes les informations de connexion:
    # - Identifiants MongoDB Atlas
    # - Adresse du cluster
    # - Options de connexion
    uri = os.getenv("MONGO_URI")
    
    # --- CRÉATION DU CLIENT MONGODB ---
    # MongoClient gère la connexion au cluster Atlas
    # Utilise un pool de connexions pour les performances
    client = MongoClient(uri)
    
    # --- SÉLECTION DE LA BASE DE DONNÉES ---
    # Accès à la base 'deals_db' contenant nos données Dealabs
    db = client["deals_db"]
    
    # --- RETOUR DE LA COLLECTION ---
    # Retourne la collection 'deals' pour effectuer des requêtes
    return db["deals"]


def get_unique_categories():
    """
    Récupère toutes les catégories de deals disponibles.
    
    Cette fonction interroge MongoDB pour obtenir toutes les valeurs uniques
    du champ 'group_display_summary' (catégories), les filtre, les trie,
    et ajoute l'option "Toutes" en première position.
    
    Traitement:
        1. Connexion à MongoDB
        2. Extraction des catégories distinctes
        3. Filtrage des valeurs None/vides
        4. Tri alphabétique
        5. Ajout de "Toutes" au début
    
    Returns:
        list: Liste de strings contenant "Toutes" suivi des catégories
              triées alphabétiquement.
              Exemple: ["Toutes", "Consoles", "High-Tech", "Jeux Vidéo"]
    
    Raises:
        Exception: En cas d'erreur de connexion, retourne des catégories
                   par défaut.
    
    Example:
        >>> categories = get_unique_categories()
        >>> print(categories[0])  # Affiche "Toutes"
        >>> print(len(categories))  # Nombre de catégories + 1
    """
    try:
        # --- CONNEXION À MONGODB ---
        collection = get_db_collection()
        
        # --- EXTRACTION DES CATÉGORIES DISTINCTES ---
        # distinct() retourne toutes les valeurs uniques d'un champ
        # Équivalent SQL: SELECT DISTINCT group_display_summary
        categories = collection.distinct("group_display_summary")
        
        # --- FILTRAGE DES VALEURS INVALIDES ---
        # Liste comprehension pour éliminer:
        # - None (valeurs nulles)
        # - Chaînes vides ("")
        # - Valeurs falsy
        categories = [cat for cat in categories if cat]
        
        # --- CONSTRUCTION DE LA LISTE FINALE ---
        # "Toutes" en premier, suivi des catégories triées
        # sorted() effectue un tri alphabétique
        return ["Toutes"] + sorted(categories)
        
    except Exception as e:
        # --- GESTION DES ERREURS ---
        # Si la connexion échoue, affiche l'erreur dans la console
        print(f"Erreur lors de la récupération des catégories: {e}")
        
        # Retourne des catégories par défaut pour ne pas bloquer l'app
        # Retourne des catégories par défaut pour ne pas bloquer l'app
        return ["Toutes", "High-Tech", "Consoles", "Jeux Vidéo"]


def get_deals_rag(query, category_filter="Toutes", max_price=1200):
    """
    Effectue une recherche vectorielle hybride sur les deals Dealabs.
    
    Cette fonction implémente le cœur du système RAG (Retrieval-Augmented
    Generation). Elle convertit la requête utilisateur en vecteur, effectue
    une recherche de similarité dans MongoDB Atlas, applique des filtres,
    et retourne les deals les plus pertinents.
    
    Processus détaillé:
        1. Connexion à MongoDB Atlas
        2. Génération de l'embedding de la requête (vectorisation)
        3. Construction des filtres (prix, catégorie)
        4. Exécution du pipeline d'agrégation MongoDB:
           - $vectorSearch: recherche de similarité cosinus
           - $project: sélection et calcul du score
        5. Retour des résultats triés par pertinence
    
    Technologie:
        - Modèle: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
        - Index: MongoDB Atlas Search (index vectoriel)
        - Métrique: Similarité cosinus
        - Limite: 5 meilleurs résultats
    
    Args:
        query (str): La requête de recherche en langage naturel de
                    l'utilisateur.
                    Exemple: "ordinateur portable pour jouer"
        
        category_filter (str, optional): Filtre de catégorie à appliquer.
                                        Défaut: "Toutes" (pas de filtre).
                                        Exemples: "High-Tech", "Consoles"
        
        max_price (int, optional): Prix maximum en euros pour filtrer les
                                   deals. Défaut: 1200.
    
    Returns:
        list: Liste de dictionnaires représentant les deals trouvés.
              Chaque dictionnaire contient:
              - title (str): Titre du deal
              - price (float): Prix en euros
              - group_display_summary (str): Catégorie
              - url (str): Lien vers le deal
              - text (str): Description complète
              - score (float): Score de pertinence (0 à 1)
              
              Exemple:
              [
                  {
                      "title": "PC Gamer RTX 3060",
                      "price": 899.99,
                      "group_display_summary": "High-Tech",
                      "url": "https://dealabs.com/...",
                      "text": "Description complète...",
                      "score": 0.87
                  },
                  ...
              ]
    
    Raises:
        pymongo.errors.OperationFailure: Si l'index vectoriel n'existe pas
                                         ou est mal configuré.
        Exception: Autres erreurs de connexion ou d'exécution.
    
    Example:
        >>> # Recherche simple sans filtres
        >>> results = get_deals_rag("smartphone")
        >>> print(f"Trouvé {len(results)} deals")
        
        >>> # Recherche avec filtres
        >>> results = get_deals_rag(
        ...     query="ordinateur portable",
        ...     category_filter="High-Tech",
        ...     max_price=1000
        ... )
        >>> for deal in results:
        ...     print(f"{deal['title']}: {deal['price']}€ "
        ...           f"(score: {deal['score']:.2f})")
    """
    # --- CONNEXION À MONGODB ---
    # Récupération de la collection contenant les deals
    collection = get_db_collection()
    
    # --- PHASE 1 : GÉNÉRATION DE L'EMBEDDING ---
    # Configuration du modèle de sentence-transformers
    # Ce modèle convertit du texte en vecteurs de 384 dimensions
    # Les vecteurs similaires sémantiquement seront proches dans l'espace
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Initialisation du modèle HuggingFace pour les embeddings
    # Cache automatiquement le modèle après le premier téléchargement
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Conversion de la requête utilisateur en vecteur numérique
    # Ce vecteur sera comparé aux vecteurs pré-calculés des deals
    # Exemple: "ordinateur gaming" → [0.234, -0.123, 0.567, ...]
    query_vector = embeddings.embed_query(query)

    # --- PHASE 2 : CONSTRUCTION DES FILTRES ---
    # Initialisation du dictionnaire de filtres avec le prix maximum
    # $lte = "less than or equal" (inférieur ou égal)
    search_filter = {"price": {"$lte": max_price}}
    
    # Ajout conditionnel du filtre de catégorie
    # Si "Toutes" est sélectionné, pas de filtre de catégorie
    if category_filter != "Toutes":
        search_filter["group_display_summary"] = category_filter

    # --- PHASE 3 : PIPELINE D'AGRÉGATION MONGODB ---
    # Construction du pipeline d'agrégation en 2 étapes
    pipeline = [
        {
            # --- ÉTAPE 1 : RECHERCHE VECTORIELLE ---
            # $vectorSearch est une opération spécifique à Atlas Search
            "$vectorSearch": {
                # Nom de l'index vectoriel configuré dans Atlas
                # Doit correspondre au nom dans MongoDB Atlas
                "index": "vector_index",
                
                # Champ contenant les vecteurs d'embedding pré-calculés
                # Chaque deal a son vecteur stocké dans ce champ
                "path": "embedding",
                
                # Vecteur de la requête utilisateur à comparer
                # MongoDB calcule la similarité cosinus avec chaque deal
                "queryVector": query_vector,
                
                # Nombre de candidats à évaluer dans la première phase
                # Plus élevé = meilleur rappel mais plus lent
                # Recommandé: 10-20x le limit
                "numCandidates": 100,
                
                # Nombre maximum de résultats finaux à retourner
                # Top-5 deals les plus similaires
                "limit": 5,
                
                # Filtres pré-calculés appliqués avant la recherche
                # Réduit l'espace de recherche pour les performances
                "filter": search_filter
            }
        },
        {
            # --- ÉTAPE 2 : PROJECTION DES CHAMPS ---
            # Sélection des champs à retourner dans les résultats
            "$project": {
                "title": 1,  # Titre du deal
                "price": 1,  # Prix en euros
                "group_display_summary": 1,  # Catégorie du deal
                "url": 1,  # Lien vers la page Dealabs
                "text": 1,  # Description complète du deal
                
                # Score de similarité vectorielle (métadonnée spéciale)
                # $meta "vectorSearchScore" retourne le score de 0 à 1
                # 1 = correspondance parfaite, 0 = aucune similarité
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    # --- EXÉCUTION DU PIPELINE ---
    # aggregate() exécute le pipeline et retourne un curseur
    # list() convertit le curseur en liste Python
    # Les résultats sont déjà triés par score décroissant
    return list(collection.aggregate(pipeline))