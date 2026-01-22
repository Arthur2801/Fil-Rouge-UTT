#Script d'indexation des deals dans Elasticsearch avec embeddings vectoriels.
#Utilise le modèle SentenceTransformer en local pour générer les embeddings.


import json
import random
import time
from typing import Optional

# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


# Modèle d'embedding local
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384

# Elasticsearch
ES_HOST = "http://localhost:9200"
ES_INDEX = "deals"

# Dossier de données
DATA_DIR = r"C:\Users\yassi\Desktop\Cours UTT\Fil-Rouge-UTT\Repo\Fil-Rouge-UTT\data"

# Performance
BATCH_SIZE = 50  # Nombre de documents par batch pour l'indexation bulk

# Charger le modèle d'embedding en local
print(f"🔄 Chargement du modèle {MODEL_ID}...")
model = SentenceTransformer(MODEL_ID)
print("✓ Modèle chargé")

es = Elasticsearch(ES_HOST, request_timeout=60)

# Mapping de l'index Elasticsearch

MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "-1"  # Désactiver le refresh pendant l'indexation
    },
    "mappings": {
        "properties": {
            "title": {"type": "text", "analyzer": "french"},
            "text": {"type": "text", "analyzer": "french"},
            "html_stripped_description": {"type": "text", "analyzer": "french"},
            "group_display_summary": {"type": "keyword"},
            "price": {"type": "float"},
            "url": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

def embed_text(text: str) -> Optional[list[float]]:
    
    # Génère un embedding pour un texte donné via le modèle local.

    try:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Erreur embedding: {e}")
        return None


def pick_best_comment(deal: dict) -> Optional[str]:
    
    #Sélectionne le commentaire avec le plus de likes.
    
    
    comments = deal.get("comments", [])
    if not comments:
        return None

    max_likes = max(
        c.get("reaction_counters", {}).get("like", 0) for c in comments
    )

    best_comments = [
        c for c in comments
        if c.get("reaction_counters", {}).get("like", 0) == max_likes
    ]

    return random.choice(best_comments).get("content_unformatted") # En cas d'égalité, en choisit un au hasard.


def build_embedding_text(deal: dict) -> str:
    
    # transforme un deal en texte sémantique prêt à être vectorisé.

    parts = []

    if title := deal.get("title"):
        parts.append(f"Titre : {title}.")

    if category := deal.get("group_display_summary"):
        parts.append(f"Catégorie : {category}.")

    if description := deal.get("html_stripped_description"):
        # Limiter la description pour éviter les textes trop longs
        parts.append(f"Description : {description[:500]}.")

    if (price := deal.get("price")) is not None:
        parts.append(f"Prix : {price} euros.")

    if temp := deal.get("temperature_level"):
        parts.append(f"Niveau d'attractivité : {temp}.")

    if best_comment := pick_best_comment(deal):
        parts.append(f"Commentaire populaire : {best_comment[:200]}.")

    return " ".join(parts)


def parse_price(price) -> Optional[float]:
    #Convertit un prix en float de manière sécurisée.

    if price is None:
        return None
    try:
        return float(price)
    except (ValueError, TypeError):
        return None


def prepare_document(deal_id: str, deal: dict) -> Optional[dict]:
    # Prépare un document pour l'indexation bulk.

    text = build_embedding_text(deal)
    embedding = embed_text(text)
    
    if not embedding or len(embedding) != EMBEDDING_DIMS:
        return None # Retourne None si l'embedding échoue

    return {
        "_index": ES_INDEX,
        "_id": deal_id,
        "_source": {
            "title": deal.get("title"),
            "text": text,
            "html_stripped_description": deal.get("html_stripped_description"),
            "group_display_summary": deal.get("group_display_summary"),
            "price": parse_price(deal.get("price")),
            "url": deal.get("deal_uri"),
            "embedding": embedding
        }
    }


def create_index():
    # Crée ou recrée l'index Elasticsearch

    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
        print(f"✓ Index '{ES_INDEX}' supprimé")

    es.indices.create(index=ES_INDEX, body=MAPPING)
    print(f"✓ Index '{ES_INDEX}' créé")


def index_deals(data: dict):
    #Indexe les deals par batch avec affichage de progression

    total = len(data)
    indexed = 0
    failed = 0
    
    deals_list = list(data.items())
    
    for i in range(0, total, BATCH_SIZE):
        batch = deals_list[i:i + BATCH_SIZE]
        actions = []
        
        for deal_id, deal in batch:
            doc = prepare_document(deal_id, deal)
            if doc:
                actions.append(doc)
            else:
                failed += 1
        
        if actions:
            success, errors = bulk(es, actions, raise_on_error=False)
            indexed += success
            if errors:
                failed += len(errors)
        
        # Affichage de la progression
        progress = min(i + BATCH_SIZE, total)
        print(f"\r⏳ Progression: {progress}/{total} ({100*progress//total}%)", end="", flush=True)
    
    print()  # Nouvelle ligne après la barre de progression
    return indexed, failed


def finalize_index():
    # Réactive le refresh et force un refresh de l'index

    es.indices.put_settings(
        index=ES_INDEX,
        body={"index": {"refresh_interval": "1s"}}
    )
    es.indices.refresh(index=ES_INDEX)


def load_all_deals(data_dir: str) -> dict:
    #charge tous les fichiers JSON du dossier data et les fusionne
    
    import os
    import glob
    
    all_deals = {}
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    print(f"📂 Fichiers trouvés: {len(json_files)}")
    
    for filepath in sorted(json_files):
        filename = os.path.basename(filepath)
        print(f"   📄 Chargement de {filename}...")
        with open(filepath, "r", encoding="utf-8") as f:
            file_data = json.load(f)
            # Si c'est une liste, convertir en dict avec index
            if isinstance(file_data, list):
                for i, deal in enumerate(file_data):
                    deal_id = deal.get("deal_id") or deal.get("id") or f"{filename}_{i}"
                    all_deals[str(deal_id)] = deal
            # Si c'est un dict, fusionner directement
            elif isinstance(file_data, dict):
                all_deals.update(file_data)
        print(f"   ✓ {filename}: {len(file_data) if isinstance(file_data, (list, dict)) else 0} deals")
    
    return all_deals

# ========================== MAIN ==========================

if __name__ == "__main__":

    print("Indexation des deals dans Elasticsearch")
    
    # Charger les données depuis tous les fichiers
    print(f"\n📂 Chargement des fichiers depuis {DATA_DIR}...")
    data = load_all_deals(DATA_DIR)
    print(f"\n✓ Total: {len(data)} deals chargés")
    
    # Créer l'index
    print("\n📦 Création de l'index...")
    create_index()
    
    # Indexer les documents
    print("\n🚀 Indexation en cours...")
    start_time = time.time()
    indexed, failed = index_deals(data)
    elapsed = time.time() - start_time
    
    # Finaliser
    print("\n🔄 Finalisation de l'index...")
    finalize_index()
    
    # Résumé
    print("📊 RÉSUMÉ")
    print(f"\n✓ Documents indexés: {indexed}")
    print(f"✗ Échecs: {failed}")
    print(f"⏱ Temps total: {elapsed:.1f}s")
    print(f"📈 Vitesse: {indexed/elapsed:.1f} docs/s" if elapsed > 0 else "")
    
    # Vérification
    count = es.count(index=ES_INDEX)
    print(f"\n✓ Total dans l'index: {count['count']} documents")

