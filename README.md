
# Titre: Assistant Intelligent Dealabs

# Description du projet
Cette application est un assistant de recherche intelligent utilisant l'architecture RAG (Retrieval-Augmented Generation). Contrairement à une recherche classique par mots-clés, cet outil utilise la recherche sémantique pour comprendre l'intention de l'utilisateur et interroger une base de données vectorielle MongoDB Atlas.


## Déploiement Cloud
L'application est officiellement déployée et accessible pour test via le lien suivant : 👉 https://martial-dealabs-raggit-dq2ot2gjjmgdq83mwhmoj2.streamlit.app/
## Fonctionnalités
- Recherche Sémantique : Capacité à trouver des produits par concept (ex: "ordinateur pour montage vidéo" au lieu de "PC 16Go RAM").

- Filtrage Hybride : Affinage des résultats par budget (curseur de prix) et par catégories dynamiques.

- Interface Intuitive : Développée avec Streamlit pour une expérience utilisateur fluide.

- Accès Direct : Boutons de redirection vers les sites marchands intégrés à chaque article.
## Architecture Technique
- Base de Données : MongoDB Atlas avec Vector Search Index.

- Modèle d'Embedding : sentence-transformers/all-MiniLM-L6-v2 (Hugging Face).

- Backend : Python 3.11+ avec LangChain pour l'orchestration.

- Frontend : Streamlit.

- Industrialisation : Projet prêt pour la conteneurisation via Docker.
## Configuration pour les Développeurs

1. Variables d'environnement
Pour faire tourner le projet localement, créez un fichier .env :

MONGO_URI=mongodb+srv://<votre_user>:<votre_password>@cluster0.ou16sxf.mongodb.net/


Note : Pour la version déployée, ces identifiants sont gérés via les Secrets de Streamlit Cloud.

2. Index de recherche Atlas
L'index sur MongoDB doit être nommé vector_index et configurer le champ embedding avec 384 dimensions.
## Installation Locale

1. Cloner la branche : 
git checkout <ma-branche-de-travail> (Bash)

2. Installer les dépendances :
pip install -r requirements.txt (Bash)

3. Lancer l'application :
streamlit run app.py (Bash)
## Schéma des Métadonnées (Mapping)

- embedding: Vecteurs IA (384 dim)
- group_display_summary: Catégories utilisées pour le filtrage
- price: Prix numérique pour le filtrage par budget
- url: Lien source pour la redirection
- text: Description complète de l'article
## Authors

- Arthur
- Martial
- Yassine

