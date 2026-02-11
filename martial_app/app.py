"""
Application Streamlit pour la recherche intelligente de deals Dealabs.

Ce module implémente une interface utilisateur web permettant de rechercher
des deals sur Dealabs en utilisant la recherche vectorielle (RAG) et un LLM.

Architecture:
    - Interface Streamlit pour l'interaction utilisateur
    - Recherche vectorielle MongoDB Atlas pour trouver des deals similaires
    - LLM (via GROQ) pour analyser et recommander les meilleurs deals

Auteur: Projet Master Big Data
Date: Janvier 2026
"""

# Imports de la bibliothèque standard Python
import os

# Imports de bibliothèques tierces
import streamlit as st
from PIL import Image

# Imports locaux depuis le module rag_logic
from rag_logic import (
    get_deals_rag,
    get_unique_categories,
    get_llm_answer
)


def main():
    """
    Fonction principale de l'application Streamlit.
    
    Cette fonction orchestre toute la logique de l'application :
    1. Configuration de la page et affichage du logo
    2. Gestion des filtres (catégorie, prix)
    3. Gestion du formulaire de recherche
    4. Traitement des résultats et affichage
    
    Returns:
        None
    """
    # --- CONFIGURATION DE LA PAGE STREAMLIT ---
    # Configuration initiale de la page web (titre, icône)
    st.set_page_config(
        page_title="Dealabs Smart Search",  # Titre dans l'onglet du navigateur
        page_icon="🔥"  # Icône dans l'onglet du navigateur
    )

    # --- GESTION DU LOGO DANS LA SIDEBAR ---
    # Récupération du chemin absolu du répertoire courant
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construction du chemin complet vers le fichier logo
    logo_path = os.path.join(current_dir, "notre_logo.png")

    # Vérification de l'existence du logo et affichage
    if os.path.exists(logo_path):
        # Chargement de l'image avec PIL (Python Imaging Library)
        logo = Image.open(logo_path)
        
        # Affichage dans la sidebar avec largeur adaptative
        st.sidebar.image(logo, width="stretch")
    else:
        # Message d'erreur si le logo n'est pas trouvé
        st.sidebar.error("Fichier logo introuvable sur le serveur.")

    # --- INTERFACE DE FILTRAGE (SIDEBAR) ---
    # Récupération de toutes les catégories disponibles depuis MongoDB
    categories = get_unique_categories()
    
    # Widget de sélection déroulante pour choisir une catégorie
    # Permet de filtrer les deals par type (High-Tech, Jeux, etc.)
    selected_cat = st.sidebar.selectbox("Choisir un groupe", categories)
    
    # Widget slider pour définir le budget maximum
    # min=0, max=10000, valeur par défaut=1200
    # Widget slider pour définir le budget maximum
    # min=0, max=10000, valeur par défaut=1200
    max_p = st.sidebar.slider("Budget maximum (€)", 0, 10000, 1200)

    # --- ZONE PRINCIPALE : TITRE ET BARRE DE RECHERCHE ---
    # Affichage du titre principal de l'application
    st.title("Assistant Intelligent Dealabs 🤖")
    
    # --- GESTION DE L'ÉTAT DU FORMULAIRE ---
    # Initialisation de session_state pour persister les données entre reloads
    # session_state permet de conserver l'état de l'application
    if 'query_submitted' not in st.session_state:
        st.session_state.query_submitted = False
    
    # --- FORMULAIRE DE RECHERCHE ---
    # Utilisation d'un formulaire Streamlit avec clear_on_submit=True
    # Cela permet de vider automatiquement le champ après soumission
    with st.form(key="search_form", clear_on_submit=True):
        # Champ de saisie texte pour la requête utilisateur
        query = st.text_input(
            "Que cherchez-vous ?",  # Label du champ
            placeholder="ex: ordinateur pour jouer",  # Texte d'exemple
            key="search_input"  # Clé unique pour ce widget
        )
        
        # Bouton de soumission du formulaire avec icône de recherche
        submit_button = st.form_submit_button("🔍 Rechercher")
    
    # --- TRAITEMENT DE LA SOUMISSION ---
    # Vérification que le formulaire a été soumis ET qu'une requête existe
    if submit_button and query:
        # Sauvegarde de la requête dans session_state pour la persister
        st.session_state.last_query = query
        
        # Activation du flag indiquant qu'une recherche a été effectuée
        st.session_state.query_submitted = True
    
    # --- RÉCUPÉRATION DE LA DERNIÈRE REQUÊTE ---
    # Si une recherche a été effectuée, récupérer la dernière requête
    # Cela permet de maintenir les résultats même après rechargement
    if (st.session_state.get('query_submitted', False) and
            st.session_state.get('last_query')):
        query = st.session_state.last_query

    # --- TRAITEMENT DE LA RECHERCHE ---
    # Vérification qu'une requête existe avant de lancer la recherche
    if query:
        # Affichage d'un spinner pendant la recherche vectorielle
        with st.spinner("Recherche sémantique..."):
            # Appel de la fonction RAG pour rechercher les deals
            # Étapes 4 à 6 du processus RAG :
            # - Vectorisation de la requête
            # - Recherche dans MongoDB Atlas
            # - Récupération des deals similaires
            results = get_deals_rag(query, selected_cat, max_p)

            # --- VÉRIFICATION DES RÉSULTATS ---
            if results:
                # --- SYSTÈME DE FILTRAGE PAR PERTINENCE ---
                # Définition du seuil de pertinence (65%)
                # Les deals au-dessus de ce seuil sont considérés pertinents
                # Les deals en-dessous sont des suggestions alternatives
                PERTINENCE_THRESHOLD = 0.65
                
                # Séparation des deals en deux catégories selon leur score
                # Liste comprehension pour filtrer les deals pertinents
                relevant_deals = [
                    deal for deal in results
                    if deal.get('score', 0) >= PERTINENCE_THRESHOLD
                ]
                
                # Liste comprehension pour filtrer les deals similaires
                similar_deals = [
                    deal for deal in results
                    if deal.get('score', 0) < PERTINENCE_THRESHOLD
                ]
                
                # Flag booléen : True si au moins un deal pertinent existe
                # Flag booléen : True si au moins un deal pertinent existe
                has_relevant = len(relevant_deals) > 0
                
                # --- SECTION ANALYSE CHATBOT (LLM) ---
                # Affichage conditionnel selon la pertinence des résultats
                if has_relevant:
                    # Cas 1 : Des deals pertinents ont été trouvés
                    st.subheader("🤖 Analyse de l'Assistant")
                    
                    # Message informatif avec le nombre de deals pertinents
                    st.info(
                        f"**{len(relevant_deals)} deal(s) pertinent(s) "
                        f"trouvé(s)** pour votre recherche"
                    )
                else:
                    # Cas 2 : Aucun deal pertinent, seulement des suggestions
                    st.subheader("💡 Suggestions Alternatives")
                    
                    # Message d'avertissement expliquant la situation
                    st.warning(
                        "Aucun deal exact trouvé. Voici des suggestions "
                        "similaires qui pourraient vous intéresser :"
                    )
                
                # --- GÉNÉRATION DE LA RÉPONSE LLM ---
                # Spinner pendant l'analyse par le modèle de langage
                with st.spinner("Analyse en cours..."):
                    try:
                        # Sélection des deals à analyser par le LLM
                        # Si pertinents trouvés : uniquement ceux-ci
                        # Sinon : tous les résultats (suggestions)
                        deals_to_analyze = (
                            relevant_deals if has_relevant else results
                        )
                        
                        # Étapes 7 à 9 du processus RAG :
                        # - Construction du prompt avec contexte
                        # - Envoi au LLM (GROQ)
                        # - Récupération et affichage de la réponse
                        answer = get_llm_answer(query, deals_to_analyze)
                        st.write(answer)
                        
                    except Exception as e:
                        # Gestion des erreurs lors de l'appel au LLM
                        st.error(
                            f"Erreur lors de la génération de la "
                            f"réponse : {e}"
                        )
                
                # Ligne de séparation visuelle entre sections
                # Ligne de séparation visuelle entre sections
                st.divider()

                # --- AFFICHAGE DÉTAILLÉ DES RÉSULTATS ---
                # Sélection des deals à afficher (pertinents ou tous)
                deals_to_display = relevant_deals if has_relevant else results
                
                # En-tête conditionnel selon le type de deals affichés
                if has_relevant:
                    # Affichage pour les deals pertinents
                    st.subheader(
                        f"📌 {len(relevant_deals)} Deal(s) Pertinent(s)"
                    )
                else:
                    # Affichage pour les suggestions alternatives
                    st.subheader(
                        f"🔍 {len(results)} Suggestion(s) Similaire(s)"
                    )
                
                # --- BOUCLE D'AFFICHAGE DES DEALS ---
                # Itération sur chaque deal à afficher
                for deal in deals_to_display:
                    # --- EXTRACTION DES DONNÉES ---
                    # Récupération sécurisée avec valeurs par défaut
                    title = deal.get('title', 'Sans titre')
                    price = deal.get('price', 0)
                    
                    # --- AFFICHAGE DU TITRE ET DU PRIX ---
                    # Utilisation de Markdown pour le formatage
                    # :orange[] colore le texte en orange
                    st.markdown(f"### :orange[{title}] — **{price}€**")

                    # --- MÉTADONNÉES EN COLONNES ---
                    # Création de 2 colonnes (ratio 2:1)
                    col1, col2 = st.columns([2, 1])
                    
                    # Colonne 1 : Catégorie du deal
                    with col1:
                        category = deal.get('main_group_name', 'N/A')
                        st.caption(f" Catégorie : {category}")
                    
                    # Colonne 2 : Score de pertinence en pourcentage
                    with col2:
                        # Conversion du score (0-1) en pourcentage
                        # arrondi à 1 décimale
                        score_pct = round(deal.get('score', 0) * 100, 1)
                        st.caption(f" Pertinence : {score_pct}%")

                    # --- DESCRIPTION DÉTAILLÉE (EXPANDER) ---
                    # Vérification de la présence du champ 'text'
                    if "text" in deal:
                        # Widget expander pour afficher/masquer les détails
                        with st.expander(
                            "Voir les détails et la description complète"
                        ):
                            # Nettoyage du texte : ajout de sauts de ligne
                            # après chaque point pour améliorer la lisibilité
                            clean_text = deal["text"].replace(". ", ".\n\n")
                            st.write(clean_text)

                    # --- BOUTON D'ACTION ---
                    # Vérification de la présence et validité de l'URL
                    if "url" in deal and deal["url"]:
                        # Bouton lien vers le deal sur Dealabs
                        st.link_button(
                            "🚀 PROFITER DE L'OFFRE SUR LE SITE",
                            deal["url"],
                            use_container_width=True  # Largeur complète
                        )
                    else:
                        # Message informatif si pas de lien disponible
                        st.info("ℹ️ Lien indisponible")

                    # Séparateur entre chaque deal
                    st.divider()
                    
            else:
                # --- AUCUN RÉSULTAT TROUVÉ ---
                # Message d'avertissement en cas de recherche infructueuse
                st.warning(
                    "Aucun deal ne correspond. "
                    "Vérifiez l'index vectoriel Atlas."
                )


# --- POINT D'ENTRÉE DU PROGRAMME ---
# Vérification que le script est exécuté directement
# (pas importé comme module)
if __name__ == "__main__":
    main()  # Appel de la fonction principale
