"""Application Streamlit pour la recherche intelligente de deals Dealabs."""

# Imports de la bibliothèque standard Python
import os  # Pour vérifier l'existence de fichiers

# Imports de bibliothèques tierces
import streamlit as st  # Framework web pour créer l'interface utilisateur
from PIL import Image  # Pour charger et afficher des images

# Imports locaux depuis le module rag_logic
from rag_logic import get_deals_rag, get_unique_categories


def main():
    """Fonction principale de l'application."""
    
    # --- CONFIGURATION DE LA PAGE STREAMLIT ---
    # Configuration initiale de la page web (titre, icône)
    # Doit être appelé en premier avant tout autre élément Streamlit
    st.set_page_config(
        page_title="Dealabs Smart Search",  # Titre dans l'onglet du navigateur
        page_icon="🔥"  # Icône dans l'onglet du navigateur
    )

    # --- GESTION DU LOGO DANS LA SIDEBAR ---
    # Vérification de l'existence du fichier logo
    if os.path.exists("notre_logo.png"):
        # Chargement de l'image avec PIL
        logo = Image.open("notre_logo.png")
        # Affichage dans la barre latérale avec largeur adaptative
        st.sidebar.image(logo, use_container_width=True)

    # --- INTERFACE DE FILTRAGE (SIDEBAR) ---
    # Récupération dynamique des catégories depuis MongoDB
    categories = get_unique_categories()
    
    # Menu déroulant pour sélectionner une catégorie
    # L'utilisateur peut choisir "Toutes" ou une catégorie spécifique
    selected_cat = st.sidebar.selectbox("Choisir un groupe", categories)
    
    # Slider pour définir le budget maximum
    # Valeurs : min=0€, max=10000€, par défaut=1200€
    max_p = st.sidebar.slider("Budget maximum (€)", 0, 10000, 1200)

    # --- ZONE PRINCIPALE : TITRE ET BARRE DE RECHERCHE ---
    # Affichage du titre principal de l'application
    st.title("Assistant Intelligent Dealabs 🤖")
    
    # Champ de saisie pour la requête de recherche de l'utilisateur
    query = st.text_input(
        "Que cherchez-vous ?",  # Label du champ
        placeholder="ex: ordinateur pour jouer"  # Texte d'exemple grisé
    )

    # --- TRAITEMENT DE LA RECHERCHE ---
    # Si l'utilisateur a saisi une requête (champ non vide)
    if query:
        # Affichage d'un spinner pendant le traitement
        with st.spinner("Recherche sémantique..."):
            # Appel de la fonction RAG avec les filtres sélectionnés
            results = get_deals_rag(query, selected_cat, max_p)

            # --- AFFICHAGE DES RÉSULTATS ---
            if results:
                # Boucle sur chaque deal retourné
                for deal in results:
                    # --- EN-TÊTE DU DEAL ---
                    # Extraction des données du deal
                    title = deal['title']  # Titre du deal
                    price = deal['price']  # Prix en euros
                    
                    # Affichage du titre en orange avec le prix en gras
                    st.markdown(f"### :orange[{title}] — **{price}€**")

                    # --- MÉTADONNÉES EN COLONNES ---
                    # Création de 2 colonnes (ratio 2:1)
                    col1, col2 = st.columns([2, 1])
                    
                    # Colonne gauche : Catégorie du deal
                    with col1:
                        category = deal.get('group_display_summary', 'N/A')
                        st.caption(f" Catégorie : {category}")
                    
                    # Colonne droite : Score de pertinence
                    with col2:
                        # Conversion du score (0-1) en pourcentage
                        score_pct = round(deal['score'] * 100, 1)
                        st.caption(f" Pertinence : {score_pct}%")

                    # --- DESCRIPTION DÉTAILLÉE (EXPANDABLE) ---
                    # Si le deal contient une description textuelle
                    if "text" in deal:
                        # Titre du menu déroulant (sur 2 lignes pour PEP8)
                        expander_title = " Voir les détails et la " \
                                        "description complète"
                        
                        # Menu déroulant (replié par défaut)
                        with st.expander(expander_title):
                            # Nettoyage du texte : ajout de sauts de ligne
                            # après chaque phrase pour améliorer la lisibilité
                            clean_text = deal["text"].replace(". ", ".\n\n")
                            st.write(clean_text)

                    # --- BOUTON D'ACTION ---
                    # Si le deal possède une URL valide
                    if "url" in deal and deal["url"]:
                        # Bouton cliquable qui redirige vers le site du deal
                        st.link_button(
                            "🚀 PROFITER DE L'OFFRE SUR LE SITE",
                            deal["url"],  # URL de destination
                            use_container_width=True  # Bouton pleine largeur
                        )
                    else:
                        # Si pas d'URL, affichage d'un message informatif
                        st.info("ℹ️ Lien indisponible")

                    # Séparateur horizontal entre les deals
                    st.divider()
            else:
                # --- AUCUN RÉSULTAT TROUVÉ ---
                # Message d'avertissement si la recherche ne retourne rien
                st.warning(
                    "Aucun deal ne correspond. "
                    "Vérifiez l'index vectoriel Atlas."
                )


# --- POINT D'ENTRÉE DE L'APPLICATION ---
# Exécution de la fonction main() si le script est lancé directement
# (pas importé comme module)
if __name__ == "__main__":
    main()