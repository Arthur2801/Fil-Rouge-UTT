"""
Interface Streamlit pour la recherche intelligente de produits Dealabs.
"""

import streamlit as st
from PIL import Image
import os
from rag_logic import get_deals_rag


def load_logo():
    """Charge le logo du projet 'notre_logo' depuis le dossier local."""
    # Liste des extensions possibles pour votre fichier
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        logo_path = f"notre_logo{ext}"
        if os.path.exists(logo_path):
            return Image.open(logo_path)
    return None


def main():
    """Fonction principale de l'application web."""
    st.set_page_config(
        page_title="Dealabs Smart Search", 
        page_icon="🔥", 
        layout="wide"
    )

    # Sidebar : Logo et Filtres
    logo_img = load_logo()
    if logo_img:
        st.sidebar.image(logo_img, use_container_width=True)
    else:
        st.sidebar.title("Projet IA Gen")

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Critères de recherche")
    
    cat_list = ["Toutes", "Informatique", "Smartphone", "Audio", "Jeux Vidéo"]
    selected_cat = st.sidebar.selectbox("Choisir une catégorie", cat_list)
    max_p = st.sidebar.slider("Budget maximum (€)", 0, 2000, 500)

    # Zone principale
    st.title("Assistant Intelligent Dealabs 🤖")
    st.write("Trouvez les meilleures offres parmi nos 15 000 deals indexés.")

    user_query = st.text_input(
        "Que cherchez-vous ?", 
        placeholder="Ex: Un écran PC pour faire du montage vidéo..."
    )

    if user_query:
        with st.spinner("Recherche sémantique en cours..."):
            results = get_deals_rag(
                user_query, 
                category_filter=selected_cat, 
                max_price=max_p
            )

            if results:
                st.success(f"{len(results)} deals pertinents trouvés !")
                for deal in results:
                    with st.container():
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.subheader(deal['title'])
                            st.info(f"Catégorie : {deal['category']}")
                        with c2:
                            st.write(f"### {deal['price']} €")
                            st.link_button("Voir l'offre", deal['url'])
                        st.divider()
            else:
                st.warning("Aucun deal ne correspond à votre recherche actuelle.")


if __name__ == "__main__":
    main()