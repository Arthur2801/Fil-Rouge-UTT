import streamlit as st
import os
from PIL import Image
from rag_logic import get_deals_rag, get_unique_categories

def main():
    st.set_page_config(page_title="Dealabs Smart Search", page_icon="🔥")

    # Gestion du logo avec correction de dépréciation
    if os.path.exists("notre_logo.png"):
        logo = Image.open("notre_logo.png")
        # On utilise 'use_container_width' car 'use_column_width' est déprécié
        st.sidebar.image(logo, use_container_width=True)

    # Interface Sidebar
    categories = get_unique_categories()
    selected_cat = st.sidebar.selectbox("Choisir un groupe", categories)
    max_p = st.sidebar.slider("Budget maximum (€)", 0, 5000, 1200)

    st.title("Assistant Intelligent Dealabs 🤖")
    query = st.text_input("Que cherchez-vous ?", placeholder="ex: ordinateur pour jouer")

    if query:
        with st.spinner("Recherche sémantique..."):
            results = get_deals_rag(query, selected_cat, max_p)
            
            if results:
                for deal in results:
                    st.subheader(f"{deal['title']} - {deal['price']}€")
                    # Affichage du groupe dynamique
                    st.write(f"Groupe : {deal.get('group_display_summary', 'N/A')}")
                    st.divider()
            else:
                st.warning("Aucun deal ne correspond. Vérifiez l'index vectoriel Atlas.")

if __name__ == "__main__":
    main()