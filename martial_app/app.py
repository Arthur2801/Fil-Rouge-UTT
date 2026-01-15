"""
Interface utilisateur Streamlit pour l'application de recherche de deals.
"""

import streamlit as st
# Importation de vos fonctions depuis rag_logic.py
from rag_logic import search_deals, format_context


def main():
    """
    Fonction principale pour lancer l'application Streamlit.
    """
    st.set_page_config(page_title="Dealabs Smart Search", page_icon="🚀")
    
    st.title("Assistant Intelligent Dealabs 🚀")
    st.write("Trouvez les meilleurs bons plans grâce à l'IA générative.")

    # Zone de saisie utilisateur
    user_input = st.text_input(
        "Que cherchez-vous ?", 
        placeholder="Ex: Un smartphone pas cher ou un PC gamer..."
    )

    if user_input:
        with st.spinner("Recherche des meilleurs deals en cours..."):
            try:
                # 1. Appel à la logique de recherche (Retrieval)
                results = search_deals(user_input)
                
                # 2. Mise en forme des résultats pour l'affichage
                if results:
                    st.subheader("Les meilleurs plans trouvés :")
                    for hit in results:
                        deal = hit['_source']
                        # Affichage simple de chaque deal trouvé
                        with st.expander(f"{deal.get('title')} - {deal.get('price')}€"):
                            st.write(f"Lien: {deal.get('url')}")
                            # Ici Arthur pourra intégrer sa température prédite
                            st.info(f"Pertinence (Score): {hit['_score']}")
                    
                    # 3. Préparation pour le LLM (Génération)
                    # context = format_context(results)
                    # st.write("Réponse de l'IA (en cours de développement...)")
                    
                else:
                    st.warning("Aucun deal ne correspond à votre recherche.")
                    
            except Exception as e:
                st.error(f"Erreur de connexion à la base de données : {e}")


if __name__ == "__main__":
    main()