"""
Module d'internationalisation (i18n) pour Dealabs Smart Search.
Contient les traductions pour français, anglais et espagnol.
"""

# Dictionnaire de traductions
LANG_NAMES = {"fr": "Français", "en": "English", "es": "Español"}

# Traductions complètes pour l'interface
TRANSLATIONS = {
    "fr": {
        # Titres et sections
        "assistant_analysis": "🤖 Analyse de l'Assistant",
        "alternative_suggestions": "💡 Suggestions Alternatives",
        "predictive_analysis": "🤖 Analyse Prédictive IA",
        "relevant_deals": "Deal(s) Pertinent(s)",
        "similar_suggestions": "🔍 Suggestion(s) Similaire(s)",
        # Messages informatifs
        "deals_found": "deal(s) pertinent(s) trouvé(s) pour votre recherche",
        "no_exact_deal": "Aucun deal exact trouvé. Voici des suggestions similaires qui pourraient vous intéresser :",
        "analyzing": "Analyse en cours...",
        "error_llm": "Erreur lors de la génération de la réponse :",
        "no_new_deals": "🔮 Aucun nouveau deal prometteur détecté pour cette recherche. Essayez de désactiver le Mode Anticipation pour voir tous les résultats.",
        "no_results": "Aucun deal ne correspond. Vérifiez l'index vectoriel Atlas.",
        # Métadonnées des deals
        "temp": "Temp",
        "status": "Statut",
        "new": "Nouveau",
        "old": "Ancien",
        "relevance": "Pertinence",
        "no_reviews": "Pas d'avis",
        "prediction_ml": "Prédiction ML",
        "reliability": "Fiabilité",
        # Prédictions
        "hot": "CHAUD",
        "cold": "FROID",
        "potential": "Potentiel",
        "deal_number": "Deal #",
        # Actions
        "view_details": "Voir les détails et la description complète",
        "get_deal": "🚀 PROFITER DE L'OFFRE SUR LE SITE",
        "link_unavailable": "ℹ️ Lien indisponible",
    },
    "en": {
        # Titles and sections
        "assistant_analysis": "🤖 Assistant Analysis",
        "alternative_suggestions": "💡 Alternative Suggestions",
        "predictive_analysis": "🤖 AI Predictive Analysis",
        "relevant_deals": "Relevant Deal(s)",
        "similar_suggestions": "🔍 Similar Suggestion(s)",
        # Informative messages
        "deals_found": "relevant deal(s) found for your search",
        "no_exact_deal": "No exact deal found. Here are similar suggestions that might interest you:",
        "analyzing": "Analyzing...",
        "error_llm": "Error generating response:",
        "no_new_deals": "🔮 No new promising deals detected for this search. Try disabling Anticipation Mode to see all results.",
        "no_results": "No matching deals. Check Atlas vector index.",
        # Deal metadata
        "temp": "Temp",
        "status": "Status",
        "new": "New",
        "old": "Old",
        "relevance": "Relevance",
        "no_reviews": "No reviews",
        "prediction_ml": "ML Prediction",
        "reliability": "Reliability",
        # Predictions
        "hot": "HOT",
        "cold": "COLD",
        "potential": "Potential",
        "deal_number": "Deal #",
        # Actions
        "view_details": "View details and full description",
        "get_deal": "🚀 GET THE DEAL ON THE SITE",
        "link_unavailable": "ℹ️ Link unavailable",
    },
    "es": {
        # Títulos y secciones
        "assistant_analysis": "🤖 Análisis del Asistente",
        "alternative_suggestions": "💡 Sugerencias Alternativas",
        "predictive_analysis": "🤖 Análisis Predictivo IA",
        "relevant_deals": "Oferta(s) Relevante(s)",
        "similar_suggestions": "🔍 Sugerencia(s) Similar(es)",
        # Mensajes informativos
        "deals_found": "oferta(s) relevante(s) encontrada(s) para tu búsqueda",
        "no_exact_deal": "No se encontró ninguna oferta exacta. Aquí hay sugerencias similares que podrían interesarte:",
        "analyzing": "Analizando...",
        "error_llm": "Error al generar la respuesta:",
        "no_new_deals": "🔮 No se detectaron nuevas ofertas prometedoras para esta búsqueda. Intenta desactivar el Modo Anticipación para ver todos los resultados.",
        "no_results": "Ninguna oferta coincide. Verifica el índice vectorial Atlas.",
        # Metadatos de ofertas
        "temp": "Temp",
        "status": "Estado",
        "new": "Nuevo",
        "old": "Antiguo",
        "relevance": "Relevancia",
        "no_reviews": "Sin opiniones",
        "prediction_ml": "Predicción ML",
        "reliability": "Fiabilidad",
        # Predicciones
        "hot": "CALIENTE",
        "cold": "FRÍO",
        "potential": "Potencial",
        "deal_number": "Oferta #",
        # Acciones
        "view_details": "Ver detalles y descripción completa",
        "get_deal": "🚀 APROVECHAR LA OFERTA EN EL SITIO",
        "link_unavailable": "ℹ️ Enlace no disponible",
    },
}


def get_language_name(lang_code):
    """Retourne le nom de la langue."""
    return LANG_NAMES.get(lang_code, "Français")


def get_translation(lang_code, key):
    """
    Retourne la traduction pour une clé donnée.

    Args:
        lang_code (str): Code de langue (fr, en, es)
        key (str): Clé de traduction

    Returns:
        str: Texte traduit ou clé si traduction non trouvée
    """
    return TRANSLATIONS.get(lang_code, TRANSLATIONS["fr"]).get(key, key)


def get_all_translations(lang_code):
    """
    Retourne toutes les traductions pour une langue.

    Args:
        lang_code (str): Code de langue (fr, en, es)

    Returns:
        dict: Dictionnaire de traductions
    """
    return TRANSLATIONS.get(lang_code, TRANSLATIONS["fr"])
