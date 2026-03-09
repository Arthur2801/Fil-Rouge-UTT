"""
Module d'internationalisation (i18n) pour Dealabs Smart Search.
Contient les traductions pour français, anglais et espagnol.
"""

# Dictionnaire de traductions
LANG_NAMES = {
    "fr": "Français",
    "en": "English", 
    "es": "Español"
}


def get_language_name(lang_code):
    """Retourne le nom de la langue."""
    return LANG_NAMES.get(lang_code, "Français")
