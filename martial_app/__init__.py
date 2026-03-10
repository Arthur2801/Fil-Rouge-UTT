"""
Package martial_app - Application Streamlit pour Dealabs RAG.

Ce package contient l'application principale et les modules associés.
"""

__version__ = "1.0.0"
__author__ = "Projet Master Big Data"

# Exports principaux
from . import rag_logic
from . import i18n

__all__ = ['rag_logic', 'i18n']
