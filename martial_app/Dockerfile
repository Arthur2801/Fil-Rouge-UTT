# 1. Utiliser une version légère de Python
FROM python:3.11-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3. Installer les dépendances système nécessaires (pour Pillow et les outils de build)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 4. Copier le fichier des dépendances
COPY requirements.txt .

# 5. Installer les bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copier tout le reste du code (app.py, rag_logic.py, logo, .env)
COPY . .

# 7. Exposer le port par défaut de Streamlit
EXPOSE 8501

# 8. Commande pour lancer l'application au démarrage du conteneur
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]