#!/bin/bash

# --- 1. Instalación de Ollama y Modelo ---
echo "Instalando Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "Instalando modelo llama3.2:1b..."
ollama pull llama3.2:1b

# --- 2. Modelos de spaCy ---
echo "Descargando modelos de spaCy..."
poetry run python -m spacy download en_core_web_md
poetry run python -m spacy download es_core_news_lg
poetry run python -m spacy download fr_core_news_sm
poetry run python -m spacy download de_core_news_sm
poetry run python -m spacy download it_core_news_sm
poetry run python -m spacy download pt_core_news_sm
poetry run python -m spacy download xx_ent_wiki_sm
echo "Configuración de entorno completada."