FROM python:3.13

RUN git clone https://github.com/fran2410/EmRAGE.git

# RUN curl -sSL https://install.python-poetry.org | python3 -

RUN pip install poetry

WORKDIR /EmRAGE

RUN poetry install
ENV PATH="/EmRAGE/.venv/bin:$PATH"
# RUN poetry run python -m spacy download en_core_web_md && \
#     poetry run python -m spacy download es_core_news_lg && \
#     poetry run python -m spacy download fr_core_news_sm && \
#     poetry run python -m spacy download de_core_news_sm && \
#     poetry run python -m spacy download it_core_news_sm && \
#     poetry run python -m spacy download pt_core_news_sm && \
#     poetry run python -m spacy download xx_ent_wiki_sm

# RUN curl -fsSL https://ollama.com/install.sh | sh


# Instalamos Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Iniciamos Ollama momentáneamente para descargar el modelo y que quede guardado en la imagen
RUN ollama serve & sleep 5 && ollama pull llama3.2:1b

# Creamos un script de entrada para asegurar que Ollama siempre arranque con el contenedor
RUN echo '#!/bin/bash\n\
ollama serve > /dev/null 2>&1 &\n\
sleep 5\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]




CMD ["/bin/bash", "-c", "echo 'Usage:\n  poetry run python src/email_loader.py\n  poetry run python src/embeddings_system.py\n  poetry run python src/rag_engine.py\n'  && \
    exec bash \
"]
