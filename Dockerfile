FROM python:3.13

RUN git clone https://github.com/fran2410/EmRAGE.git

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN pip install poetry

WORKDIR /EmRAGE

RUN poetry install

RUN poetry run python -m spacy download en_core_web_md && \
    poetry run python -m spacy download es_core_news_lg && \
    poetry run python -m spacy download fr_core_news_sm && \
    poetry run python -m spacy download de_core_news_sm && \
    poetry run python -m spacy download it_core_news_sm && \
    poetry run python -m spacy download pt_core_news_sm && \
    poetry run python -m spacy download xx_ent_wiki_sm

RUN curl -fsSL https://ollama.com/install.sh | sh

CMD ["/bin/bash", "-c", "\
    echo 'Iniciando Ollama...' && \
    ollama serve > /dev/null 2>&1 & \
    sleep 3 && \
    echo 'Descargando modelo llama3.2:1b...' && \
    ollama pull llama3.2:1b && \
    echo 'Modelo descargado. Entorno listo.' && \
    echo 'Usage:\n  poetry run python src/email_loader.py\n  poetry run python src/embeddings_system.py\n  poetry run python src/rag_engine.py\n'  && \
    exec bash \
"]
