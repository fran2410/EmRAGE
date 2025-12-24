FROM python:3.13

RUN git clone https://github.com/fran2410/EmRAGE.git

RUN pip install poetry

WORKDIR /EmRAGE

RUN poetry install
ENV PATH="/EmRAGE/.venv/bin:$PATH"

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN ollama serve & sleep 5 && ollama pull llama3.2:1b

RUN echo '#!/bin/bash\n\
ollama serve > /dev/null 2>&1 &\n\
sleep 5\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash", "-c", "echo 'Usage:\n  poetry run emrage config\n  poetry run emrage run\n  poetry run emrage update\n'  && \
    exec bash \
"]
