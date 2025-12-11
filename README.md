
<img src="iconos/icono_recortado.png" alt="Descripción" width="100"/>



[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/) 

# EmRAGE
Development of an email search system based on LLMs.

# Install from Github

## Clone the repository
   ```bash
   git clone https://github.com/fran2410/EmRAGE.git
   cd EmRAGE
   ```
## 1. Conda

For installing Conda on your system, please visit the official Conda documentation [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).


#### Create and activate the Conda environment
```bash
conda create -n EmRAGE   python=3.13 
conda activate EmRAGE   
```

## 2. Poetry

For installing Poetry on your system, please visit the official Poetry documentation [here](https://python-poetry.org/docs/#installation).

#### Install project dependencies
Run the following command in the root of the repository to install dependencies:
```bash
poetry install
```

## 3. Models
To install the models you can use the script `install.sh` or run the commands manually:
```
# --- 1. Ollama y llama3.2 ---
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b

# --- 2. spaCy models ---
poetry run python -m spacy download en_core_web_md
poetry run python -m spacy download es_core_news_lg
poetry run python -m spacy download fr_core_news_sm
poetry run python -m spacy download de_core_news_sm
poetry run python -m spacy download it_core_news_sm
poetry run python -m spacy download pt_core_news_sm
poetry run python -m spacy download xx_ent_wiki_sm
```
# Installing through Docker

We provide a Docker image with the scripts already installed. To run through Docker, you may build the Dockerfile provided in the repository by running:

```bash
docker build -t emrage .
```

Then, to run your image just type:

```bash
docker run --rm -it --add-host=host.docker.internal:host-gateway emrage
```

And you will be ready to use the scripts (see section below). If you want to have access to the results we recommend [mounting a volume](https://docs.docker.com/storage/volumes/). For example, the following command will mount the current directory as the `out` folder in the Docker image:

```bash
docker run -it --rm --add-host=host.docker.internal:host-gateway -v $PWD/out:/EmRAGE/out emrage 
```
If you move any files produced by the scripts or set the output folder to `/out`, you will be able to see them in your current directory in the `/out` folder.

# Usage
## 1. Email Processing
Processes .eml files and converts them into structured format.

```bash
python src/email_loader.py
```
By default it reads from `data/emails` and writes to `data/processed/emails_processed.json`.
#### Command Line Options

```
--emails                 # Emails folder
--output                 # Output JSON path
```

## 2. Semantic Indexing
Once the emails are processed, they are indexed for quick searches.
```bash
python src/embeddings_system.py
```
#### Command Line Options
```
# Model testing options
--test-models             # Test multiple embedding models
--test-file               # Path to test file (default: data/evaluate/test_preguntas_146.txt)
--topk                    # Top K results to consider (default: 3)
--n-results               # Number of results per query (default: 10)
--device                  # CPU or CUDA device (default: cpu)

# Database paths
--db-path                 # Custom database path
--json-path               # Custom JSON data path
```
## 3. Search and Response System
To use the interactive search system:
```bash
python src/rag_engine.py
```
#### Command Line Options
```
# Database paths
--db-path                 # Custom database path
--contact-db-path         # Custom contacts database path
```
# Metadata

The system uses a uniform metadata schema to represent and store the information of each email before generating embeddings and indexing them in the vector database.

## Structure (`Email`)

Each email is represented by an instance of the `Email` class (defined in `email_loader.py`), whose main fields are:"

| Field | Type | Description |
|-------|------|--------------|
| **id** | `str` | Unique 12 character ID from truncated MD5 hash. |
| **message_id** | `str` | Original Message-ID (`Message-ID` del encabezado). |
| **date** | `str` | ISO UTC normalized date. |
| **from_address** | `str` | Sender address. |
| **to_addresses** | `list[str]` | Primary recipients. | 
| **cc_addresses**, **bcc_addresses** | `list[str]` | CC and BCC. |
| **subject** | `str` | Email subject. |
| **body** | `str` | Plain text body. |
| **thread_id** | `str` | Thread identifier. | 
| **in_reply_to**, **references** | `str` | Reply metadata | 
| **x_filename** | `str` | Original `.eml` filename. | 
---

## Indexing Metadata

During indexing (`embeddings_system.py`), each email is split into *chunks* (text fragments) and enriched with additional metadata for semantic search.

| Field | Type | Description |
|-------|------|--------------|
| **email_id** | `str` | Reference to `Email.id`. |
| **message_id**, **from**, **to**, **cc**, **bcc**, **date**, **subject**, **thread_id**, **in_reply_to**, **references**, **x_filename** | Copied metadata from `Email`. |
| **chunk_type** | `"subject_body"` | Fragment type. |
| **chunk_index** | `int` | Fragment index. |
| **chunk_start**, **chunk_end** | `int` | Text positions. |
| **total_chunks** | `int` | Total fragments. |

Texts are normalized by:
- Eliminating multiple spaces and empty lines.
- Replacing URLs with `[URL]` and emails with `[EMAIL]`.
- Cleaning repeated separators or symbols.
---

## Contact Metadata

Additionally, the system generates a separate contact database (`ContactVectorDB`):

| Field | Type | Description |
|-------|------|--------------|
| **email_address** | `str` | Normalized address. |
| **display_name** | `str` | Visible name. |
| **sent_ids** | `list[str]` | Emails sent. |
| **received_ids** | `list[str]` | Emails received. |
| **total_emails** | `int` | Total related messages. |


## License

This project is distributed under the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0). Contributions to the project must follow the same licensing terms.
