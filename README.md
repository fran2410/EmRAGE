
<img src="iconos/icono_recortado.png" alt="Descripción" width="100"/>



[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/) 

# EmRAGE
EmRAGE is a CLI tool designed to index your Thunderbird emails and interact with them using a RAG (Retrieval-Augmented Generation) engine.

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

# Installing through Docker

We provide a Docker image with the scripts already installed. To run through Docker, you may build the Dockerfile provided in the repository by running:

```bash
docker build -t emrage .
```

Then, to run your image just type:

```bash
docker run -it emrage
```

And you will be ready to use the scripts (see section below). If you want to have access to the results we recommend [mounting a volume](https://docs.docker.com/storage/volumes/). We recommend the following command that will mount your email directory as the `out` folder in the Docker image:

```bash
docker run -it -v {YOUR_THUNDERBIRD_EMAIL_DIRECTORY}:/EmRAGE/out emrage
```
To locate your Thunderbird email folder, navigate to: `Account Settings` → `Server Settings` → `Local Directory`.

# Usage

## 1. Initial Configuration
Before running the engine, you need to set up your local paths. The system will also automatically check for and install dependencies like Ollama and the necessary NLP models.

```bash
poetry run emrage config
```
- **Thunderbird Path:** Enter the path to your local mail folder.

  - Tip: In Thunderbird, go to  `Account Settings` → `Server Settings` → `Local Directory` to find it.

- **Database Path:** Choose where you want to store the generated vector databases (defaults to data/db).
## 2. Run the Interactive Session
Once configured and indexed, start the RAG engine to begin chatting with your email data:
```bash
poetry run emrage run
```
## 3. Update Database
If you have received new emails and want to re-index your local database, run:
```bash
poetry run emrage update
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
