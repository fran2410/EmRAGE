
<img src="iconos/icono_recortado.png" alt="Descripción" width="100"/>



[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/) 

# EmRAGE
Desarrollo de un sistema de búsqueda para correo electrónico basado en LLMs

# Instalar desde Github

##  Clonar el repositorio:
   ```bash
   git clone https://github.com/fran2410/EmRAGE.git
   cd EmRAGE
   ```
## 1. Conda

Para instalar Conda en tu sistema visita por favor la documentación oficial de Conda [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).


#### Crear y activar el entorno Conda
```bash
conda create -n EmRAGE   python=3.13 
conda activate EmRAGE   
```

## 2. Poetry

Para instalar Poetry en tu sistema visita por favor la documentación oficial de Poetry [here](https://python-poetry.org/docs/#installation).

#### Instalar las dependencias del proyecto
Ejecuta el siguiente comando en la raíz de el repositorio para instalar las dependencias necesarias.
```bash
poetry install
```

## 3. Modelos
Para instalar los modelos a utilizar se dispone de un script llamado `install.sh` pero se puede hacer de forma manual ejecutando los siguientes comandos:
```
# --- 1. Ollama y Modelo ---
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b

# --- 2. Modelos de spaCy ---
poetry run python -m spacy download en_core_web_md
poetry run python -m spacy download es_core_news_lg
poetry run python -m spacy download fr_core_news_sm
poetry run python -m spacy download de_core_news_sm
poetry run python -m spacy download it_core_news_sm
poetry run python -m spacy download pt_core_news_sm
poetry run python -m spacy download xx_ent_wiki_sm
```

# Uso
## 1. Procesamiento de Correos Electrónicos
El primer paso es procesar los archivos .eml y convertirlos a un formato estructurado:

```bash
python src/email_loader.py
```
Por defecto, busca correos en la carpeta `data/emails` y guarda el resultado en `data/processed/emails_processed.json`.

## 2. Indexación para Búsqueda Semántica
Una vez procesados los correos, se indexan para búsquedas rápidas:

```bash
python src/embeddings_system.py
```

## 3. Sistema de Búsqueda y Respuestas
Para usar el sistema interactivo de búsqueda:

```bash
python src/rag_engine.py
```

# Metadatos

El sistema utiliza un esquema de metadatos uniforme para representar y almacenar la información de cada correo electrónico antes de generar embeddings e indexarlos en la base de datos vectorial.

## Estructura (`Email`)

Cada correo se representa mediante una instancia de la clase `Email` (definida en `email_loader.py`), cuyos campos principales son:

| Campo | Tipo | Descripción |
|-------|------|--------------|
| **id** | `str` | Identificador único de 12 caracteres (hash MD5 truncado). |
| **message_id** | `str` | ID original del mensaje (`Message-ID` del encabezado). | Extraído del campo `Message-ID`. |
| **date** | `str` | Fecha normalizada en formato ISO UTC. | Parseada con `parsedate_to_datetime()` y convertida a UTC. |
| **from_address** | `str` | Dirección de correo del remitente. | Campo `From`, normalizado. |
| **to_addresses** | `list[str]` | Lista de destinatarios principales. | 
| **cc_addresses**, **bcc_addresses** | `list[str]` | Copias visibles/ocultas. |
| **subject** | `str` | Asunto del correo. |
| **body** | `str` | Cuerpo del mensaje en texto plano. |
| **thread_id** | `str` | Identificador del hilo. | 
| **in_reply_to**, **references** | `str` | 
| **x_filename** | `str` | Nombre del archivo `.eml` original. | 
---

## Metadatos de indexación (para embeddings)

Durante la indexación (`embeddings_system.py`), cada correo se divide en *chunks* (fragmentos de texto) y se enriquece con metadatos adicionales para búsqueda semántica.

| Campo | Tipo | Descripción |
|-------|------|--------------|
| **email_id** | `str` | Referencia al `Email.id` original. |
| **message_id**, **from**, **to**, **cc**, **bcc**, **date**, **subject**, **thread_id**, **in_reply_to**, **references**, **x_filename** | Copiados desde el objeto `Email`. |
| **chunk_type** | `"subject_body"` | Indica el tipo de fragmento. |
| **chunk_index** | `int` | Índice del fragmento dentro del correo. |
| **chunk_start**, **chunk_end** | `int` | Posiciones inicial y final del fragmento. |
| **total_chunks** | `int` | Total de fragmentos del correo. |

Los textos se normalizan mediante:
- Eliminación de espacios múltiples y líneas vacías.  
- Sustitución de URLs por `[URL]` y correos por `[EMAIL]`.  
- Limpieza de separadores o símbolos repetidos.  

---

## Metadatos de contactos

Además, el sistema genera una base de datos separada de contactos (`ContactVectorDB`):

| Campo | Tipo | Descripción |
|-------|------|--------------|
| **email_address** | `str` | Dirección de correo normalizada. |
| **display_name** | `str` | Nombre visible o alias. |
| **sent_ids** | `list[str]` | IDs de correos enviados por este contacto. |
| **received_ids** | `list[str]` | IDs de correos recibidos de este contacto. |
| **total_emails** | `int` | Total de mensajes relacionados. |


## License

This project is distributed under the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0). Contributions to the project must follow the same licensing terms.
