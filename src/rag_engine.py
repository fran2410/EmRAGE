import argparse
import json
import os
import click
import requests
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from functools import lru_cache
import spacy
from spacy.language import Language
import subprocess
from src.embeddings_system import EmailVectorDB, ContactVectorDB,MultilingualEmbedder
        

@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float
    model_used: str

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "timestamp": datetime.now().isoformat(),
        }


class MultilingualContactExtractor:

    MODELS = {
        "en": ["en_core_web_md"],
        "es": ["es_core_news_lg"],
        "fr": ["fr_core_news_sm"],
        "de": ["de_core_news_sm"],
        "it": ["it_core_news_sm"],
        "pt": ["pt_core_news_sm"],
        "xx": ["xx_ent_wiki_sm"],
    }

    LANGUAGE_INDICATORS = {
        # Español
        'es': {
            'buscar', 'correo', 'correos', 'email', 'emails', 'mensaje', 'mensajes',
            'enviados', 'recibidos', 'de', 'para', 'con', 'sin', 'asunto', 'adjunto',
            'por', 'fecha', 'día', 'semana', 'mes', 'año', 'antes', 'después', 'entre',
            'quién', 'cuándo', 'dónde', 'mostrar', 'filtrar', 'listar', 'ver', 'tengo',
            'quiero', 'dame', 'muéstrame', 'enséñame', 'últimos', 'anteriores', 'reenviados',
            'pendientes', 'importantes', 'reunión', 'proyecto', 'cliente', 'informe', 'recordatorio',
            'contiene', 'palabra', 'texto', 'incluye', 'temas', 'asuntos', 'respuestas',
            'no', 'todos', 'solo', 'entre', 'hasta', 'desde', 'ayer', 'hoy', 'mañana',
            'enviado', 'recibido', 'filtra', 'busca', 'muestra', 'hazme', 'encuéntrame', 'en', 'que', 
        },
        # Inglés
        'en': {
            'find', 'search', 'show', 'list', 'get', 'display', 'emails', 'email', 'messages',
            'sent', 'received', 'from', 'to', 'with', 'without', 'subject', 'attachment',
            'by', 'date', 'day', 'week', 'month', 'year', 'before', 'after', 'between',
            'who', 'when', 'where', 'filter', 'only', 'all', 'latest', 'previous', 'forwarded',
            'pending', 'important', 'meeting', 'project', 'client', 'report', 'reminder',
            'contains', 'including', 'text', 'words', 'topic', 'reply', 'responses',
            'yesterday', 'today', 'tomorrow', 'sent', 'received', 'emails', 'messages',
            'searching', 'query', 'show me', 'give me', 'find me', 'display', 'get me'
        },
        # Francés
        'fr': {
            'chercher', 'trouver', 'afficher', 'montrer', 'lister', 'obtenir', 'emails', 'courriel',
            'message', 'messages', 'envoyé', 'reçu', 'de', 'à', 'avec', 'sans', 'objet', 'pièce',
            'jointe', 'par', 'date', 'jour', 'semaine', 'mois', 'année', 'avant', 'après', 'entre',
            'qui', 'quand', 'où', 'filtrer', 'seulement', 'tous', 'dernier', 'précédent', 'transféré',
            'important', 'réunion', 'projet', 'client', 'rapport', 'rappel', 'contenant', 'texte',
            'mots', 'réponse', 'réponses', 'aujourd\'hui', 'hier', 'demain', 'montre', 'trouve',
            'donne-moi', 'affiche-moi', 'liste-moi', 'recherche', 'courriels'
        },
        # Alemán
        'de': {
            'suche', 'finden', 'zeige', 'liste', 'erhalte', 'emails', 'nachrichten',
            'gesendet', 'empfangen', 'von', 'an', 'mit', 'ohne', 'betreff', 'anhang',
            'nach', 'datum', 'tag', 'woche', 'monat', 'jahr', 'vor', 'nach', 'zwischen',
            'wer', 'wann', 'wo', 'filtern', 'nur', 'alle', 'letzte', 'vorherige', 'weitergeleitet',
            'wichtig', 'besprechung', 'projekt', 'kunde', 'bericht', 'erinnerung', 'enthält',
            'text', 'wörter', 'antwort', 'heute', 'gestern', 'morgen', 'zeige mir', 'suche nach'
        },
        # Italiano
        'it': {
            'cerca', 'trova', 'mostra', 'elenca', 'ottieni', 'email', 'messaggi', 'inviati', 'ricevuti',
            'da', 'a', 'con', 'senza', 'oggetto', 'allegato', 'per', 'data', 'giorno', 'settimana',
            'mese', 'anno', 'prima', 'dopo', 'tra', 'chi', 'quando', 'dove', 'filtra', 'solo', 'tutti',
            'ultimi', 'precedenti', 'inoltrati', 'importanti', 'riunione', 'progetto', 'cliente',
            'rapporto', 'promemoria', 'contiene', 'testo', 'parole', 'risposta', 'oggi', 'ieri',
            'domani', 'fammi', 'mostrami', 'cercami', 'trovami'
        },
        # Portugués
        'pt': {
            'buscar', 'procurar', 'encontrar', 'mostrar', 'listar', 'obter', 'emails', 'mensagens',
            'enviados', 'recebidos', 'de', 'para', 'com', 'sem', 'assunto', 'anexo', 'por', 'data',
            'dia', 'semana', 'mês', 'ano', 'antes', 'depois', 'entre', 'quem', 'quando', 'onde',
            'filtrar', 'somente', 'todos', 'últimos', 'anteriores', 'reenviados', 'importantes',
            'reunião', 'projeto', 'cliente', 'relatório', 'lembrete', 'contém', 'texto', 'palavras',
            'resposta', 'ontem', 'hoje', 'amanhã', 'me mostre', 'procure', 'ache', 'encontre'
        }
    }

    def __init__(self, cache_size: int = 256, default_language: str = "xx"):
        self.default_language = default_language
        self.nlp_models: Dict[str, Language] = {}

        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )

        self._load_models()

        print(f"[NER] Modelos cargados: {list(self.nlp_models.keys())}")

    def _load_models(self):

        priority_langs = ["en", "es", "fr", "de", "it", "pt", "xx"]

        for lang in priority_langs:
            model_loaded = False
            for model_name in self.MODELS.get(lang, []):
                try:

                    nlp = spacy.load(
                        model_name,
                        disable=["parser", "lemmatizer", "textcat", "attribute_ruler"],
                    )
                    self.nlp_models[lang] = nlp
                    print(f"[NER]  {model_name} cargado")
                    model_loaded = True
                    break
                except OSError:
                    continue

            if not model_loaded:
                print(f"[NER]  No se encontró modelo para '{lang}'")

    def detect_language(self, text: str) -> str:

        text_lower = text.lower()
        words = set(text_lower.split())

        scores = {}
        for lang, indicators in self.LANGUAGE_INDICATORS.items():
            score = len(words & indicators)
            if score > 0:
                scores[lang] = score

        if scores:
            detected_lang = max(scores.items(), key=lambda x: x[1])[0]

            if detected_lang in self.nlp_models:
                return detected_lang

        if self.default_language in self.nlp_models:
            return self.default_language

        return list(self.nlp_models.keys())[0]

    @lru_cache(maxsize=256)
    def extract_contacts(
        self, query: str, language: Optional[str] = None
    ) -> Tuple[str, ...]:

        contacts = set()

        email_matches = self.email_pattern.findall(query)
        contacts.update([e.lower() for e in email_matches])

        if language is None:
            language = self.detect_language(query)

        print(f"[NER] Idioma detectado: {language}")

        if language in self.nlp_models:
            ner_contacts = self._extract_entities(query, language)
            contacts.update(ner_contacts)
        else:
            print(f"[NER] Modelo para '{language}' no disponible")

        if not contacts:
            print("[NER] No se encontraron contactos,")
            return None

        result_list = list(contacts)

        result_list.sort(key=lambda s: (-len(s.split()), -len(s), s))

        result = tuple(result_list)

        if result:
            print(f"[NER]  Contactos extraídos (ordenados): {result}")
        else:
            print("[NER]  No se encontraron contactos")

        return result

    def _extract_entities(self, text: str, language: str) -> Set[str]:

        contacts = set()
        nlp = self.nlp_models.get(language)

        if not nlp:
            return contacts

        try:
            doc = nlp(text)

            for ent in doc.ents:

                if ent.label_ in ("PERSON", "PER"):
                    name = ent.text.strip()

                    if not self._is_valid_entity(name):
                        continue

                    contacts.add(name.lower())

                    if " " in name:
                        parts = name.split()

                        first_name = parts[0]
                        last_name = parts[-1]

                        if len(first_name) > 2:
                            contacts.add(first_name.lower())
                        if len(last_name) > 2 and last_name != first_name:
                            contacts.add(last_name.lower())

        except Exception as e:
            print(f"[NER] Error procesando con '{language}': {e}")

        return contacts

    @staticmethod
    def _is_valid_entity(name: str) -> bool:

        if not name or len(name) < 2:
            return False

        if not any(c.isalpha() for c in name):
            return False

        digit_ratio = sum(c.isdigit() for c in name) / len(name)
        if digit_ratio > 0.5:
            return False

        stopwords = {
            # Inglés
            'the', 'and', 'for', 'with', 'from', 'this', 'that', 'these', 'those',
            'a', 'an', 'of', 'to', 'in', 'on', 'at', 'by', 'as', 'it', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'i', 'you', 'he', 'she', 'we', 'they', 'them', 'their', 'my',
            'your', 'our', 'his', 'her', 'its', 'not', 'but', 'or', 'if', 'then', 'so', 'such',
            'about', 'into', 'out', 'up', 'down', 'over', 'under', 'again', 'more', 'most', 'no',
            'nor', 'only', 'own', 'same', 'too', 'very', 'can', 'will', 'just', 'do', 'does', 'did',

            # Español
            'el', 'la', 'los', 'las', 'de', 'del', 'para', 'con', 'por', 'este', 'esta', 'estos', 'estas',
            'un', 'una', 'unos', 'unas', 'en', 'y', 'o', 'a', 'que', 'se', 'su', 'sus', 'al', 'como',
            'lo', 'le', 'les', 'me', 'te', 'mi', 'tu', 'nos', 'ya', 'muy', 'sin', 'sobre', 'entre',
            'también', 'cuando', 'donde', 'qué', 'quien', 'cual', 'porque', 'todo', 'todos', 'todas',
            'cada', 'más', 'menos', 'ser', 'estar', 'hay', 'fue', 'son', 'era', 'han', 'ha',

            # Francés
            'le', 'la', 'les', 'de', 'du', 'des', 'pour', 'avec', 'un', 'une', 'dans', 'en', 'sur', 
            'par', 'au', 'aux', 'ce', 'cet', 'cette', 'ces', 'et', 'ou', 'mais', 'ne', 'pas', 'se', 
            'sa', 'son', 'ses', 'leur', 'leurs', 'nous', 'vous', 'ils', 'elles', 'qui', 'que', 'quoi', 
            'dont', 'où', 'comme', 'plus', 'moins', 'tout', 'tous', 'toute', 'toutes', 'été', 'être', 'avoir',

            # Alemán
            'der', 'die', 'das', 'den', 'dem', 'des', 'für', 'mit', 'und', 'oder', 'aber', 'nicht', 'ein',
            'eine', 'einer', 'eines', 'einem', 'einen', 'im', 'in', 'an', 'auf', 'aus', 'bei', 'von',
            'zu', 'zum', 'zur', 'nach', 'über', 'unter', 'zwischen', 'dass', 'wie', 'so', 'noch', 'schon',
            'wir', 'ihr', 'sie', 'er', 'es', 'mein', 'dein', 'sein', 'ihr', 'unser', 'euer', 'dies', 'das',

            # Italiano
            'il', 'lo', 'la', 'gli', 'le', 'i', 'di', 'a', 'da', 'in', 'su', 'con', 'per', 'tra', 'fra',
            'un', 'una', 'uno', 'e', 'o', 'ma', 'che', 'come', 'più', 'meno', 'molto', 'poco', 'questo',
            'quello', 'quella', 'quelli', 'quelle', 'suo', 'sua', 'loro', 'mio', 'mia', 'tuo', 'tua',
            'nostro', 'vostro', 'sono', 'era', 'sei', 'hai', 'avere', 'essere',

            # Portugués
            'o', 'a', 'os', 'as', 'do', 'da', 'dos', 'das', 'para', 'com', 'por', 'em', 'no', 'na',
            'nos', 'nas', 'um', 'uma', 'uns', 'umas', 'de', 'se', 'que', 'é', 'foi', 'era', 'ser', 
            'estar', 'são', 'também', 'como', 'mas', 'ou', 'porque', 'onde', 'quando', 'muito', 'pouco',
            'este', 'esta', 'esses', 'essas', 'seu', 'sua', 'meu', 'minha', 'teu', 'tua', 'nosso', 'nossa'
        }

        
        if name.lower() in stopwords:
            return False

        return True

    def get_loaded_models(self) -> List[str]:

        return list(self.nlp_models.keys())


class OllamaHandler:
    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        base_url: str = "http://localhost:11434",
        timeout: int = 240,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = 0.3
        self.max_tokens = 1000

    def generate(self, prompt: str, context: Optional[str] = None):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_gpu": 0,
            },
        }

        full_response = ""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True,
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            response_part = chunk.get("response", "")
                            full_response += response_part
                            yield response_part

                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                raise Exception(f"Error en Ollama: {response.status_code}")

        except Exception as e:
            print(f"Error generando respuesta: {e}")
            yield "Error al generar la respuesta."


class EmailRAGEngine:
    def __init__(
        self,
        vector_db,
        contact_db=None,
        ollama_handler: Optional[OllamaHandler] = None,
        use_contact_filter: bool = True,
        default_language: str = "en",
    ):

        self.vector_db = vector_db
        self.contact_db = contact_db
        self.ollama = ollama_handler or OllamaHandler()
        self.use_contact_filter = use_contact_filter

        self.contact_extractor = MultilingualContactExtractor(
            default_language=default_language
        )

        self.top_k_retrieval: int = 7
        self.top_k_contacts: int = 5
        self.min_similarity_threshold: float = 0.3
        self.language: str = "es"

    def _extract_contact_queries(
        self, question: str, language: Optional[str] = None
    ) -> List[str]:
        contacts_tuple = self.contact_extractor.extract_contacts(question, language)
        if not contacts_tuple:
            return []
        return list(contacts_tuple)

    def _create_prompt(self, query: str, context: str) -> str:
        if self.language == "es":
            system_message = """Eres un asistente experto en búsqueda y análisis de correos electrónicos.

Tu tarea es responder preguntas basándote ÚNICAMENTE en los emails proporcionados como contexto, en los emails puedes encontrar información como su id, quien lo manda, quien lo recibe,
la fecha, el asunto y el cuerpo del email. Los emails están ordenados por relevancia.
            
Reglas importantes:
1. Responde SOLO con información que aparece en los emails proporcionados
2. SIEMPRE cita la fuente: indica qué EMAIL(s) contiene(n) la información
3. Sé muy conciso pero completo
4. Si múltiples emails tienen información relevante, menciona los mas importantes 
5. Usa un tono profesional pero amigable

Formato de respuesta:
- Respuesta directa a la pregunta"""

            user_prompt = f"""Contexto de emails relevantes:
{context}

Pregunta del usuario: {query}

Por favor, responde la pregunta basándote en los emails anteriores. Recuerda citar las fuentes."""

        else:
            system_message = """You are an expert assistant for email search and analysis.

Your task is to answer questions based ONLY on the emails provided as context.

Important rules:
1. Answer ONLY with information from the provided emails
2. ALWAYS cite sources: indicate which EMAIL(s) contain the information
3. If information is not in the emails, clearly state "I did not find this information in the provided emails"
4. Be concise but complete
5. If multiple emails have relevant information, mention all of them
6. Use a professional but friendly tone

Response format:
- Direct answer to the question
- Sources: [EMAIL X, EMAIL Y]"""

            user_prompt = f"""Context from relevant emails:
{context}

User question: {query}

Please answer the question based on the above emails. Remember to cite sources."""

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt

    def query(
        self,
        question: str,
        n_results: int = 7,
        filters: Optional[Dict] = None,
        language: Optional[str] = None,
    ):

        start_time = datetime.now()
        filtered_email_ids = None
        contact_filter_applied = False

        if self.use_contact_filter and self.contact_db:
            contact_queries = self._extract_contact_queries(question, language)

            if not contact_queries:
                print(
                    "[CONTACT FILTER] No se extrajeron contactos, buscando en TODOS los emails."
                )
            else:
                print(f"[CONTACT FILTER] Buscando contactos: {contact_queries}")

                email_scores: Dict[str, float] = {}

                for cq in contact_queries:
                    weight = max(1, len(cq.split()))
                    contacts = self.contact_db.search_contacts(
                        cq, n_results=self.top_k_contacts
                    )
                    if contacts:
                        print(
                            f"[CONTACT FILTER] '{cq}' -> {len(contacts)} matches (peso={weight})"
                        )
                        for contact in contacts:
                            for eid in contact.get("email_ids", []):
                                email_scores[eid] = email_scores.get(eid, 0.0) + weight
                    else:
                        print(f"[CONTACT FILTER] '{cq}' -> Sin matches en ContactDB")

                if email_scores:
                    sorted_ids = sorted(
                        email_scores.items(), key=lambda x: (-x[1], -len(x[0]))
                    )
                    top_ids = [eid for eid, _ in sorted_ids]
                    filtered_email_ids = top_ids
                    contact_filter_applied = True
                    print(
                        f"[CONTACT FILTER] Aplicando filtro: {len(filtered_email_ids)} emails seleccionados"
                    )
                else:
                    print(f"[CONTACT FILTER] No se obtuvieron emails desde ContactDB")
                    print(
                        f"[CONTACT FILTER] Buscando en TODOS los emails sin filtro de contacto"
                    )

                print(
                    f"[CONTACT FILTER] ⏱Tiempo extracción contactos: {(datetime.now() - start_time).total_seconds():.2f}s"
                )
        else:
            print(
                "[CONTACT FILTER] Filtro de contactos desactivado, buscando en todos los emails"
            )

        if filtered_email_ids:
            where_filter = {"email_id": {"$in": filtered_email_ids}}
            if filters:
                where_filter = {"$and": [where_filter, filters]}
            print(
                f"[SEARCH] Búsqueda con filtro de contacto: {len(filtered_email_ids)} emails"
            )
        else:
            where_filter = filters
            print(f"[SEARCH] Búsqueda global sin filtro de contacto")

        print(
            f"[SEARCH] Tiempo pre búsqueda: {(datetime.now() - start_time).total_seconds():.2f}s"
        )

        search_results = self.vector_db.search_with_reranking(
            query=question,
            n_results=n_results,
            filter_metadata=where_filter,
            group_threads=True,
        )
        # with open("email_scores.txt", 'w', encoding='utf-8') as f:
        #     f.write(str(search_results))
        if not search_results["results"]:
            yield " No encontré emails relevantes para tu pregunta."
            yield {
                "type": "metadata",
                "sources": [],
                "model": self.ollama.model_name,
                "time": (datetime.now() - start_time).total_seconds(),
                "ner_models": self.contact_extractor.get_loaded_models(),
                "contact_filter_applied": contact_filter_applied,
                "total_filtered_emails": len(filtered_email_ids)
                if filtered_email_ids
                else 0,
            }
            return

        relevant_results = search_results["results"]
        context = self._build_context(question, relevant_results)
        answer_generator = self._generate_answer(question, context)

        for chunk in answer_generator:
            yield chunk

        sources = self._prepare_sources(relevant_results)
        time_elapsed_seconds = (datetime.now() - start_time).total_seconds()

        total_emails = sum(result.get("thread_size", 1) for result in relevant_results)
        threads_count = sum(
            1 for result in relevant_results if result.get("is_thread", False)
        )

        yield {
            "type": "metadata",
            "sources": sources,
            "model": self.ollama.model_name,
            "time": time_elapsed_seconds,
            "contact_filter_applied": contact_filter_applied,
            "total_filtered_emails": len(filtered_email_ids)
            if filtered_email_ids
            else 0,
            "ner_models": self.contact_extractor.get_loaded_models(),
            "detected_language": self.contact_extractor.detect_language(question),
            "results_count": len(relevant_results),
            "threads_count": threads_count,
            "total_emails_in_context": total_emails,
        }

    def _build_context(self, query: str, retrieved_results: List[Dict]) -> str:
        if not retrieved_results:
            return "No se encontraron emails relevantes."

        context_parts = []
        email_counter = 1

        for result in retrieved_results:
            is_thread = result.get("is_thread", False)

            if is_thread:

                thread_id = result.get("thread_id", "unknown")
                thread_size = result.get("thread_size", 0)
                thread_emails = result.get("thread_emails", [])

                thread_header = f"\n{'='*70}\n"
                thread_header += f"HILO DE CONVERSACIÓN (Thread ID: {thread_id})\n"
                thread_header += f"Total de emails: {thread_size}\n"
                thread_header += f"Relevancia del hilo: {result['best_distance']:.4f}\n"
                thread_header += f"{'='*70}\n"
                context_parts.append(thread_header)

                for email in thread_emails:
                    email_info = f"ID: {email['email_id']}\n"
                    email_info += f"De: {email['from']}\n"
                    email_info += f"Para: {email['to']}\n"
                    email_info += f"Asunto: {email['subject']}\n"
                    email_info += f"Fecha: {email['date']}\n"
                    email_info += (
                        f"Relevancia individual: {email['best_distance']:.4f}\n"
                    )

                    email_info += "\nContenido:\n"
                    sorted_chunks = sorted(email["chunks"], key=lambda x: x["distance"])
                    for chunk in sorted_chunks[:2]:
                        chunk_preview = chunk["text"][:400]
                        email_info += f"{chunk_preview}...\n\n"

                    context_parts.append(email_info)
                    email_counter += 1

                context_parts.append(f"{'='*70}\n")

            else:

                email_info = f"ID: {result['email_id']}\n"
                email_info += f"De: {result['from']}\n"
                email_info += f"Para: {result['to']}\n"
                email_info += f"Asunto: {result['subject']}\n"
                email_info += f"Fecha: {result['date']}\n"
                email_info += f"Relevancia: {result['best_distance']:.4f}\n"

                email_info += "\nContenido relevante:\n"
                sorted_chunks = sorted(result["chunks"], key=lambda x: x["distance"])
                for chunk in sorted_chunks[:2]:
                    chunk_preview = chunk["text"][:400]
                    email_info += f"{chunk_preview}...\n\n"

                context_parts.append(email_info)
                email_counter += 1

        full_context = "\n".join(context_parts)

        print(
            f"[CONTEXT] Contexto construido: {len(full_context)} caracteres, {email_counter-1} emails"
        )

        return full_context

    def _prepare_sources(self, results: List[Dict]) -> List[Dict[str, Any]]:
        sources = []
        for result in results:
            is_thread = result.get("is_thread", False)

            if is_thread:

                thread_emails = result.get("thread_emails", [])
                source = {
                    "type": "thread",
                    "thread_id": result.get("thread_id", ""),
                    "thread_size": result.get("thread_size", 0),
                    "subject": result["subject"],
                    "relevance_score": round(result["best_distance"], 3),
                    "emails": [
                        {
                            "email_id": email["email_id"],
                            "from": email["from"],
                            "message_id": email["message_id"],
                            "to": email["to"],
                            "date": email["date"],
                            "relevance_score": round(email["best_distance"], 3),
                        }
                        for email in thread_emails
                    ],
                }
            else:

                source = {
                    "type": "email",
                    "email_id": result["email_id"],
                    "message_id": result["message_id"],
                    "subject": result["subject"],
                    "from": result["from"],
                    "to": result["to"],
                    "date": result["date"],
                    "relevance_score": round(result["best_distance"], 3),
                }

            sources.append(source)

        return sources

    def _generate_answer(self, question: str, context: str) -> str:
        prompt = self._create_prompt(question, context)
        # with open("prompt.txt", 'w', encoding='utf-8') as f:
        #     f.write(f"{prompt}\n")
        return self.ollama.generate(prompt)

class LLMEvaluator:

    def __init__(self, evaluator_model: str = "llama3.2:3b"):
        self.ollama = OllamaHandler(model_name=evaluator_model)
        self.ollama.temperature = 0.1  
        self.ollama.max_tokens = 2000
    
    def _create_evaluation_prompt(
        self,
        query: str,
        context: str,
        response: str,
        sources: List[Dict],
        expected_email_id: Optional[str] = None
    ) -> str:
        """Crea el prompt para evaluar la respuesta"""
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Eres un evaluador experto de sistemas RAG (Retrieval-Augmented Generation) para correos electrónicos.

Tu tarea es evaluar la calidad de una respuesta generada por un asistente de IA que responde preguntas sobre emails.

Debes evaluar los siguientes aspectos en una escala del 1 al 10:

1. **PRECISIÓN** (1-10): ¿La respuesta es factualmente correcta según el contexto proporcionado?
2. **RELEVANCIA** (1-10): ¿La respuesta responde directamente a la pregunta del usuario?
3. **USO_DE_FUENTES** (1-10): ¿La respuesta cita correctamente las fuentes (emails) utilizadas?
4. **COHERENCIA** (1-10): ¿La respuesta es clara, bien estructurada y fácil de entender?
5. **COMPLETITUD** (1-10): ¿La respuesta es completa o le falta información relevante?
6. **RECUPERACIÓN** (1-10): ¿Los emails recuperados son relevantes para responder la pregunta?

FORMATO DE RESPUESTA OBLIGATORIO:
PRECISIÓN: [número]
RELEVANCIA: [número]
USO_DE_FUENTES: [número]
COHERENCIA: [número]
COMPLETITUD: [número]
RECUPERACIÓN: [número]
COMENTARIOS: [tu análisis detallado]

IMPORTANTE: Debes responder EXACTAMENTE en ese formato, con los números del 1 al 10.<|eot_id|><|start_header_id|>user<|end_header_id|>

PREGUNTA DEL USUARIO:
{query}

CONTEXTO PROPORCIONADO (Emails recuperados):
{context[:3000]}...

RESPUESTA GENERADA:
{response}

FUENTES UTILIZADAS:
{self._format_sources(sources)}
"""
        
        if expected_email_id:
            email_found = self._check_email_in_sources(expected_email_id, sources)
            prompt += f"""
EMAIL ESPERADO: {expected_email_id}
¿ESTÁ EN LAS FUENTES?: {'SÍ' if email_found else 'NO'}
"""
        
        prompt += """
Por favor, evalúa la respuesta según los criterios mencionados.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def _format_sources(self, sources: List[Dict]) -> str:
        formatted = []
        for idx, source in enumerate(sources, 1):
            if source.get("type") == "thread":
                formatted.append(f"{idx}. HILO - {source.get('subject')} (Relevancia: {source.get('relevance_score')})")
            else:
                formatted.append(f"{idx}. EMAIL [{source.get('email_id')}] - {source.get('subject')} (Relevancia: {source.get('relevance_score')})")
        return "\n".join(formatted)
    
    def _check_email_in_sources(self, email_id: str, sources: List[Dict]) -> bool:
        for source in sources:
            if source.get("type") == "thread":
                for email in source.get("emails", []):
                    if email.get("email_id") == email_id:
                        return True
            elif source.get("email_id") == email_id:
                return True
        return False
    
    def _parse_evaluation(self, raw_response: str) -> Dict[str, Any]:
        scores = {
            "precisión": 0,
            "relevancia": 0,
            "uso_de_fuentes": 0,
            "coherencia": 0,
            "completitud": 0,
            "recuperación": 0,
            "comentarios": ""
        }
        
        lines = raw_response.strip().split('\n')
        comentarios_started = False
        comentarios_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("COMENTARIOS:"):
                comentarios_started = True
                comentarios_lines.append(line.replace("COMENTARIOS:", "").strip())
            elif comentarios_started:
                comentarios_lines.append(line)
            else:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace("_", "_")
                    try:
                        score = int(value.strip().split()[0])
                        if "PRECISIÓN" in line or "PRECISION" in line:
                            scores["precisión"] = score
                        elif "RELEVANCIA" in line:
                            scores["relevancia"] = score
                        elif "USO" in line or "FUENTES" in line:
                            scores["uso_de_fuentes"] = score
                        elif "COHERENCIA" in line:
                            scores["coherencia"] = score
                        elif "COMPLETITUD" in line:
                            scores["completitud"] = score
                        elif "RECUPERACIÓN" in line or "RECUPERACION" in line:
                            scores["recuperación"] = score
                    except (ValueError, IndexError):
                        continue
        
        scores["comentarios"] = " ".join(comentarios_lines).strip()
        return scores
    
    def evaluate(
        self,
        query: str,
        context: str,
        response: str,
        sources: List[Dict],
        expected_email_id: Optional[str] = None,
        response_time: float = 0.0
    ) -> Dict[str, Any]:

        print(f"\n[EVALUATOR] Evaluando respuesta...")
        
        prompt = self._create_evaluation_prompt(
            query, context, response, sources, expected_email_id
        )
        
        raw_evaluation = ""
        for chunk in self.ollama.generate(prompt):
            raw_evaluation += chunk
        
        scores = self._parse_evaluation(raw_evaluation)
        
        score_values = [
            scores["precisión"],
            scores["relevancia"],
            scores["uso_de_fuentes"],
            scores["coherencia"],
            scores["completitud"],
            scores["recuperación"]
        ]
        avg_score = sum(score_values) / len(score_values) if score_values else 0
        
        email_found = False
        if expected_email_id:
            email_found = self._check_email_in_sources(expected_email_id, sources)
        
        result = {
            "scores": scores,
            "average_score": round(avg_score, 2),
            "response_time": response_time,
            "email_found": email_found if expected_email_id else None,
            "expected_email_id": expected_email_id,
            "raw_evaluation": raw_evaluation
        }
        
        print(f"[EVALUATOR]  Promedio: {result['average_score']}/10")
        
        return result
def open_email(msg_id):
    if os.path.exists('/.dockerenv'):
        print(f" [INFO] Estás en Docker. Para ver el correo, busca el ID: {msg_id}")
    else:
        try:
            subprocess.run(
                ["xdg-open", f"mid:{msg_id}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("xdg-open no está instalado.")
            
def start_interactive_session(db_path, contact_db_path):
    try:


        embedder = MultilingualEmbedder()
        contact_db = ContactVectorDB(db_path=contact_db_path, embedder=embedder)
        db = EmailVectorDB(db_path=db_path, embedder=embedder)

        print(" EmailVectorDB y ContactDB cargados correctamente.\n")
    except Exception as e:
        print(f" Error inicializando DBs: {e}")
        db = None
        contact_db = None

    ollama_instance = OllamaHandler(model_name="llama3.2:1b")

    try:
        rag = (
            EmailRAGEngine(
                vector_db=db,
                contact_db=contact_db,
                ollama_handler=ollama_instance,
                use_contact_filter=True,
                default_language="en",
            )
            if db is not None
            else None
        )

        if rag:
            print(
                f"\nModelos NER cargados: {', '.join(rag.contact_extractor.get_loaded_models())}"
            )
            print(f"Modelo LLM: {ollama_instance.model_name}")
            print("\nEscribe tu consulta o 'exit' para salir.\n")
    except Exception as e:
        print(f" Error creando EmailRAGEngine: {e}")
        rag = None

    if rag is not None:
        try:
            while True:
                q = input("\n\033[1;34mPregunta > \033[0m").strip()
                if not q: continue
                if q.lower() in ("exit", "quit", "salir"): break
                
                last_metadata = None
                for item in rag.query(q):
                    if isinstance(item, str):
                        print(item, end="", flush=True)
                    elif isinstance(item, dict) and item.get("type") == "metadata":
                        last_metadata = item

                print("\n")
                
                if last_metadata and last_metadata.get("sources"):
                    sources = last_metadata["sources"]
                    
                    while True:
                        print("\n" + "─" * 40)
                        print(f"\033[1mFUENTES ENCONTRADAS ({len(sources)})\033[0m")
                        for idx, s in enumerate(sources, 1):
                            tipo = "HILO" if s.get("type") == "thread" else "EMAIL"
                            sub = s.get('subject', 'Sin Asunto')
                            print(f"  \033[1;33m{idx}\033[0m. {tipo} Asunto --> {sub[:60]}...")
                        
                        print("-" * 40)
                        print("  \033[1m[#]\033[0m Abrir fuente en Thunderbird")
                        print("  \033[1m[N]\033[0m Nueva pregunta")
                        print("  \033[1m[Q]\033[0m Salir")
                        print("─" * 40)

                        choice = input("\033[1mSelecciona una opción > \033[0m").strip().lower()

                        if choice == 'n':
                            break 
                        elif choice == 'q':
                            print("Saliendo...")
                            return 
                        elif choice.isdigit():
                            idx = int(choice) - 1
                            if 0 <= idx < len(sources):
                                selected = sources[idx]
                                msg_id = selected.get('message_id')
                                if not msg_id and selected.get('type') == 'thread':
                                    msg_id = selected.get('emails')[0].get('message_id')
                                
                                if msg_id:
                                    print(f"Abriendo: {selected.get('subject')}...")
                                    open_email(msg_id)
                                else:
                                    print("No se pudo encontrar el ID del mensaje.")
                            else:
                                print("Número fuera de rango.")
                        else:
                            print("Opción no válida.")
                else:
                    print("No se encontraron fuentes para esta consulta.")

        except KeyboardInterrupt:
            print("\n\nSaliendo...")
    pass

# ---CUSTOM---
DB_PATH = "data/emails_vectordb"
CONTACT_DB_PATH = "data/emails_vectordb_contacts"

# ---ENRON---
# DB_PATH = "data/test_vectordb"
# CONTACT_DB_PATH = "data/test_vectordb_contacts"

@click.command()
@click.option("--db-path", default=DB_PATH, help="Ruta al VectorDB de emails")
@click.option("--contact-db-path", default=CONTACT_DB_PATH, help="Ruta al VectorDB de contactos")
def main_cli(db_path, contact_db_path):
    start_interactive_session(db_path, contact_db_path)

if __name__ == "__main__":
    main_cli()
    
# if __name__ == "__main__":
#     # ---CUSTOM---
#     # DB_PATH = "data/emails_vectordb"
#     # CONTACT_DB_PATH = "data/emails_vectordb_contacts"

#     # ---ENRON---
#     DB_PATH = "data/test_vectordb"
#     CONTACT_DB_PATH = "data/test_vectordb_contacts"
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--db-path", type=str, default=DB_PATH)
#     parser.add_argument("--contact-db-path", type=str, default=CONTACT_DB_PATH)
#     args = parser.parse_args()
    
#     try:
#         from embeddings_system import (
#             EmailVectorDB,
#             ContactVectorDB,
#             MultilingualEmbedder,
#         )

#         embedder = MultilingualEmbedder()
#         contact_db = ContactVectorDB(db_path=CONTACT_DB_PATH, embedder=embedder)
#         db = EmailVectorDB(db_path=DB_PATH, embedder=embedder)

#         print(" EmailVectorDB y ContactDB cargados correctamente.\n")
#     except Exception as e:
#         print(f" Error inicializando DBs: {e}")
#     except Exception as e:
#         print(f" Error creando EmailRAGEngine: {e}")
#         rag = None

#     if rag is not None:
#         try:
#             while True:
#                 q = input("\nPregunta > ").strip()
#                 if not q:
#                     continue
#                 if q.lower() in ("exit", "quit", "salir"):
#                     break

#                 try:
#                     print(f"\nRespuesta:\n", end="", flush=True)

#                     last_element = None
#                     for item in rag.query(q):
#                         if isinstance(item, str):
#                             print(item, end="", flush=True)
#                         elif isinstance(item, dict) and item.get("type") == "metadata":
#                             last_element = item

#                     print("\n")
                    
#                     if last_element:
#                         print("─" * 70)
#                         print(f"    Tiempo: {last_element['time']:.2f}s")
#                         print(f"    Modelo LLM: {last_element['model']}")
#                         print(f"    NER: {', '.join(last_element['ner_models'])}")
#                         print(
#                             f"    Idioma detectado: {last_element['detected_language'].upper()}"
#                         )

#                         print(
#                             f"    Resultados: {last_element['results_count']} "
#                             f"({last_element['threads_count']} hilos, "
#                             f"{last_element['total_emails_in_context']} emails totales)"
#                         )

#                         if last_element.get("filtered_by_contact"):
#                             print(
#                                 f" Filtrado por contacto: {last_element['total_filtered_emails']} emails"
#                             )

#                         if last_element.get("sources"):
#                             print(f"\n  Fuentes ({len(last_element['sources'])}):")
#                             for idx, s in enumerate(last_element["sources"], 1):
#                                 if s.get("type") == "thread":
#                                     print(f"  {idx}. HILO: {s.get('subject')}")
#                                     print(
#                                         f"     Thread ID: {s.get('thread_id')} ({s.get('thread_size')} emails)"
#                                     )
#                                     print(
#                                         f"     Relevancia: {s.get('relevance_score')}"
#                                     )
#                                 else:
#                                     print(
#                                         f"  {idx}.  [{s.get('email_id')}]  |  {s.get('subject')}"
#                                     )
#                                     print(f"     De: {s.get('from')} | {s.get('date')}")
#                                     print(
#                                         f"     Relevancia: {s.get('relevance_score')}"
#                                     )
#                                 subprocess.run(["xdg-open", f"mid:{s.get('message_id')}"])

#                         print("─" * 70)

#                 except Exception as e:
#                     print(f"\n Error: {e}")
#                     import traceback

#                     traceback.print_exc()

#         except KeyboardInterrupt:
#             print("\n\nSaliendo...")
