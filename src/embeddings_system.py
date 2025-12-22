import time

import click
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
from tqdm import tqdm
from src.data_loader import Email
import argparse
import statistics
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MultilingualEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
    ):
        print(f"Modelo: {model_name} (device: {device})")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        if not texts:
            return np.array([])

        texts = [t if t else " " for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True)


class ContactVectorDB:
    def __init__(
        self,
        db_path: str = "data/vectordb_contacts",
        collection_name: str = "contacts",
        embedder: Optional[MultilingualEmbedder] = None,
    ):

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder or MultilingualEmbedder()

        print(f"Inicializando ContactDB en {self.db_path}")
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        try:
            self.collection = self.client.get_collection(collection_name)
            print(
                f"Colección '{collection_name}' cargada. Contactos: {self.collection.count()}"
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Contact search collection"},
            )
            print(f"Nueva colección '{collection_name}' creada")

    def _extract_display_name(self, email_str: str) -> str:
        match = re.match(r"^(.+?)\s*<(.+?)>$", email_str)
        if match:
            return match.group(1).strip()

        if "@" in email_str:
            username = email_str.split("@")[0]
            return username.replace(".", " ").replace("_", " ")

        return email_str

    def _normalize_email(self, email_str: str) -> str:
        match = re.search(r"<(.+?)>", email_str)
        if match:
            return match.group(1).strip().lower()
        return email_str.strip().lower()

    def index_contacts(self, emails: List[Any], batch_size: int = 100):

        contact_map = {}

        for email in tqdm(emails, desc="Extrayendo contactos"):
            all_addresses = []

            if email.from_address:
                all_addresses.append((email.from_address, "sent"))

            for addr in email.to_addresses or []:
                all_addresses.append((addr, "received"))

            for addr in email.cc_addresses or []:
                all_addresses.append((addr, "received"))

            for addr in email.bcc_addresses or []:
                all_addresses.append((addr, "received"))

            for addr_str, role in all_addresses:
                email_addr = self._normalize_email(addr_str)

                if email_addr not in contact_map:
                    contact_map[email_addr] = {
                        "email": email_addr,
                        "display_name": self._extract_display_name(addr_str),
                        "sent_ids": [],
                        "received_ids": [],
                    }

                if role == "sent":
                    if email.id not in contact_map[email_addr]["sent_ids"]:
                        contact_map[email_addr]["sent_ids"].append(email.id)
                else:
                    if email.id not in contact_map[email_addr]["received_ids"]:
                        contact_map[email_addr]["received_ids"].append(email.id)

        documents = []
        metadatas = []
        ids = []

        for email_addr, contact_info in contact_map.items():
            display_name = contact_info["display_name"]
            username = email_addr.split("@")[0].replace(".", " ").replace("_", " ")

            doc_text = f"{display_name} {username} {email_addr}"

            documents.append(doc_text)
            metadatas.append(
                {
                    "email_address": email_addr,
                    "display_name": display_name,
                    "sent_ids": json.dumps(contact_info["sent_ids"]),
                    "received_ids": json.dumps(contact_info["received_ids"]),
                    "total_emails": len(contact_info["sent_ids"])
                    + len(contact_info["received_ids"]),
                }
            )
            ids.append(f"contact_{email_addr}")

        if not documents:
            print("No hay contactos para indexar")
            return

        print(f"Indexando {len(documents)} contactos únicos...")
        embeddings = self.embedder.encode(documents, batch_size=batch_size)

        for i in tqdm(
            range(0, len(documents), batch_size), desc="Insertando contactos"
        ):
            end_idx = min(i + batch_size, len(documents))

            self.collection.add(
                embeddings=embeddings[i:end_idx].tolist(),
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx],
            )

        print(f"Total contactos en ContactDB: {self.collection.count()}")

    def search_contacts(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:

        query_embedding = self.embedder.encode_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )

        if not results["ids"][0]:
            return []

        contacts = []
        for idx in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][idx]

            sent_ids = json.loads(metadata["sent_ids"])
            received_ids = json.loads(metadata["received_ids"])
            all_ids = list(set(sent_ids + received_ids))

            contacts.append(
                {
                    "email_address": metadata["email_address"],
                    "display_name": metadata["display_name"],
                    "email_ids": all_ids,
                    "sent_count": len(sent_ids),
                    "received_count": len(received_ids),
                    "distance": results["distances"][0][idx],
                }
            )

        return contacts

    def delete_collection(self):
        try:
            self.client.delete_collection(self.collection.name)
            print(f"Colección '{self.collection.name}' eliminada")
        except Exception as e:
            print(f"Error eliminando colección: {e}")

    def delete_db(self):
        try:
            self.client.delete_collection(self.collection.name)
        except:
            pass

        try:
            if self.db_path.exists():
                shutil.rmtree(self.db_path)
                print(f"ContactDB eliminada: {self.db_path}")
        except Exception as e:
            print(f"Error eliminando ContactDB: {e}")


class EmailVectorDB:
    def __init__(
        self,
        db_path: str = "data/vectordb",
        collection_name: str = "emails",
        embedder: Optional[MultilingualEmbedder] = None,
        contact_db: Optional[ContactVectorDB] = None,
    ):

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder or MultilingualEmbedder()

        print(f"Inicializando ChromaDB en {self.db_path}")
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        try:
            self.collection = self.client.get_collection(collection_name)
            print(
                f"Colección '{collection_name}' cargada. Documentos: {self.collection.count()}"
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Email search collection"},
            )
            print(f"Nueva colección '{collection_name}' creada")

        self.contact_db = contact_db
        self.chunk_size = 300
        self.chunk_overlap = 50

    def chunk_text(
        self, text: str, chunk_size: int = 300, overlap: int = 50
    ) -> List[Tuple[str, int, int]]:
        if not text:
            return []

        words = text.split()
        chunks = []

        if len(words) <= chunk_size:
            return [(text, 0, len(text))]

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)

            start_pos = len(" ".join(words[:i])) if i > 0 else 0
            end_pos = start_pos + len(chunk_text)

            chunks.append((chunk_text, start_pos, end_pos))

            if i + chunk_size >= len(words):
                break

        return chunks

    def _normalize_text(self, text: str) -> str:

        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"[=\-_]{3,}", " ", text)

        text = re.sub(r"http[s]?://\S+", "[URL]", text)

        text = re.sub(r"\b[\w.]+@[\w.]+\b", "[EMAIL]", text)

        return text.strip()

    def index_emails(self, emails: List[Any], batch_size: int = 100):

        if self.contact_db:
            print("\n=== Indexando ContactDB ===")
            self.contact_db.index_contacts(emails, batch_size=batch_size)

        print("\n=== Indexando EmailDB ===")
        documents = []
        metadatas = []
        seen_email_ids = set()  
        ids = []

        for email in tqdm(emails, desc="Preparando documentos"):
            if email.id in seen_email_ids:
                continue  
            seen_email_ids.add(email.id)
            email_docs, email_metas, email_ids = self._prepare_email_documents(email)
            documents.extend(email_docs)
            metadatas.extend(email_metas)
            ids.extend(email_ids)

        if not documents:
            print("ERROR No hay documentos para indexar")
            return

        print(f"Total de chunks a indexar: {len(documents)}")

        print("Generando embeddings...")
        embeddings = self.embedder.encode(documents, batch_size=batch_size)

        print("Insertando en ChromaDB...")
        for i in tqdm(range(0, len(documents), batch_size), desc="Insertando batches"):
            end_idx = min(i + batch_size, len(documents))

            self.collection.add(
                embeddings=embeddings[i:end_idx].tolist(),
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx],
            )

        print(f"Total documentos en DB: {self.collection.count()}")

    def _prepare_email_documents(
        self, email: Any
    ) -> Tuple[List[str], List[Dict], List[str]]:
        documents = []
        metadatas = []
        ids = []

        base_metadata = {
            "email_id": email.id,
            "message_id": email.message_id,
            "from": email.from_address,
            "to": ", ".join(email.to_addresses) if email.to_addresses else "",
            "cc": ", ".join(email.cc_addresses) if email.cc_addresses else "",
            "bcc": ", ".join(email.bcc_addresses) if email.bcc_addresses else "",
            "date": email.date or "",
            "subject": email.subject or "",
            "thread_id": email.thread_id or "",
            "in_reply_to": email.in_reply_to or "",
            "references": email.references or "",
            "x_filename": email.x_filename or "",
        }

        subject = email.subject or ""
        body = email.body or ""

        if not body and len(subject) < 200:
            doc_id = f"{email.id}_chunk_0"
            doc_text = f"Subject: {subject}"
            doc_text = self._normalize_text(doc_text)

            documents.append(doc_text)

            metadata = base_metadata.copy()
            metadata.update(
                {
                    "chunk_type": "subject_body",
                    "chunk_index": 0,
                    "chunk_start": 0,
                    "chunk_end": 0,
                    "total_chunks": 1,
                }
            )
            metadatas.append(metadata)
            ids.append(doc_id)
            return documents, metadatas, ids

        if not body and not subject:
            return documents, metadatas, ids

        chunks = self.chunk_text(body, self.chunk_size, self.chunk_overlap)
        total_chunks = len(chunks) if chunks else 1

        for idx, (chunk_text, start_pos, end_pos) in enumerate(chunks):
            if subject:
                doc_text = f"Subject: {subject}\n\n{chunk_text}"
            else:
                doc_text = chunk_text
            doc_text = self._normalize_text(doc_text)
            doc_id = f"{email.id}_chunk_{idx}"
            documents.append(doc_text)

            metadata = base_metadata.copy()
            metadata.update(
                {
                    "chunk_type": "subject_body",
                    "chunk_index": idx,
                    "chunk_start": start_pos,
                    "chunk_end": end_pos,
                    "total_chunks": total_chunks,
                }
            )
            metadatas.append(metadata)
            ids.append(doc_id)

        return documents, metadatas, ids

    def search_with_reranking(
        self,
        query: str,
        n_results: int = 10,
        filter_metadata: Optional[Dict] = None,
        group_threads: bool = False,
    ) -> Dict[str, Any]:

        initial_n = n_results * 3
        query_embedding = self.embedder.encode_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=initial_n,
            where=filter_metadata,
            include=["metadatas", "documents", "distances"],
        )

        if not results["ids"][0]:
            return {"query": query, "results": [], "total": 0}

        reranked_results = self._rerank_results(results, query)

        processed = self._process_search_results(
            reranked_results, query, group_threads=group_threads
        )

        processed["results"] = processed["results"][:n_results]
        processed["total"] = len(processed["results"])

        return processed

    def _rerank_results(self, results: Dict, query: str) -> Dict:


        docs = results["documents"][0]
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        vectorizer = TfidfVectorizer(max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform(docs + [query])
            query_tfidf = tfidf_matrix[-1]
            doc_tfidf = tfidf_matrix[:-1]
            tfidf_scores = cosine_similarity(query_tfidf, doc_tfidf)[0]
        except:
            tfidf_scores = np.zeros(len(docs))

        query_terms = set(query.lower().split())
        bm25_scores = []
        for doc in docs:
            doc_terms = doc.lower().split()
            matches = sum(1 for term in query_terms if term in doc_terms)
            score = matches / len(query_terms) if query_terms else 0
            bm25_scores.append(score)

        max_dist = max(distances) if distances else 1
        norm_distances = [1 - (d / max_dist) for d in distances]

        combined_scores = []
        for i in range(len(docs)):
            score = (
                0.5 * norm_distances[i] + 0.3 * tfidf_scores[i] + 0.2 * bm25_scores[i]
            )
            combined_scores.append(score)

        sorted_indices = np.argsort(combined_scores)[::-1]

        return {
            "ids": [[ids[i] for i in sorted_indices]],
            "documents": [[docs[i] for i in sorted_indices]],
            "metadatas": [[metadatas[i] for i in sorted_indices]],
            "distances": [[1 - combined_scores[i] for i in sorted_indices]],
        }

    def _process_search_results(
        self, results: Dict, query: str, group_threads: bool = False
    ) -> Dict[str, Any]:

        if not results["ids"][0]:
            return {"query": query, "results": [], "total": 0}

        email_results = {}

        for idx, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][idx]
            document = results["documents"][0][idx]
            distance = results["distances"][0][idx]

            email_id = metadata["email_id"]

            if email_id not in email_results:
                email_results[email_id] = {
                    "email_id": email_id,
                    "message_id": metadata.get("message_id", ""),
                    "from": metadata.get("from", ""),
                    "to": metadata.get("to", ""),
                    "subject": metadata.get("subject", ""),
                    "date": metadata.get("date", ""),
                    "thread_id": metadata.get("thread_id", ""),
                    "chunks": [],
                    "best_distance": distance,
                }

            email_results[email_id]["chunks"].append(
                {
                    "text": document,
                    "chunk_type": metadata.get("chunk_type", "unknown"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "distance": distance,
                }
            )

            if distance < email_results[email_id]["best_distance"]:
                email_results[email_id]["best_distance"] = distance

        sorted_emails = sorted(email_results.values(), key=lambda x: x["best_distance"])

        if not group_threads:
            return {
                "query": query,
                "results": sorted_emails,
                "total": len(sorted_emails),
            }

        processed_threads = set()
        final_results = []

        for email in sorted_emails:
            thread_id = email.get("thread_id", "")

            if not thread_id:
                final_results.append(
                    {**email, "is_thread": False, "thread_emails": [email]}
                )
                continue

            if thread_id in processed_threads:
                continue

            thread_emails = [
                e for e in sorted_emails if e.get("thread_id") == thread_id
            ]

            if len(thread_emails) == 1:
                final_results.append(
                    {**email, "is_thread": False, "thread_emails": [email]}
                )
                processed_threads.add(thread_id)
                continue

            processed_threads.add(thread_id)

            thread_emails_sorted = sorted(
                thread_emails, key=lambda x: x.get("date", "")
            )

            best_email = min(thread_emails, key=lambda x: x["best_distance"])

            final_results.append(
                {
                    "email_id": best_email["email_id"],
                    "from": best_email["from"],
                    "to": best_email["to"],
                    "subject": best_email["subject"],
                    "date": best_email["date"],
                    "thread_id": thread_id,
                    "chunks": best_email["chunks"],
                    "best_distance": best_email["best_distance"],
                    "is_thread": True,
                    "thread_size": len(thread_emails),
                    "thread_emails": thread_emails_sorted,
                }
            )

        final_results.sort(key=lambda x: x["best_distance"])

        return {"query": query, "results": final_results, "total": len(final_results)}

    def delete_collection(self):
        try:
            self.client.delete_collection(self.collection.name)
            print(f"Colección '{self.collection.name}' eliminada")
        except Exception as e:
            print(f"Error eliminando colección: {e}")

    def delete_db(self):
        try:
            self.client.delete_collection(self.collection.name)
            print(f"Colección '{self.collection.name}' eliminada")
        except Exception as e:
            print(f"Error eliminando colección: {e}")

        try:
            if self.db_path.exists():
                shutil.rmtree(self.db_path)
                print(f"Base de datos eliminada: {self.db_path}")
        except Exception as e:
            print(f"Error eliminando directorio de BD: {e}")

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()

        if count > 0:
            try:
                sample = self.collection.get(
                    limit=min(count, 2000), include=["metadatas"]
                )
                metas = sample.get("metadatas", [])
            except Exception:
                sample = self.collection.get(limit=1000, include=["metadatas"])
                metas = sample.get("metadatas", [])

            chunk_types = {}
            for m in metas:
                ct = m.get("chunk_type", "unknown")
                chunk_types[ct] = chunk_types.get(ct, 0) + 1
        else:
            chunk_types = {}

        stats = {
            "total_chunks": count,
            "db_path": str(self.db_path),
            "embedding_dim": self.embedder.embedding_dim,
        }

        if self.contact_db:
            stats["contacts_count"] = self.contact_db.collection.count()

        return stats


def load_tests_from_tsv(test_path: str) -> List[Tuple[str, str]]:
    tests = []
    with open(test_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue

            if "\t" in line:
                expected_id, question = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    expected_id, question = parts
                else:
                    continue
            tests.append((expected_id.strip(), question.strip()))
    return tests


def get_email_ids_in_collection(db: EmailVectorDB) -> set:
    try:
        total = db.collection.count()
        sample = db.collection.get(limit=max(1000, total), include=["metadatas"])
    except Exception:
        sample = db.collection.get(limit=1000, include=["metadatas"])
    metas = sample.get("metadatas", [])
    ids = {m.get("email_id") for m in metas if m.get("email_id")}
    return ids


def run_retrieval_tester(
    db: EmailVectorDB, test_file: str, topk: int = 3, n_results: int = 10) -> Dict[str, Any]:
    tests = load_tests_from_tsv(test_file)
    if not tests:
        raise ValueError(f"No tests found in {test_file}")

    present_ids = get_email_ids_in_collection(db)

    results_per_query = []
    p_at_1 = []
    p_at_3 = []
    p_at_10 = []
    missing_in_db = 0

    for expected_id, question in tqdm(tests, desc="Ejecutando tests"):
        if expected_id not in present_ids:
            print(f"ID esperado no encontrado en DB: {expected_id}")
            missing_in_db += 1
            p_at_1.append(0)
            p_at_3.append(0)
            p_at_10.append(0)
            results_per_query.append(
                {
                    "expected": expected_id,
                    "question": question,
                    "found": False,
                    "rank": None,
                    "top_ids": [],
                }
            )
            continue

        search_res = db.search_with_reranking(
            question, n_results=n_results, group_threads=True
        )
        predicted_ids = [r["email_id"] for r in search_res.get("results", [])]

        rank = None
        if expected_id in predicted_ids:
            rank = predicted_ids.index(expected_id) + 1

        p1 = 1 if predicted_ids and predicted_ids[0] == expected_id else 0
        p_at_1.append(p1)

        p3 = 1 if expected_id in predicted_ids[:3] else 0
        p_at_3.append(p3)

        p10 = 1 if expected_id in predicted_ids[:10] else 0
        p_at_10.append(p10)
        

        results_per_query.append(
            {
                "expected": expected_id,
                "question": question,
                "found": rank is not None,
                "rank": rank,
                "top_ids": predicted_ids[: max(topk, 10)],
            }
        )

    metrics = {
        "n_queries": len(tests),
        "n_missing_in_db": missing_in_db,
        "precision_at_1": statistics.mean(p_at_1) if p_at_1 else 0.0,
    }
    

    metrics["precision_at_3"] = statistics.mean(p_at_3) if p_at_3 else 0.0
    metrics["precision_at_10"] = statistics.mean(p_at_10) if p_at_10 else 0.0

    summary = {"metrics": metrics, "per_query": results_per_query}

    return summary


def test_multiple_models(
    json_path: str,
    test_file: str,
    output_file: str,
    topk: int = 3,
    n_results: int = 10,
    device: str = "cpu",    
):
    import torch
    import gc

    models_to_test = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/LaBSE",
        "intfloat/multilingual-e5-base",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]

    print(f"\n{'='*80}")
    print(f"Cargando emails desde {json_path}")
    print(f"{'='*80}\n")

    with open(json_path, "r", encoding="utf-8") as f:
        emails_data = json.load(f)
    emails = [Email(**e) for e in emails_data]
    print(f"Total emails cargados: {len(emails)}")
    print(f"Dispositivo: {device.upper()}\n")

    all_results = {}

    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"PROBANDO MODELO: {model_name}")
        print(f"{'='*80}\n")

        db_path = (
            f"data/test_vectordb_{model_name.replace('/', '_').replace('-', '_')}"
        )

        db = None
        embedder = None
        start_time = time.perf_counter()

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            embedder = MultilingualEmbedder(model_name=model_name, device=device)
            db = EmailVectorDB(db_path=db_path, embedder=embedder)

            print(f"\nIndexando {len(emails)} emails...")
            db.index_emails(emails, batch_size=32)

            print(f"\nEjecutando tests desde {test_file}...")
            summary = run_retrieval_tester(
                db, test_file, topk=topk, n_results=n_results
            )

            elapsed_time = time.perf_counter() - start_time

            all_results[model_name] = summary["metrics"]
            all_results[model_name]["execution_time_sec"] = round(elapsed_time, 2)

            print(f"\n--- Resultados para {model_name} ---")
            m = summary["metrics"]
            print(f"Tiempo total: {elapsed_time:.2f} s")
            print(f"Queries: {m['n_queries']}")
            print(f"Missing expected IDs in DB: {m['n_missing_in_db']}")
            print(f"Precision@1: {m['precision_at_1']:.3f}")
            print(f"Precision@3: {m['precision_at_3']:.3f}")
            print(f"Precision@10: {m['precision_at_10']:.3f}")

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            all_results[model_name] = {
                "error": str(e),
                "execution_time_sec": round(elapsed_time, 2),
                "precision_at_1": 0.0,
            }
            all_results[model_name]["precision_at_3"] = 0.0
            all_results[model_name]["precision_at_10"] = 0.0

        finally:
            try:
                print(f"\nLimpiando base de datos temporal...")
                if db is not None:
                    db.delete_db()
            except Exception as e:
                print(f"Error al limpiar BD: {e}")

            del embedder
            del db
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\n{'='*80}")
    print("RESUMEN COMPARATIVO DE TODOS LOS MODELOS")
    print(f"{'='*80}\n")

    print(f"{'Modelo':<50} {'P@1':<8} {'P@3':<8} {'P@10':<8}")
    print(f"{'-'*80}")

    for model_name, metrics in all_results.items():
        if "error" not in metrics:
            print(
                f"{model_name:<50} {metrics['precision_at_1']:<8.3f} {metrics['precision_at_3']:<8.3f} {metrics['precision_at_10']:<8.3f}"
            )
        else:
            print(f"{model_name:<50} ERROR")

    valid_results = {k: v for k, v in all_results.items() if "error" not in v}

    if valid_results:
        best_model_p1 = max(
            valid_results.items(), key=lambda x: x[1].get("precision_at_1", 0)
        )
        best_model_p3 = max(
            valid_results.items(), key=lambda x: x[1].get("precision_at_3", 0)
        )
        best_model_p10 = max(
            valid_results.items(), key=lambda x: x[1].get("precision_at_10", 0)
        )

        print(f"\n{'='*80}")
        print(
            f"Mejor modelo por P@1: {best_model_p1[0]} (P@1: {best_model_p1[1].get('precision_at_1', 0):.3f})"
        )
        print(
            f"Mejor modelo por P@3: {best_model_p3[0]} (P@3: {best_model_p3[1].get('precision_at_3', 0):.3f})"
        )
        print(
            f"Mejor modelo por P@10: {best_model_p10[0]} (P@10: {best_model_p10[1].get('precision_at_10', 0):.3f})"
        )
        print(f"{'='*80}\n")
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"Resultados guardados en: {output_file}")

    return all_results


def test_embeddings_and_db(json_path: str, emails_db_path: str, contacts_db_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        emails_data = json.load(f)

    emails = [Email(**e) for e in emails_data]

    embedder = MultilingualEmbedder()
    contact_db = ContactVectorDB(db_path=contacts_db_path, embedder=embedder)
    db = EmailVectorDB(db_path=emails_db_path, embedder=embedder, contact_db=contact_db)

    print(f"Cargando emails desde {json_path} ({len(emails)} encontrados)")
    db.index_emails(emails)

    print("\n=== Test búsqueda por contacto ===")
    contacts = contact_db.search_contacts("sergio", n_results=10)
    for c in contacts:
        print(
            f"  {c['display_name']} ({c['email_address']}) - {len(c['email_ids'])} emails - Distancia: {c['distance']:.4f}"
        )

    results = db.search_with_reranking(
        "enseñame el email en el que me mandan la entrada del congreso try it del 2024",
        n_results=10,
        group_threads=True,
    )
    print(f"\n Resultados de búsqueda:")
    print(f"Query: '{results['query']}'")
    print(f"Emails encontrados: {results['total']}")
    print(f"Mostrando los top {len(results['results'])} resultados:\n")
    for idx, res in enumerate(results["results"]):
        if res.get("is_thread"):
            print(
                f"------------------------ ID: {res['email_id']} --------------------------"
            )
            print(
                f"[{idx+1}] Hilo ({res['thread_size']} emails) - Subject: {res['subject']} - From: {res['from']} - Date: {res['date']} - Distancia: {res['best_distance']:.4f}"
            )
        else:
            print(
                f"------------------------ ID: {res['email_id']} --------------------------"
            )
            print(
                f"[{idx+1}] Email - Subject: {res['subject']} - From: {res['from']} - Date: {res['date']} - Distancia: {res['best_distance']:.4f}"
            )

    stats = db.get_stats()
    print(f"\n Estadísticas de la DB:")
    print(f"  - Total chunks: {stats['total_chunks']}")
    print(f"  - Contactos: {stats.get('contacts_count', 0)}")

    return db
def make_embeddings_and_db_from_emails(
    emails: List[Email],
    emails_db_path: str = "data/vectordb",
    contacts_db_path: str = "data/vectordb_contacts",
):

    embedder = MultilingualEmbedder()
    contact_db = ContactVectorDB(db_path=contacts_db_path, embedder=embedder)
    db = EmailVectorDB(db_path=emails_db_path, embedder=embedder, contact_db=contact_db)

    print(f"Cargando {len(emails)} emails desde objetos Email")
    db.index_emails(emails)

    stats = db.get_stats()
    print(f"\n Estadísticas de la DB:")
    print(f"  - Total chunks: {stats['total_chunks']}")
    print(f"  - Contactos: {stats.get('contacts_count', 0)}")

    return db

#---LOCAL TESTING---
JSON_PATH = "data/processed/emails_processed.json"
EMAILS_DB_PATH = "data/emails_vectordb"
CONTACTS_DB_PATH = "data/emails_vectordb_contacts"

# # #---ENRON---
# JSON_PATH = "data/processed/enron_sample_10000+146+noise.json"
# EMAILS_DB_PATH = "data/test_vectordb"
# CONTACTS_DB_PATH = "data/test_vectordb_contacts"

TEST_PATH = "data/evaluate/test_preguntas_146.txt"
DATA_JSON_PATH = "data/processed/enron_sample_1000+146+noise.json"
    
@click.command()
@click.option("--json-path", default=JSON_PATH, help="Ruta al JSON de emails")
@click.option("--emails-db-path", default=EMAILS_DB_PATH, help="Ruta a la DB de emails")
@click.option("--contacts-db-path", default=CONTACTS_DB_PATH, help="Ruta a la DB de contactos")
@click.option("--test-models", is_flag=True, help="Ejecutar comparativa de modelos")
@click.option("--run-tester", is_flag=True, help="Ejecutar test de recuperación")
@click.option("--db-path", default=EMAILS_DB_PATH, help="Ruta a la DB")
@click.option("--test-file", default=TEST_PATH, help="Archivo de tests")
@click.option("--topk", default=3, help="Número de resultados topK para evaluar")
@click.option("--n-results", default=10, help="Número de resultados a recuperar por consulta")
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Dispositivo para el modelo de embeddings")
def main_cli(json_path, emails_db_path, contacts_db_path, test_models, run_tester, db_path, test_file, topk, n_results, device):
    if test_models:
        test_multiple_models(
            json_path=DATA_JSON_PATH,
            test_file=TEST_PATH,
            topk=topk,
            n_results=n_results,
            device=device,
            output_file=DATA_JSON_PATH.replace("enron_sample", "data/evaluate/model_comparison"),
        )
        pass
    elif run_tester:
        db = EmailVectorDB(db_path=db_path)
        summary = run_retrieval_tester(
            db, test_file, topk=topk, n_results=n_results)

        print("\n--- Tester summary ---")
        m = summary["metrics"]
        print(f"Queries: {m['n_queries']}")
        print(f"Missing expected IDs in DB: {m['n_missing_in_db']}")
        print(f"Precision@1: {m['precision_at_1']:.3f}")
        print(f"Precision@3: {m['precision_at_3']:.3f}")
        print(f"Precision@10: {m['precision_at_10']:.3f}")
        pass
    else:
        test_embeddings_and_db(
            json_path=json_path,
            emails_db_path=emails_db_path,
            contacts_db_path=contacts_db_path,
        )
        pass
if __name__ == "__main__":
    main_cli()
