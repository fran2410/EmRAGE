import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import traceback
from rag_engine import EmailRAGEngine, OllamaHandler, LLMEvaluator
from embeddings_system import EmailVectorDB, ContactVectorDB, MultilingualEmbedder


def load_test_data(questions_file: str, answers_file: str, num_questions: int) -> List[Tuple[str, str, str]]:

    questions = {}
    with open(questions_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if "\t" in line:
                email_id, question = line.split("\t", 1)
                questions[email_id.strip()] = question.strip()
    
    answers = {}
    with open(answers_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if "\t" in line:
                email_id, answer = line.split("\t", 1)
                answers[email_id.strip()] = answer.strip()
    
    test_data = []
    for email_id, question in questions.items():
        expected_answer = answers.get(email_id, "")
        if expected_answer:
            test_data.append((email_id, question, expected_answer))
            if num_questions and len(test_data) >= num_questions:
                break
    
    return test_data


def format_evaluation_report(
    query: str,
    expected_email_id: str,
    response: str,
    expected_answer: str,
    sources: List[Dict],
    evaluation: Dict,
    index: int
) -> str:
    
    report = f"""
{'='*100}
EVALUACIÓN #{index}
{'='*100}

PREGUNTA: {query}
EMAIL ESPERADO: {expected_email_id}

{'─'*100}
RESPUESTA DEL LLM PEQUEÑO:
{'─'*100}
{response}

{'─'*100}
RESPUESTA ESPERADA (GOLD STANDARD):
{'─'*100}
{expected_answer}

{'─'*100}
EVALUACIÓN DEL JUEZ
{'─'*100}

RESPUESTA CORRECTA: {evaluation['verdict']}

COMENTARIO DEL JUEZ:
{evaluation['comment']}

TIEMPO DE RESPUESTA: {evaluation['response_time']:.2f}s

FUENTES UTILIZADAS ({len(sources)}):
"""
    
    for idx, source in enumerate(sources, 1):
        if source.get("type") == "thread":
            report += f"   {idx}. [HILO] {source.get('subject')} "
            report += f"(Relevancia: {source.get('relevance_score')}, "
            report += f"{source.get('thread_size')} emails)\n"
        else:
            report += f"   {idx}. [{source.get('email_id')}] {source.get('subject')} "
            report += f"(Relevancia: {source.get('relevance_score')})\n"
    
    report += "\n"
    
    return report


class SimpleJudge:
    
    def __init__(self, judge_model: str = "llama3.2:3b"):
        self.ollama = OllamaHandler(model_name=judge_model)
        self.ollama.temperature = 0.1
        self.ollama.max_tokens = 1000
    
    def _create_judge_prompt(
        self,
        query: str,
        generated_response: str,
        expected_answer: str,
        expected_email_id: str
    ) -> str:
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Eres un juez imparcial que evalúa respuestas de un sistema RAG de emails.

Tu tarea es determinar si la respuesta generada es CORRECTA o INCORRECTA comparándola con la respuesta esperada.

Criterios:
1. La respuesta generada debe contener la misma información clave que la respuesta esperada
2. No necesita ser idéntica palabra por palabra, pero sí semánticamente equivalente
3. Debe mencionar los puntos principales del gold standard

FORMATO DE RESPUESTA OBLIGATORIO:
VEREDICTO: [CORRECTO / INCORRECTO]
COMENTARIO: [Tu análisis breve en 1-2 líneas]

IMPORTANTE: 
- Si la respuesta generada dice "no encontré" o similar, marca como INCORRECTO
- Si contiene información diferente o irrelevante, marca como INCORRECTO
- Solo marca CORRECTO si la información es equivalente al gold standard<|eot_id|><|start_header_id|>user<|end_header_id|>

PREGUNTA:
{query}

EMAIL ESPERADO: {expected_email_id}

RESPUESTA GENERADA:
{generated_response}

RESPUESTA ESPERADA (GOLD STANDARD):
{expected_answer}

¿Es correcta la respuesta generada?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    

    def _parse_judge_response(self, raw_response: str) -> Tuple[str, str]:
        lines = raw_response.strip().split('\n')
        
        verdict = "INCORRECTO"
        comment = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("VEREDICTO:"):
                verdict_text = line.replace("VEREDICTO:", "").strip()
                if "CORRECTO" in verdict_text.upper():
                    verdict = "CORRECTO" if "INCORRECTO" not in verdict_text.upper() else "INCORRECTO"
                else:
                    verdict = "INCORRECTO"
            elif line.startswith("COMENTARIO:"):
                comment = line.replace("COMENTARIO:", "").strip()
            elif comment:  
                comment += " " + line
        
        if not comment:
            comment = "Sin comentarios"
        
        return verdict, comment
    
    def evaluate(
        self,
        query: str,
        generated_response: str,
        expected_answer: str,
        expected_email_id: str,
        sources: List[Dict],
        response_time: float = 0.0
    ) -> Dict:        
        print(f"\n[JUEZ] Evaluando respuesta...")
        
        prompt = self._create_judge_prompt(
            query,
            generated_response,
            expected_answer,
            expected_email_id
        )
        
        raw_judge_response = ""
        for chunk in self.ollama.generate(prompt):
            raw_judge_response += chunk
            
        verdict, comment = self._parse_judge_response(raw_judge_response)
        
        result = {
            "verdict": verdict,
            "comment": comment,
            "response_time": response_time,
            "raw_judge_response": raw_judge_response
        }
        
        print(f"[JUEZ] Veredicto: {verdict}")
        
        return result


def run_evaluation(
    questions_file: str,
    answers_file: str,
    output_file: str = None,
    db_path: str = "data/emails_vectordb",
    contact_db_path: str = "data/emails_vectordb_contacts",
    rag_model: str = "llama3.2:1b",
    judge_model: str = "llama3.2:3b",
    num_questions: int = 10
):

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/evaluate/evaluation_report_{timestamp}.txt"
    
    print("\n" + "="*100)
    print("SISTEMA DE EVALUACIÓN RAG CON JUEZ")
    print("="*100)
    print(f"Archivo de preguntas: {questions_file}")
    print(f"Archivo de respuestas: {answers_file}")
    print(f"Archivo de salida: {output_file}")
    print(f"Modelo RAG: {rag_model}")
    print(f"Modelo Juez: {judge_model}")
    print("="*100 + "\n")
    
    test_data = load_test_data(questions_file, answers_file, num_questions)
    
    if not test_data:
        print("No hay datos de prueba para evaluar")
        return
    
    print(f"Cargados {len(test_data)} casos de prueba\n")
    
    print("Inicializando sistemas...")
    try:
        embedder = MultilingualEmbedder()
        contact_db = ContactVectorDB(db_path=contact_db_path, embedder=embedder)
        vector_db = EmailVectorDB(db_path=db_path, embedder=embedder)
        
        ollama_rag = OllamaHandler(model_name=rag_model)
        
        rag = EmailRAGEngine(
            vector_db=vector_db,
            contact_db=contact_db,
            ollama_handler=ollama_rag,
            use_contact_filter=True,
            default_language="es"
        )
        
        judge = SimpleJudge(judge_model=judge_model)
        
        print("Sistemas inicializados correctamente\n")
        
    except Exception as e:
        print(f"Error inicializando sistemas: {e}")
        traceback.print_exc()
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        
        f.write("="*100 + "\n")
        f.write("REPORTE DE EVALUACIÓN CON JUEZ\n")
        f.write("="*100 + "\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de preguntas: {len(test_data)}\n")
        f.write(f"Modelo RAG: {rag_model}\n")
        f.write(f"Modelo Juez: {judge_model}\n")
        f.write("="*100 + "\n\n")
        
        total_correct = 0
        total_time = 0
        
        for idx, (email_id, query, expected_answer) in enumerate(test_data, 1):
            print(f"\n[{idx}/{len(test_data)}] Procesando: {query[:60]}...")
            
            try:
                response_text = ""
                metadata = None
                start_time = datetime.now()
                
                for item in rag.query(query):
                    if isinstance(item, str):
                        response_text += item
                    elif isinstance(item, dict) and item.get("type") == "metadata":
                        metadata = item
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                if not metadata:
                    print(f"No se obtuvo metadata")
                    continue
                
                sources = metadata.get("sources", [])
                
                evaluation = judge.evaluate(
                    query=query,
                    generated_response=response_text,
                    expected_answer=expected_answer,
                    expected_email_id=email_id,
                    sources=sources,
                    response_time=response_time
                )
                
                if "CORRECTO" == evaluation["verdict"]:
                    total_correct += 1
                total_time += response_time
                
                report = format_evaluation_report(
                    query, email_id, response_text, expected_answer,
                    sources, evaluation, idx
                )
                f.write(report)
                f.flush()
                
                print(f"Evaluado - {evaluation['verdict']}")
                
            except Exception as e:
                error_msg = f"\nERROR en pregunta #{idx}: {e}\n"
                print(error_msg)
                f.write(error_msg)
                traceback.print_exc()
                continue
        
        n = len(test_data)
        summary = f"""
{'='*100}
RESUMEN FINAL
{'='*100}

ESTADÍSTICAS GENERALES:
   • Total de preguntas: {n}
   • Aciertos: {total_correct} ({(total_correct/n)*100:.1f}%)
   • Fallos: {n - total_correct} ({((n - total_correct)/n)*100:.1f}%)

TIEMPO:
   • Tiempo total: {total_time:.2f}s
   • Tiempo promedio por pregunta: {total_time/n:.2f}s

{'='*100}
"""
        
        f.write(summary)
        print("\n" + summary)
    
    print(f"\nEvaluación completada. Reporte guardado en: {output_file}\n")


if __name__ == "__main__":
    QUESTIONS_FILE = "data/evaluate/test_preguntas_10.txt"
    ANSWERS_FILE = "data/evaluate/test_respuestas_10.txt"
    OUTPUT_FILE = None  
    
    DB_PATH = "data/test_vectordb"
    CONTACT_DB_PATH = "data/test_vectordb_contacts"
    
    RAG_MODEL = "llama3.2:1b"
    JUDGE_MODEL = "llama3.2:3b"
    NUM_QUESTIONS = 10

    run_evaluation(
        questions_file=QUESTIONS_FILE,
        answers_file=ANSWERS_FILE,
        output_file=OUTPUT_FILE,
        db_path=DB_PATH,
        contact_db_path=CONTACT_DB_PATH,
        rag_model=RAG_MODEL,
        judge_model=JUDGE_MODEL,
        num_questions=NUM_QUESTIONS
    )