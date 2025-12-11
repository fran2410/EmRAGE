import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import traceback
from rag_engine import EmailRAGEngine, OllamaHandler, LLMEvaluator
from embeddings_system import EmailVectorDB, ContactVectorDB, MultilingualEmbedder


def load_test_questions(file_path: str, num_questions: int) -> List[Tuple[str, str]]:
    tests = []
    with open(file_path, "r", encoding="utf-8") as f:
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
            if num_questions and len(tests) >= num_questions:
                break
    return tests


def format_evaluation_report(
    query: str,
    expected_email_id: str,
    response: str,
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

RESPUESTA GENERADA:
{response}

{'─'*100}
EVALUACIÓN (Modelo: llama3.2:3b)
{'─'*100}

 PUNTUACIONES (1-10):
   • Precisión:        {evaluation['scores']['precisión']}/10
   • Relevancia:       {evaluation['scores']['relevancia']}/10
   • Uso de Fuentes:   {evaluation['scores']['uso_de_fuentes']}/10
   • Coherencia:       {evaluation['scores']['coherencia']}/10
   • Completitud:      {evaluation['scores']['completitud']}/10
   • Recuperación:     {evaluation['scores']['recuperación']}/10
   
    PROMEDIO:         {evaluation['average_score']}/10

  TIEMPO DE RESPUESTA: {evaluation['response_time']:.2f}s

  EMAIL CORRECTO RECUPERADO: {' SÍ' if evaluation['email_found'] else '✗ NO'}

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
    
    report += f"""
 COMENTARIOS DEL EVALUADOR:
{evaluation['scores']['comentarios']}

"""
    
    return report


def run_evaluation(
    questions_file: str,
    output_file: str = None,
    db_path: str = "data/emails_vectordb",
    contact_db_path: str = "data/emails_vectordb_contacts",
    rag_model: str = "llama3.2:1b",
    evaluator_model: str = "llama3.2:3b",
    num_questions: int = 146
):

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/data/evaluate/evaluation_report_{timestamp}.txt"
    
    print("\n" + "="*100)
    print("SISTEMA DE EVALUACIÓN RAG")
    print("="*100)
    print(f" Archivo de preguntas: {questions_file}")
    print(f" Archivo de salida: {output_file}")
    print(f" Modelo RAG: {rag_model}")
    print(f" Modelo Evaluador: {evaluator_model}")
    print("="*100 + "\n")
    
    questions = load_test_questions(questions_file, num_questions)
    
    if not questions:
        print(" No hay preguntas para evaluar")
        return
    
    print("  Inicializando sistemas...")
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
        
        evaluator = LLMEvaluator(evaluator_model=evaluator_model)
        
        print(" Sistemas inicializados correctamente\n")
        
    except Exception as e:
        print(f" Error inicializando sistemas: {e}")
        traceback.print_exc()
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        
        f.write("="*100 + "\n")
        f.write("REPORTE DE EVALUACIÓN DEL SISTEMA RAG\n")
        f.write("="*100 + "\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de preguntas: {len(questions)}\n")
        f.write(f"Modelo RAG: {rag_model}\n")
        f.write(f"Modelo Evaluador: {evaluator_model}\n")
        f.write("="*100 + "\n\n")
        
        total_scores = {
            "precisión": 0,
            "relevancia": 0,
            "uso_de_fuentes": 0,
            "coherencia": 0,
            "completitud": 0,
            "recuperación": 0
        }
        total_time = 0
        emails_found = 0
        
        for idx, (email_id, query) in enumerate(questions, 1):
            print(f"\n[{idx}/{len(questions)}] Procesando: {query[:60]}...")
            
            try:
                response_text = ""
                metadata = None
                context_used = ""
                start_time = datetime.now()
                
                for item in rag.query(query):
                    if isinstance(item, str):
                        response_text += item
                    elif isinstance(item, dict) and item.get("type") == "metadata":
                        metadata = item
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                if not metadata:
                    print(f"     No se obtuvo metadata")
                    continue
                
                sources = metadata.get("sources", [])
                
                evaluation = evaluator.evaluate(
                    query=query,
                    context="",  
                    response=response_text,
                    sources=sources,
                    expected_email_id=email_id,
                    response_time=response_time
                )
                
                for key in total_scores:
                    total_scores[key] += evaluation["scores"][key]
                total_time += response_time
                if evaluation["email_found"]:
                    emails_found += 1
                
                report = format_evaluation_report(
                    query, email_id, response_text, sources, evaluation, idx
                )
                f.write(report)
                f.flush()  
                
                print(f"    Evaluado - Promedio: {evaluation['average_score']}/10")
                
            except Exception as e:
                error_msg = f"\n ERROR en pregunta #{idx}: {e}\n"
                print(error_msg)
                f.write(error_msg)
                traceback.print_exc()
                continue
        
        n = len(questions)
        avg_scores = {k: v/n for k, v in total_scores.items()}
        overall_avg = sum(avg_scores.values()) / len(avg_scores)
        
        summary = f"""
{'='*100}
RESUMEN GENERAL
{'='*100}

 PROMEDIOS GENERALES:
   • Precisión:        {avg_scores['precisión']:.2f}/10
   • Relevancia:       {avg_scores['relevancia']:.2f}/10
   • Uso de Fuentes:   {avg_scores['uso_de_fuentes']:.2f}/10
   • Coherencia:       {avg_scores['coherencia']:.2f}/10
   • Completitud:      {avg_scores['completitud']:.2f}/10
   • Recuperación:     {avg_scores['recuperación']:.2f}/10
   
    PROMEDIO TOTAL:   {overall_avg:.2f}/10

  TIEMPO PROMEDIO DE RESPUESTA: {total_time/n:.2f}s

  EMAILS CORRECTOS RECUPERADOS: {emails_found}/{n} ({(emails_found/n)*100:.1f}%)

{'='*100}
"""
        
        f.write(summary)
        print("\n" + summary)
    
    print(f"\n Evaluación completada. Reporte guardado en: {output_file}\n")


if __name__ == "__main__":
    QUESTIONS_FILE = "data/evaluate/test_preguntas_146.txt"  
    OUTPUT_FILE = None  

    DB_PATH = "data/test_vectordb"
    CONTACT_DB_PATH = "data/test_vectordb_contacts"
    
    RAG_MODEL = "llama3.2:1b"
    EVALUATOR_MODEL = "llama3.2:3b"
    NUM_QUESTIONS = 10  

    run_evaluation(
        questions_file=QUESTIONS_FILE,
        output_file=OUTPUT_FILE,
        db_path=DB_PATH,
        contact_db_path=CONTACT_DB_PATH,
        rag_model=RAG_MODEL,
        evaluator_model=EVALUATOR_MODEL,
        num_questions=NUM_QUESTIONS
    )