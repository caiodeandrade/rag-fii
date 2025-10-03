import json
import time
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from datasets import Dataset

from ragas import evaluate

try:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision
    )
except ImportError:
    try:
        from ragas.metrics import (
            Faithfulness as faithfulness,
            AnswerRelevancy as answer_relevancy,
            ContextPrecision as context_precision
        )
        faithfulness = faithfulness()
        answer_relevancy = answer_relevancy()
        context_precision = context_precision()
    except ImportError:
        raise ImportError("Erro ao importar métricas do RAGAS")

from domain.services.logging_service import RAGLogger

class RAGEvaluationService:
    """
    ATUALIZADO: RAGAS usa GPT-4o-mini por padrão para avaliações
    Configuração via environment variable
    """
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.logger = RAGLogger()
        
        # Configurar OpenAI - RAGAS usará GPT-4o-mini automaticamente
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # RAGAS 0.3+ usa modelos mais novos por padrão
        # Você pode configurar o modelo específico se quiser:
        # os.environ["RAGAS_LLM_MODEL"] = "gpt-4o-mini"
        
        self.metrics = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision
        }
    
    def evaluate_single_question(self, question: str, rag_service, llm_service) -> Dict[str, Any]:
        """
        Avalia uma única pergunta e retorna resultado JSON
        """
        start_time = time.time()
        
        try:
            self.logger.logger.info(f"[RAGAS] Avaliando pergunta com GPT-4o-mini: {question[:50]}...")
            
            # 1. Coletar resposta RAG com contexts
            rag_data = self._get_rag_data(rag_service, question)
            
            # 2. Coletar resposta LLM
            llm_answer = llm_service.answer_question(question)
            
            # 3. Avaliar RAG com RAGAS (usa GPT-4o-mini internamente)
            rag_scores = self._evaluate_rag_response(question, rag_data)
            
            # 4. Avaliar LLM (apenas answer_relevancy)
            llm_scores = self._evaluate_llm_response(question, llm_answer)
            
            processing_time = time.time() - start_time
            
            # 5. Compilar resultado
            result = {
                "evaluation_summary": {
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": round(processing_time, 3),
                    "question_length": len(question),
                    "evaluation_model": "gpt-4o-mini"  # RAGAS usa este modelo
                },
                "rag_evaluation": {
                    "scores": rag_scores,
                    "interpretation": self._interpret_rag_scores(rag_scores)
                },
                "llm_evaluation": {
                    "scores": llm_scores,
                    "interpretation": self._interpret_llm_scores(llm_scores)
                },
                "comparison": self._compare_rag_vs_llm(rag_scores, llm_scores),
                "recommendation": self._generate_recommendation(rag_scores, llm_scores)
            }
            
            self.logger.logger.info(f"[RAGAS] Avaliação concluída em {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.log_error("SingleQuestionEvaluationError", str(e), question[:50])
            return {
                "error": str(e),
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_rag_data(self, rag_service, question: str) -> Dict[str, Any]:
        """Coleta dados RAG (resposta + contexts)"""
        try:
            answer = rag_service.answer_question_simple(question)
            
            contexts = []
            if hasattr(rag_service, 'retriever'):
                docs = rag_service.retriever.get_relevant_documents(question)
                contexts = [doc.page_content for doc in docs]
            elif hasattr(rag_service, 'qa_chain') and hasattr(rag_service.qa_chain, 'retriever'):
                docs = rag_service.qa_chain.retriever.get_relevant_documents(question)
                contexts = [doc.page_content for doc in docs]
            
            return {
                'answer': answer,
                'contexts': contexts
            }
            
        except Exception as e:
            self.logger.log_error("RAGDataCollectionError", str(e))
            return {'answer': f"Error: {str(e)}", 'contexts': []}
    
    def _evaluate_rag_response(self, question: str, rag_data: Dict) -> Dict[str, float]:
        """Avalia resposta RAG usando RAGAS (com GPT-4o-mini)"""
        try:
            dataset_dict = {
                'question': [question],
                'answer': [rag_data['answer']],
                'contexts': [rag_data['contexts']],
                'ground_truth': ['']
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # RAGAS usa GPT-4o-mini automaticamente para avaliação
            result = evaluate(dataset, metrics=list(self.metrics.values()))
            
            scores = {}
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                for metric_name in self.metrics.keys():
                    if metric_name in df.columns:
                        scores[metric_name] = float(df[metric_name].iloc[0])
            
            return scores
            
        except Exception as e:
            self.logger.log_error("RAGEvaluationError", str(e))
            return {metric: 0.0 for metric in self.metrics.keys()}
    
    def _evaluate_llm_response(self, question: str, llm_answer: str) -> Dict[str, float]:
        """Avalia resposta LLM (apenas answer_relevancy)"""
        try:
            dataset_dict = {
                'question': [question],
                'answer': [llm_answer],
                'contexts': [[]],
                'ground_truth': ['']
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            result = evaluate(dataset, metrics=[self.metrics['answer_relevancy']])
            
            score = 0.0
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                if 'answer_relevancy' in df.columns:
                    score = float(df['answer_relevancy'].iloc[0])
            
            return {'answer_relevancy': score}
            
        except Exception as e:
            self.logger.log_error("LLMEvaluationError", str(e))
            return {'answer_relevancy': 0.0}
    
    def _interpret_rag_scores(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Interpreta scores RAG em linguagem natural"""
        interpretation = {}
        
        for metric, score in scores.items():
            if metric == 'faithfulness':
                if score >= 0.8:
                    interpretation[metric] = "Excelente - Resposta muito fiel aos documentos"
                elif score >= 0.6:
                    interpretation[metric] = "Boa - Resposta majoritariamente baseada nos documentos"
                elif score >= 0.4:
                    interpretation[metric] = "Regular - Alguma base nos documentos, mas com adições"
                else:
                    interpretation[metric] = "Ruim - Resposta pouco baseada nos documentos"
                    
            elif metric == 'answer_relevancy':
                if score >= 0.8:
                    interpretation[metric] = "Excelente - Resposta muito relevante à pergunta"
                elif score >= 0.6:
                    interpretation[metric] = "Boa - Resposta relevante à pergunta"
                elif score >= 0.4:
                    interpretation[metric] = "Regular - Resposta parcialmente relevante"
                else:
                    interpretation[metric] = "Ruim - Resposta pouco relevante à pergunta"
                    
            elif metric == 'context_precision':
                if score >= 0.8:
                    interpretation[metric] = "Excelente - Documentos recuperados muito relevantes"
                elif score >= 0.6:
                    interpretation[metric] = "Boa - Documentos recuperados relevantes"
                elif score >= 0.4:
                    interpretation[metric] = "Regular - Alguns documentos relevantes"
                else:
                    interpretation[metric] = "Ruim - Documentos recuperados pouco relevantes"
        
        return interpretation
    
    def _interpret_llm_scores(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Interpreta scores LLM"""
        interpretation = {}
        
        score = scores.get('answer_relevancy', 0.0)
        if score >= 0.8:
            interpretation['answer_relevancy'] = "Excelente - LLM respondeu de forma muito relevante"
        elif score >= 0.6:
            interpretation['answer_relevancy'] = "Boa - LLM respondeu de forma relevante"
        elif score >= 0.4:
            interpretation['answer_relevancy'] = "Regular - LLM respondeu parcialmente relevante"
        else:
            interpretation['answer_relevancy'] = "Ruim - LLM respondeu de forma pouco relevante"
        
        return interpretation
    
    def _compare_rag_vs_llm(self, rag_scores: Dict, llm_scores: Dict) -> Dict[str, str]:
        """Compara RAG vs LLM"""
        comparison = {}
        
        rag_relevancy = rag_scores.get('answer_relevancy', 0.0)
        llm_relevancy = llm_scores.get('answer_relevancy', 0.0)
        
        if rag_relevancy > llm_relevancy:
            diff = ((rag_relevancy - llm_relevancy) / llm_relevancy * 100) if llm_relevancy > 0 else 100
            comparison['answer_relevancy'] = f"RAG vence por {diff:.1f}% (RAG: {rag_relevancy:.3f} vs LLM: {llm_relevancy:.3f})"
        elif llm_relevancy > rag_relevancy:
            diff = ((llm_relevancy - rag_relevancy) / rag_relevancy * 100) if rag_relevancy > 0 else 100
            comparison['answer_relevancy'] = f"LLM vence por {diff:.1f}% (LLM: {llm_relevancy:.3f} vs RAG: {rag_relevancy:.3f})"
        else:
            comparison['answer_relevancy'] = f"Empate (ambos: {rag_relevancy:.3f})"
        
        faithfulness = rag_scores.get('faithfulness', 0.0)
        context_precision = rag_scores.get('context_precision', 0.0)
        
        comparison['rag_exclusive'] = {
            'faithfulness': f"RAG: {faithfulness:.3f} (LLM não aplicável - sem documentos)",
            'context_precision': f"RAG: {context_precision:.3f} (LLM não aplicável - sem recuperação)"
        }
        
        return comparison
    
    def _generate_recommendation(self, rag_scores: Dict, llm_scores: Dict) -> str:
        """Gera recomendação baseada nos scores"""
        rag_relevancy = rag_scores.get('answer_relevancy', 0.0)
        llm_relevancy = llm_scores.get('answer_relevancy', 0.0)
        faithfulness = rag_scores.get('faithfulness', 0.0)
        context_precision = rag_scores.get('context_precision', 0.0)
        
        rag_avg = (rag_relevancy + faithfulness + context_precision) / 3
        
        if rag_avg >= 0.7 and rag_relevancy >= llm_relevancy:
            return "RECOMENDADO: Use RAG - Alta qualidade e baseado em documentos confiáveis"
        elif rag_relevancy >= 0.6 and faithfulness >= 0.6:
            return "RECOMENDADO: Use RAG - Boa qualidade e fiel aos documentos"
        elif llm_relevancy > rag_relevancy and llm_relevancy >= 0.7:
            return "CUIDADO: LLM mais relevante, mas pode alucinar. Considere melhorar recuperação RAG"
        elif faithfulness < 0.4:
            return "ATENÇÃO: RAG com baixa fidelidade. Verifique qualidade dos documentos"
        else:
            return "ANÁLISE: Ambos com performance similar. Prefira RAG para maior confiabilidade"