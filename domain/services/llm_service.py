import time
from domain.services.logging_service import RAGLogger

class LLMService:
    """
    NOTA: Esta classe já é compatível com ChatOpenAI
    O método invoke() funciona tanto com OpenAI quanto ChatOpenAI
    """
    def __init__(self, llm):
        self.llm = llm
        self.logger = RAGLogger()
    
    def answer_question(self, question):
        start_time = time.time()
        
        try:
            # invoke() funciona com chat models automaticamente
            # ChatOpenAI converte strings em mensagens internamente
            response = self.llm.invoke(question)
            
            # ChatOpenAI retorna um objeto AIMessage, extrair conteúdo
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            processing_time = time.time() - start_time
            
            self.logger.log_query_execution(
                question=question,
                method="llm_only",
                response_len=len(answer),
                time_ms=processing_time
            )
            
            return answer
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.log_error("LLMQueryError", str(e), f"Question: {question[:100]}...")
            raise
