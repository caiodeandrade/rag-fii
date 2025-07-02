class LLMService:
    def __init__(self, llm):
        self.llm = llm

    def answer_question(self, question):
        return self.llm.invoke(question)