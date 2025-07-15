from langchain.chains import RetrievalQA

class RAGService:
    def __init__(self, llm, retriever):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever
        )

    def answer_question(self, question):
        return self.qa_chain.run(question)