import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class FaissRepository:
    def __init__(self, faiss_path, pdfs_dir, openai_api_key):
        self.faiss_path = faiss_path
        self.pdfs_dir = pdfs_dir
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def load_or_create_index(self):
        if os.path.exists(self.faiss_path):
            print("[FAISS] Usou índice existente")
            return FAISS.load_local(
                self.faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

        print("[FAISS] Criou novo índice")
        pdf_files = glob.glob(os.path.join(self.pdfs_dir, "*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("Nenhum PDF encontrado.")

        docs = []
        for pdf_file in pdf_files:
            docs.extend(PyPDFLoader(pdf_file).load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        split_docs = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        vectorstore.save_local(self.faiss_path)
        return vectorstore