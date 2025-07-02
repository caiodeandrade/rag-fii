import os
from flask import Flask
from dotenv import load_dotenv
from langchain_community.llms import OpenAI

from data.repositories.local.faiss_repository import FaissRepository
from domain.services.rag_service import RAGService
from domain.services.llm_service import LLMService
from controllers.rag_controller import rag_bp, create_routes

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Config
PDFS_DIR = "assets"
FAISS_PATH = "pdf_faiss_index"

# App
app = Flask(__name__)

# Infra
faiss_repo = FaissRepository(FAISS_PATH, PDFS_DIR, openai_api_key)
vectorstore = faiss_repo.load_or_create_index()

# Domain
llm = OpenAI(openai_api_key=openai_api_key)
rag_service = RAGService(llm, vectorstore.as_retriever())
llm_service = LLMService(llm)

# Controllers
create_routes(rag_service, llm_service)
app.register_blueprint(rag_bp)

if __name__ == "__main__":
    app.run(debug=True)