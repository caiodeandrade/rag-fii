import os
import glob
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI   # ← já estava no seu código

# ------------------------------------------------------------------
# Configuração e objetos globais
# ------------------------------------------------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

pdfs_dir   = "assets"
faiss_path = "pdf_faiss_index"

# --------- NOVO: instância única do LLM (sem RAG) ----------
llm = OpenAI(openai_api_key=openai_api_key)
# -----------------------------------------------------------

def load_or_create_faiss_index():
    if os.path.exists(faiss_path):
        print("Usou índice existente")
        return FAISS.load_local(
            faiss_path,
            OpenAIEmbeddings(openai_api_key=openai_api_key),
            allow_dangerous_deserialization=True,
        )

    print("Criou novo índice FAISS")
    pdf_files = glob.glob(os.path.join(pdfs_dir, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("Nenhum PDF encontrado em 'assets/'.")

    docs = []
    for pdf_file in pdf_files:
        docs.extend(PyPDFLoader(pdf_file).load())

    splitter   = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    split_docs = splitter.split_documents(docs)

    embeddings  = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(faiss_path)
    return vectorstore


vectorstore = load_or_create_faiss_index()

# Cadeia RAG (já existente)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  # reaproveitamos a mesma instância
    retriever=vectorstore.as_retriever()
)

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

# 1. Endpoint RAG (como você já tinha)
@app.route("/query", methods=["POST"])
def query_rag():
    data      = request.json or {}
    question  = data.get("question")
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    answer = qa_chain.run(question)
    return jsonify({"answer": answer})


# 2. Endpoint **SEM** RAG — usa apenas o LLM
@app.route("/query_llm", methods=["POST"])   # ← nova rota
def query_llm():
    data      = request.json or {}
    question  = data.get("question")
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    # Chamada direta ao modelo (sem recuperação de contexto)
    answer = llm.invoke(question)            # ou llm.predict(question) em versões mais antigas
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)