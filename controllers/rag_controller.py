from flask import Blueprint, request, jsonify

rag_bp = Blueprint('rag_bp', __name__)

def create_routes(rag_service, llm_service):

    @rag_bp.route("/query", methods=["POST"])
    def query_rag():
        data = request.json or {}
        question = data.get("question")
        if not question:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        answer = rag_service.answer_question(question)
        return jsonify({"answer": answer})

    @rag_bp.route("/query_llm", methods=["POST"])
    def query_llm():
        data = request.json or {}
        question = data.get("question")
        if not question:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        answer = llm_service.answer_question(question)
        return jsonify({"answer": answer})