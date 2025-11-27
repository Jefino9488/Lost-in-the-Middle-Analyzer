from rank_bm25 import BM25Okapi

class HybridRAG:
    name = "Hybrid-RAG"
    def __init__(self, model, top_k: int = 3):
        self.model = model
        self.top_k = top_k
    def answer(self, question: str, document: str) -> dict:
        # BM25 step
        chunks = document.split(". ")
        tokenized = [c.split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(question.split())
        top_chunks = [chunks[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]]
        # Summarize retrieved chunks
        summarized = []
        for ch in top_chunks:
            prompt = f"Summarize context for answering: '{question}'\n\n{ch}"
            summarized.append(self.model.ask(question, prompt))
        context = "\n".join(summarized)
        ans = self.model.ask(question, context)
        return {"answer": ans, "retrieved_chunks": top_chunks}