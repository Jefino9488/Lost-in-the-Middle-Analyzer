from rank_bm25 import BM25Okapi

class ReRanking:
    name = "Re-Ranking"
    def __init__(self, model, top_k: int = 3):
        self.model = model
        self.top_k = top_k
    def answer(self, question: str, document: str) -> str:
        # BM25 retrieve
        chunks = document.split(". ")
        tokenized = [c.split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(question.split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_chunks = [chunks[i] for i, _ in ranked[:self.top_k]]
        # Ask model on concatenated top chunks
        context = "\n".join(top_chunks)
        return self.model.ask(question, context)