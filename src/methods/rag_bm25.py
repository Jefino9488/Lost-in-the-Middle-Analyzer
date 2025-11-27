from rank_bm25 import BM25Okapi
import re

class RAGBM25:
    name = "RAG-BM25"
    def __init__(self, model, top_k: int = 3, chunk_words: int = 200):
        self.model = model
        self.top_k = top_k
        self.chunk_words = chunk_words

    def _split(self, text, n=None):
        # naive chunker by words
        words = re.split(r"\s+", text)
        n = n or self.chunk_words
        chunks = [" ".join(words[i:i+n]) for i in range(0, len(words), n)]
        return chunks

    def answer(self, question: str, document: str) -> dict:
        chunks = self._split(document)
        tokenized = [c.split() for c in chunks]
        if not tokenized:
            return {"answer": "", "retrieved_chunks": []}
        bm25 = BM25Okapi(tokenized)
        q_tokens = question.split()
        scores = bm25.get_scores(q_tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]
        top_chunks = [chunks[i] for i in top_idx]
        context = "\n\n".join(top_chunks)
        ans = self.model.ask(question, context)
        return {"answer": ans, "retrieved_chunks": top_chunks}
