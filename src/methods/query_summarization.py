    def answer(self, question: str, document: str) -> dict:
        # Summarize query first? Or summarize doc?
        # The name implies summarizing the query or using query to summarize doc.
        # Let's assume standard RAG but with query expansion/summarization.
        # Actually, let's look at the implementation if it exists, otherwise implement simple RAG.
        # The file content was not shown fully, but I will assume it's similar to RAG.
        
        # For now, let's just wrap the existing logic if I can see it.
        # Since I can't see the file content in history, I'll read it first to be safe.
        pass
    name = "Query-Summarization"
    def __init__(self, model, top_k: int = 3):
        self.model = model
        self.top_k = top_k
    def answer(self, question: str, document: str) -> str:
        chunks = [document[i:i+800] for i in range(0, len(document), 800)]
        summaries = []
        for ch in chunks:
            prompt = f"Summarize this chunk focusing only on parts relevant to: '{question}'\n\n{ch}"
            summaries.append(self.model.ask(question, prompt))
        context = "\n".join(summaries[:self.top_k])
        return self.model.ask(question, context)