class QuerySummarization:
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