class QuerySummarization:
    name = "Query-Summarization"
    
    def __init__(self, model, top_k: int = 3):
        self.model = model
        self.top_k = top_k
    
    def answer(self, question: str, document: str) -> dict:
        """
        Summarize document chunks focusing on query-relevant parts.
        
        Strategy:
        1. Split document into chunks
        2. For each chunk, summarize focusing on parts relevant to the question
        3. Combine top_k summaries as context
        4. Ask final question against combined summaries
        
        Returns:
            dict with keys:
                - "answer": final answer string
                - "retrieved_chunks": list of chunk metadata
        """
        from src.utils import chunk_text_by_tokens
        
        # Split document into token-aware chunks
        chunks, metadatas = chunk_text_by_tokens(document, chunk_tokens=800)
        
        if not chunks:
            return {"answer": "", "retrieved_chunks": []}
        
        # Summarize each chunk with query focus
        summaries = []
        retrieved_chunks = []
        
        for chunk, metadata in zip(chunks, metadatas):
            prompt = f"Summarize this chunk focusing only on parts relevant to: '{question}'\n\n{chunk}"
            summary = self.model.ask(question, prompt)
            summaries.append(summary)
            
            retrieved_chunks.append({
                "text": chunk,
                "metadata": {
                    **metadata,
                    "summary": summary
                }
            })
        
        # Use top_k summaries as context
        context = "\n".join(summaries[:self.top_k])
        
        # Get final answer
        ans = self.model.ask(question, context)
        
        return {
            "answer": ans,
            "retrieved_chunks": retrieved_chunks[:self.top_k]
        }