class SlidingWindow:
    name = "SlidingWindow"
    def __init__(self, model, window_size: int = 800, num_windows: int = 3):
        self.model = model
        self.window_size = window_size
        self.num_windows = num_windows

    def answer(self, question: str, document: str) -> dict:
        from src.utils import chunk_text_by_tokens
        
        # Create token-aware overlapping windows
        chunks, metadatas = chunk_text_by_tokens(
            document,
            chunk_tokens=self.window_size,
            overlap_tokens=self.window_size // 4  # 25% overlap for continuity
        )
        
        if not chunks:
            return {"answer": "", "retrieved_chunks": []}
        
        # Limit to num_windows
        chunks = chunks[:self.num_windows]
        metadatas = metadatas[:self.num_windows]
        
        candidates = []
        retrieved = []
        
        for chunk, metadata in zip(chunks, metadatas):
            if not chunk:
                continue
            
            retrieved.append({
                "text": chunk,
                "metadata": metadata
            })
            candidates.append(self.model.ask(question, chunk))
        
        # First try to return any candidate that looks valid
        ans = ""
        for c in candidates:
            if c and "couldn't find" not in c.lower():
                ans = c
                break
        if not ans and candidates:
            ans = candidates[0]
            
        return {"answer": ans, "retrieved_chunks": retrieved}
