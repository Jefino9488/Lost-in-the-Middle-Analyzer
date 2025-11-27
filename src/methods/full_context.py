from src.utils import count_tokens, truncate_text_center

class FullContext:
    name = "FullContext"
    
    # Maximum context tokens to prevent API limit breaches
    # Set conservatively to work with most API limits (e.g., Gemini Flash has 1M context)
    MAX_CONTEXT_TOKENS = 100000
    
    def __init__(self, model):
        self.model = model
    
    def answer(self, question: str, document: str) -> dict:
        """
        Send full document as context, with token budget safety guards.
        
        If document exceeds MAX_CONTEXT_TOKENS, uses center-preserving truncation
        (keeps first 40%, last 40%, drops middle 20%).
        
        Returns:
            dict with keys:
                - "answer": model's answer
                - "retrieved_chunks": list with the document (or truncated version)
        """
        # Check token count
        doc_tokens = count_tokens(document)
        was_truncated = False
        context_to_use = document
        
        if doc_tokens > self.MAX_CONTEXT_TOKENS:
            # Truncate with center-preserving strategy
            context_to_use, was_truncated = truncate_text_center(
                document, 
                self.MAX_CONTEXT_TOKENS
            )
            # Note: In production, you might want to log this warning
            # For now, we'll just track it in metadata
        
        # Get answer from model
        ans = self.model.ask(question, context_to_use)
        
        # Return result with metadata
        retrieved_chunks = [{
            "text": context_to_use,
            "metadata": {
                "was_truncated": was_truncated,
                "original_tokens": doc_tokens,
                "final_tokens": count_tokens(context_to_use) if was_truncated else doc_tokens
            }
        }]
        
        return {
            "answer": ans, 
            "retrieved_chunks": retrieved_chunks
        }
