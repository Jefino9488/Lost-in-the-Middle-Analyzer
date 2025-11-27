from collections import Counter
from src.utils import normalize_for_em, bleu_score, count_tokens

class MapReduce:
    name = "Map-Reduce"
    
    def __init__(self, model, chunk_size: int = 800, top_k: int = 3):
        self.model = model
        self.chunk_size = chunk_size
        self.top_k = top_k

    def _chunk_text(self, text: str):
        """
        Token-aware chunking that respects token boundaries.
        Yields chunks as (chunk_text, start_pos, end_pos).
        """
        from src.utils import chunk_text_by_tokens
        
        chunks, metadatas = chunk_text_by_tokens(
            text,
            chunk_tokens=self.chunk_size,
            overlap_tokens=50  # Small overlap for context continuity
        )
        
        return chunks, metadatas

    def _aggregate_candidates(self, candidates: list[str]) -> str:
        """
        Aggregate candidate answers using majority voting.
        On tie, use BLEU score tie-breaker (highest average BLEU vs others wins).
        """
        if not candidates:
            return ""
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Normalize all candidates
        normalized_candidates = [normalize_for_em(c) for c in candidates]
        
        # Count votes for each normalized answer
        vote_counts = Counter(normalized_candidates)
        
        # Get the most common answers
        most_common = vote_counts.most_common()
        
        # Check if there's a clear winner
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            # Clear winner, return the original (non-normalized) version
            winner_normalized = most_common[0][0]
            # Find first occurrence of this normalized answer in original candidates
            for i, norm_cand in enumerate(normalized_candidates):
                if norm_cand == winner_normalized:
                    return candidates[i]
        
        # Tie-breaker: compute average BLEU score for each tied candidate
        max_votes = most_common[0][1]
        tied_candidates = [cand for cand, count in most_common if count == max_votes]
        
        best_candidate = None
        best_avg_bleu = -1.0
        
        for tied_norm in tied_candidates:
            # Find the original candidate
            original_candidate = None
            for i, norm_cand in enumerate(normalized_candidates):
                if norm_cand == tied_norm:
                    original_candidate = candidates[i]
                    break
            
            if original_candidate is None:
                continue
            
            # Compute average BLEU vs all other candidates
            bleu_scores = []
            for other_cand in candidates:
                if normalize_for_em(other_cand) != tied_norm:
                    bleu_scores.append(bleu_score(original_candidate, other_cand))
            
            avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
            
            if avg_bleu > best_avg_bleu:
                best_avg_bleu = avg_bleu
                best_candidate = original_candidate
        
        return best_candidate if best_candidate else candidates[0]

    def answer(self, question: str, document: str) -> dict:
        """
        Process document using map-reduce strategy:
        1. Split document into chunks
        2. Ask model for each chunk (map phase)
        3. Aggregate answers using voting + BLEU tie-breaker (reduce phase)
        
        Returns:
            dict with keys:
                - "answer": aggregated answer string
                - "retrieved_chunks": list of chunk metadata
        """
        # Chunk the document
        chunks, metadatas = self._chunk_text(document)
        
        if not chunks:
            return {"answer": "", "retrieved_chunks": []}
        
        # Map phase: get answer for each chunk
        candidates = []
        retrieved_chunks = []
        
        for chunk_text, metadata in zip(chunks, metadatas):
            # Ask model for this chunk
            candidate_answer = self.model.ask(question, chunk_text)
            candidates.append(candidate_answer)
            
            # Store chunk info
            retrieved_chunks.append({
                "text": chunk_text,
                "metadata": metadata
            })
        
        # Reduce phase: aggregate candidates
        final_answer = self._aggregate_candidates(candidates)
        
        return {
            "answer": final_answer,
            "retrieved_chunks": retrieved_chunks[:self.top_k]  # Return top_k chunks for consistency
        }
