import sys
import os
import random

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.methods.vector_rag import VectorRAG
from src.models.llm_client import DummyModel
from src.evaluation.runner import run_single

def test_metric_fix():
    print("Testing Metric Fix...")
    
    # 1. Create a "False Positive" scenario
    # Answer is "ANSWER-1234"
    # Distractor is "ANSWER-12345" (contains answer as substring)
    
    answer_code = "1234"
    answer = f"ANSWER-{answer_code}"
    
    distractor_code = "12345"
    distractor = f"ANSWER-{distractor_code}"
    
    print(f"Naive check: '{answer}' in '{distractor}' -> {answer in distractor}")
    
    doc_part1 = f"This is some text with a distractor {distractor}. " 
    doc_part2 = f"This is the real answer {answer}. "
    
    document = doc_part1 + doc_part2
    
    # Calculate offsets manually
    # doc.find(answer) finds the distractor because ANSWER-1234 is in ANSWER-12345
    # We must point to the REAL answer in doc_part2
    answer_start = len(doc_part1) + doc_part2.find(answer)
    answer_end = answer_start + len(answer)
    
    print(f"Answer: {answer}")
    print(f"Distractor: {distractor}")
    print(f"Answer Start: {answer_start}")
    
    item = {
        "question": "What is the code?",
        "document": document,
        "answer": answer,
        "answer_start_index": answer_start,
        "answer_end_index": answer_end,
        "position": "middle",
        "context_tokens": 100
    }
    
    # Mock VectorRAG to return specific chunks
    # We want to test the METRIC logic in runner.py, not the retrieval quality of Chroma
    
    # Chunk 1: Contains Distractor (False Positive if naive)
    # Text: "... distractor ANSWER-12345. "
    # Offsets: 0 to len(doc_part1)
    chunk1_text = doc_part1
    chunk1_meta = {"start": 0, "end": len(doc_part1)}
    
    # Chunk 2: Contains Answer (True Positive)
    # Text: "... real answer ANSWER-1234. "
    # Offsets: len(doc_part1) to len(document)
    chunk2_text = doc_part2
    chunk2_meta = {"start": len(doc_part1), "end": len(document)}
    
    # Scenario A: Retrieve ONLY Chunk 1 (Distractor)
    # Should be 0 hit rate with robust check
    # Should be 1 hit rate with naive check (because ANSWER-1234 is in ANSWER-12345)
    
    print("\n--- Scenario A: Retrieve only Distractor ---")
    retrieved_A = [{"text": chunk1_text, "metadata": chunk1_meta}]
    
    class MockMethod:
        def answer(self, q, d):
            return {"answer": "foo", "retrieved_chunks": retrieved_A}
            
    row_A = run_single(item, MockMethod(), provider_key="dummy")
    print(f"Hit Rate: {row_A['hit_rate']}")
    
    if row_A['hit_rate'] == 0:
        print("SUCCESS: Distractor was correctly REJECTED.")
    else:
        print("FAIL: Distractor was ACCEPTED as a hit.")
        
    # Scenario B: Retrieve ONLY Chunk 2 (Answer)
    # Should be 1 hit rate
    
    print("\n--- Scenario B: Retrieve only Answer ---")
    retrieved_B = [{"text": chunk2_text, "metadata": chunk2_meta}]
    
    class MockMethodB:
        def answer(self, q, d):
            return {"answer": "foo", "retrieved_chunks": retrieved_B}
            
    row_B = run_single(item, MockMethodB(), provider_key="dummy")
    print(f"Hit Rate: {row_B['hit_rate']}")
    
    if row_B['hit_rate'] == 1:
        print("SUCCESS: Answer was correctly ACCEPTED.")
    else:
        print("FAIL: Answer was REJECTED.")

    # Scenario C: Legacy Fallback (No metadata)
    # If we pass strings, it should use naive check and FAIL on distractor
    
    print("\n--- Scenario C: Legacy Fallback (Strings) ---")
    retrieved_C = [chunk1_text] # Distractor string
    
    class MockMethodC:
        def answer(self, q, d):
            return {"answer": "foo", "retrieved_chunks": retrieved_C}
            
    row_C = run_single(item, MockMethodC(), provider_key="dummy")
    print(f"Hit Rate: {row_C['hit_rate']}")
    
    if row_C['hit_rate'] == 1:
        print("CONFIRMED: Legacy mode is fragile (accepted distractor).")
    else:
        print("UNEXPECTED: Legacy mode rejected distractor?")


if __name__ == "__main__":
    test_metric_fix()
