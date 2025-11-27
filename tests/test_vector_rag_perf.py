import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.methods.vector_rag import VectorRAG
from src.models.llm_client import DummyModel

def test_performance():
    model = DummyModel()
    # Create a large enough document to make indexing noticeable
    document = "This is a sentence. " * 5000 
    question = "What is this?"
    
    rag = VectorRAG(model=model, top_k=3, chunk_size=500)
    
    print("Starting performance test...")
    
    # First call
    start = time.time()
    rag.answer(question, document)
    first_duration = time.time() - start
    print(f"First call duration: {first_duration:.4f}s")
    
    # Second call (same document)
    start = time.time()
    rag.answer(question, document)
    second_duration = time.time() - start
    print(f"Second call duration: {second_duration:.4f}s")
    
    # Third call (same document)
    start = time.time()
    rag.answer(question, document)
    third_duration = time.time() - start
    print(f"Third call duration: {third_duration:.4f}s")
    
    # In the current implementation, all durations should be roughly equal (and slow).
    # After fix, second and third should be much faster.
    
    if second_duration < first_duration * 0.5:
        print("SUCCESS: Subsequent calls are significantly faster.")
    else:
        print("FAIL: Subsequent calls are not significantly faster (Index is likely rebuilt).")

if __name__ == "__main__":
    test_performance()
