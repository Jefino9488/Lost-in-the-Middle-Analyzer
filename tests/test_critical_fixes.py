"""
Functional tests for critical bug fixes.
Tests MapReduce, QuerySummarization, FullContext, and SlidingWindow.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock DummyModel for testing
class MockModel:
    """Simple mock model for testing method logic without API calls."""
    def ask(self, question: str, context: str) -> str:
        # Simple logic: return ANSWER-XXXX if found in context
        import re
        match = re.search(r'ANSWER-(\d{4})', context)
        if match:
            return f"The answer is ANSWER-{match.group(1)}"
        return "I couldn't find the answer."


def test_map_reduce_basic():
    """Test MapReduce can be instantiated and returns correct structure."""
    from src.methods.map_reduce import MapReduce
    
    model = MockModel()
    mr = MapReduce(model, chunk_size=50, top_k=2)
    
    # Test with simple document
    doc = "ANSWER-1234 is in the first chunk. " * 5 + "ANSWER-1234 is in second chunk."
    result = mr.answer("What is the answer?", doc)
    
    assert isinstance(result, dict), "MapReduce should return dict"
    assert "answer" in result, "Result should have 'answer' key"
    assert "retrieved_chunks" in result, "Result should have 'retrieved_chunks' key"
    assert isinstance(result["retrieved_chunks"], list), "retrieved_chunks should be list"
    
    # Check that answer contains ANSWER-1234 (majority vote)
    assert "ANSWER-1234" in result["answer"], f"Expected ANSWER-1234 in answer, got: {result['answer']}"
    
    print("✓ MapReduce basic test passed")


def test_map_reduce_aggregation():
    """Test MapReduce aggregation with voting."""
    from src.methods.map_reduce import MapReduce
    
    mr = MapReduce(MockModel(), chunk_size=30)
    
    # Test aggregation directly
    candidates = ["answer a", "answer b", "answer a", "answer a"]
    result = mr._aggregate_candidates(candidates)
    
    assert "answer a" in result.lower(), f"Expected 'answer a', got: {result}"
    
    print("✓ MapReduce aggregation test passed")


def test_query_summarization_basic():
    """Test QuerySummarization can be instantiated and returns correct structure."""
    from src.methods.query_summarization import QuerySummarization
    
    model = MockModel()
    qs = QuerySummarization(model, top_k=2)
    
    doc = "Some document text. ANSWER-5678 is here. More text follows."
    result = qs.answer("What is the answer?", doc)
    
    assert isinstance(result, dict), "QuerySummarization should return dict"
    assert "answer" in result, "Result should have 'answer' key"
    assert "retrieved_chunks" in result, "Result should have 'retrieved_chunks' key"
    assert isinstance(result["retrieved_chunks"], list), "retrieved_chunks should be list"
    
    print("✓ QuerySummarization basic test passed")


def test_full_context_no_truncation():
    """Test FullContext with document under token limit."""
    from src.methods.full_context import FullContext
    
    model = MockModel()
    fc = FullContext(model)
    
    doc = "Short document. ANSWER-9999 is here."
    result = fc.answer("What is the answer?", doc)
    
    assert isinstance(result, dict), "FullContext should return dict"
    assert "answer" in result, "Result should have 'answer' key"
    assert "retrieved_chunks" in result, "Result should have 'retrieved_chunks' key"
    assert len(result["retrieved_chunks"]) > 0, "Should have retrieved chunks"
    
    # Check metadata
    chunk = result["retrieved_chunks"][0]
    assert "metadata" in chunk, "Chunk should have metadata"
    assert chunk["metadata"]["was_truncated"] == False, "Short doc should not be truncated"
    
    print("✓ FullContext no-truncation test passed")


def test_full_context_with_truncation():
    """Test FullContext with very large document that requires truncation."""
    from src.methods.full_context import FullContext
    
    model = MockModel()
    fc = FullContext(model)
    
    # Create a very large document (way over 100k tokens)
    # Each repetition is ~10 words, so we need lots of them
    large_doc = ("This is a large document that will exceed token limits. " * 50000) + "ANSWER-7777"
    
    result = fc.answer("What is the answer?", large_doc)
    
    assert isinstance(result, dict), "FullContext should return dict"
    chunk = result["retrieved_chunks"][0]
    
    # The document should be truncated
    assert chunk["metadata"]["was_truncated"] == True, "Large doc should be truncated"
    assert "truncated" in chunk["text"].lower(), "Truncated text should have marker"
    
    print("✓ FullContext truncation test passed")


def test_sliding_window_basic():
    """Test SlidingWindow works correctly."""
    from src.methods.sliding_window import SlidingWindow
    
    model = MockModel()
    sw = SlidingWindow(model, window_size=50, num_windows=3)
    
    doc = "Some text here. ANSWER-3333 is the answer. More text follows."
    result = sw.answer("What is the answer?", doc)
    
    assert isinstance(result, dict), "SlidingWindow should return dict"
    assert "answer" in result, "Result should have 'answer' key"
    assert "retrieved_chunks" in result, "Result should have 'retrieved_chunks' key"
    
    print("✓ SlidingWindow basic test passed")


if __name__ == "__main__":
    print("Running critical fixes functional tests...\n")
    
    try:
        test_map_reduce_basic()
        test_map_reduce_aggregation()
        test_query_summarization_basic()
        test_full_context_no_truncation()
        test_full_context_with_truncation()
        test_sliding_window_basic()
        
        print("\n" + "="*50)
        print("All tests passed! ✓")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
