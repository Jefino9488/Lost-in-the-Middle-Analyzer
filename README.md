# Lost-in-the-Middle Analyzer

A low-compute framework to benchmark and visualize long-context failures.

## Supported Methods
- **FullContext** – Pass entire document.
- **SlidingWindow** – Break into fixed-size windows and vote.
- **RAG-BM25** – Retrieve top-k with BM25 then ask LLM on retrieved context.
- **Map-Reduce** – Chunk → partial answers → majority.
- **Re-Ranking** – BM25 retrieval + re-rank top-k (already in methods).
- **Query-Summarization** – Summarize around the question per chunk.
- **Hybrid-RAG** – BM25 retrieval + summarization fusion.

## Local Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
