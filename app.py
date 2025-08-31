import pandas as pd
import streamlit as st
import random
from core.generator import make_dataset
from core.evaluation import run_experiment, plot_accuracy_by_position, plot_accuracy_by_context
from core.models import get_model
from core.methods.full_context import FullContext
from core.methods.sliding_window import SlidingWindow
from core.methods.rag_bm25 import RAGBM25
from core.methods.map_reduce import MapReduce
from core.methods.re_ranking import ReRanking
from core.methods.query_summarization import QuerySummarization
from core.methods.hybrid_rag import HybridRAG
from core.evaluation import run_single

st.set_page_config(page_title="Lost-in-the-Middle Analyzer", page_icon="📏", layout="wide")
st.title("📏 Lost-in-the-Middle Analyzer")

with st.sidebar:
    st.header("Model & Method")
    provider = st.selectbox("LLM Provider", ["dummy-local", "ollama", "vertex-gemini", "openai"], index=0)
    model_name = st.text_input("Model name (provider-specific)", value="dummy-echo")

    method_name = st.selectbox("Method", [
        "FullContext",
        "SlidingWindow",
        "RAG-BM25",
        "Map-Reduce",
        "Re-Ranking",
        "Query-Summarization",
        "Hybrid-RAG"
    ], index=2)
    top_k = st.slider("top_k / windows", 1, 10, 3)
    window_size = st.slider("Window size (tokens, approx)", 200, 4000, 800, step=100)

    st.header("Dataset")
    n_docs = st.slider("# docs", 10, 300, 50, step=10)
    context_len = st.slider("Context length (tokens, approx)", 500, 12000, 3000, step=500)
    positions = st.multiselect("Answer positions", ["start", "middle", "end"], default=["start","middle","end"])

    seed = st.number_input("Random seed", value=42)
    run_btn = st.button("Run Experiment 🚀")

# Prepare model
model = get_model(provider, model_name)

# Choose method
method_map = {
    "FullContext": FullContext(model=model),
    "SlidingWindow": SlidingWindow(model=model, window_size=window_size, num_windows=top_k),
    "RAG-BM25": RAGBM25(model=model, top_k=top_k),
    "Map-Reduce": MapReduce(model=model, chunk_size=window_size, top_k=top_k),
    "Re-Ranking": ReRanking(model=model, top_k=top_k),
    "Query-Summarization": QuerySummarization(model=model, top_k=top_k),
    "Hybrid-RAG": HybridRAG(model=model, top_k=top_k)
}
method = method_map[method_name]

if run_btn:
    random.seed(seed)
    with st.spinner("Generating dataset..."):
        ds = make_dataset(n_docs=n_docs, context_tokens=context_len, positions=positions)
    st.success(f"Generated {len(ds)} synthetic items.")

    st.subheader("Running evaluation... (low compute)")
    progress = st.progress(0, text="Starting...")
    results_list = []

    for i, item in enumerate(ds):
        answer = run_single(item, method)  # run experiment on a single doc
        results_list.append(answer)

        progress.progress((i + 1) / len(ds), text=f"Processed {i + 1}/{len(ds)} docs")

    results = pd.DataFrame(results_list)

    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_accuracy_by_position(results))
    with col2:
        st.pyplot(plot_accuracy_by_context(results))

    st.download_button("Download raw results (JSON)", data=results.to_json(orient="records"), file_name="results.json")

st.markdown("---")
st.markdown("**Tips:** Use `dummy-local` to test without calling an external LLM. When ready, switch to `vertex-gemini` and deploy to Cloud Run.")