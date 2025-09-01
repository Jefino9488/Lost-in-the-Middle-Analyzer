# app.py (replace file)
import random
import json
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.evaluation import plot_accuracy_by_position, plot_accuracy_by_context, run_experiment, aggregate_summary, run_single
from core.generator import make_dataset
from core.methods.full_context import FullContext
from core.methods.hybrid_rag import HybridRAG
from core.methods.map_reduce import MapReduce
from core.methods.query_summarization import QuerySummarization
from core.methods.rag_bm25 import RAGBM25
from core.methods.re_ranking import ReRanking
from core.methods.sliding_window import SlidingWindow
from core.models import get_model, list_ollama_models, list_gemini_models, list_openrouter_models

load_dotenv()
st.set_page_config(page_title="Lost-in-the-Middle Analyzer", page_icon="📏", layout="wide")
st.title("📏 Lost-in-the-Middle Analyzer")

if "last_config" not in st.session_state:
    st.session_state["last_config"] = None

with st.sidebar:
    st.header("Model & Method")
    provider = st.selectbox("LLM Provider", ["dummy-local", "ollama", "gemini-api", "openrouter"], index=0)
    if provider == "ollama":
        models = list_ollama_models()
        model_name = st.selectbox("Model name", models, index=0)
    elif provider == "gemini-api":
        models = list_gemini_models()
        model_name = st.selectbox("Model name", models, index=0)
    elif provider == "openrouter":
        free_only = st.checkbox("Show only free models", value=True)
        models = list_openrouter_models(free_only=free_only)
        model_name = st.selectbox("Model name", models, index=0)
    else:
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

    st.header("Run options")
    seed = st.number_input("Random seed", value=42)
    show_cost_latency = st.checkbox("Show cost/latency metrics", value=True)
    re_run_last = st.button("Re-run last config")

    run_btn = st.button("Run Experiment 🚀")

# Load model
model = get_model(provider, model_name)

# method mapping
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

# Re-run last config if pressed
if re_run_last and st.session_state["last_config"]:
    cfg = st.session_state["last_config"]
    provider = cfg["provider"]
    model_name = cfg["model_name"]
    method_name = cfg["method_name"]
    n_docs = cfg["n_docs"]
    context_len = cfg["context_len"]
    positions = cfg["positions"]
    seed = cfg["seed"]
    # get method/model afresh
    model = get_model(provider, model_name)
    method = method_map.get(method_name, method)

if run_btn:
    # store last config
    st.session_state["last_config"] = {
        "provider": provider,
        "model_name": model_name,
        "method_name": method_name,
        "n_docs": n_docs,
        "context_len": context_len,
        "positions": positions,
        "seed": seed
    }

    random.seed(seed)
    with st.spinner("Generating dataset..."):
        ds = make_dataset(n_docs=n_docs, context_tokens=context_len, positions=positions)
    st.success(f"Generated {len(ds)} synthetic items.")

    st.subheader("Running evaluation...")
    progress = st.progress(0, text="Starting...")
    rows = []
    start_run = time.time()
    for i, item in enumerate(ds):
        row = run_single(item, method, provider_key=provider, index=i)
        rows.append(row)
        progress.progress((i + 1) / len(ds), text=f"Processed {i + 1}/{len(ds)} docs")
    total_time = time.time() - start_run

    results = pd.DataFrame(rows)
    results.attrs["duration_s"] = total_time

    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_accuracy_by_position(results))
    with col2:
        st.pyplot(plot_accuracy_by_context(results))

    st.markdown("### Aggregate summary (method × position)")
    agg = aggregate_summary(results, by=["model","position"])
    st.dataframe(agg)

    # Export JSONL and CSV including metadata
    run_metadata = {
        "seed": int(seed),
        "provider": provider,
        "model_name": model_name,
        "method_name": method_name,
        "n_docs": int(n_docs),
        "context_len": int(context_len),
        "positions": positions,
        "run_start": int(start_run),
        "run_end": int(time.time())
    }
    # JSONL
    jsonl_lines = []
    for row in rows:
        out = {"metadata": run_metadata, "row": row}
        jsonl_lines.append(json.dumps(out))
    jsonl_blob = "\n".join(jsonl_lines)
    st.download_button("Download raw results (JSONL)", data=jsonl_blob, file_name="results.jsonl")

    # CSV (rows flattened)
    csv_blob = results.to_csv(index=False)
    st.download_button("Download raw results (CSV)", data=csv_blob, file_name="results.csv")

    if show_cost_latency:
        st.markdown("### Cost & Latency overview")
        total_cost = results["cost_usd"].sum()
        total_latency = results["latency_ms"].replace(-1, pd.NA).dropna().sum()
        st.metric("Total cost (USD)", f"${total_cost:.6f}")
        st.metric("Total latency (ms, summed known)", f"{int(total_latency) if not pd.isna(total_latency) else 'unknown'}")
        st.dataframe(results[["i","position","EM","edit_distance","cost_usd","latency_ms"]].head(200))

st.markdown("---")
st.markdown("**Tips:** Use `dummy-local` to test without calling an external LLM. When ready, switch to `gemini-api` or `openrouter` and ensure API keys are set in `.env`.")
