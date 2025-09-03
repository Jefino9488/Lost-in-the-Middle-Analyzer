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
st.title("Lost-in-the-Middle Analyzer")

# Header with logo and intro

st.markdown(
    """
    Analyze how LLMs handle long contexts and where answers get lost. Configure a synthetic dataset, choose a method,
    and benchmark accuracy across answer positions and context lengths. Use the Examples to get started quickly.
    """
)

# Quick-start examples to prefill settings
with st.expander("Examples", expanded=True):
    preset = st.selectbox(
        "Choose a preset",
        [
            "Quick demo (fast)",
            "Lost-in-the-middle stress",
            "Map-Reduce large context",
        ],
        help="Prefill the sidebar with a recommended configuration.",
        key="preset_choice",
    )
    if st.button("Load preset", key="load_preset_btn"):
        if preset == "Quick demo (fast)":
            st.session_state.update({
                "provider": "dummy-local",
                "method_name": "FullContext",
                "n_docs": 10,
                "context_len": 1000,
                "positions": ["start","middle","end"],
                "top_k": 3,
                "window_size": 800,
                "seed": 42,
            })
        elif preset == "Lost-in-the-middle stress":
            st.session_state.update({
                "provider": "dummy-local",
                "method_name": "SlidingWindow",
                "n_docs": 60,
                "context_len": 6000,
                "positions": ["start","middle","end"],
                "top_k": 5,
                "window_size": 1200,
                "seed": 7,
            })
        elif preset == "Map-Reduce large context":
            st.session_state.update({
                "provider": "dummy-local",
                "method_name": "Map-Reduce",
                "n_docs": 40,
                "context_len": 9000,
                "positions": ["start","middle","end"],
                "top_k": 3,
                "window_size": 1500,
                "seed": 13,
            })
        st.rerun()

if "last_config" not in st.session_state:
    st.session_state["last_config"] = None

with st.sidebar:
    st.header("Model & Method")
    provider = st.selectbox(
        "LLM Provider",
        ["dummy-local", "ollama", "gemini-api", "openrouter"],
        index=0,
        help="Choose where to run the model: local dummy for fast testing, Ollama for local LLMs, or hosted APIs.",
        key="provider",
    )
    if provider == "ollama":
        models = list_ollama_models()
        model_name = st.selectbox(
            "Model name",
            models,
            index=0,
            help="Ollama local models from `ollama list`. With Docker Compose, set OLLAMA_BASE_URL to use the API.",
            key="model_name",
        )
    elif provider == "gemini-api":
        models = list_gemini_models()
        model_name = st.selectbox(
            "Model name",
            models,
            index=0,
            help="Google Gemini model to query via API.",
            key="model_name",
        )
    elif provider == "openrouter":
        free_only = st.checkbox("Show only free models", value=True, help="Filter OpenRouter catalog to free-access models.")
        models = list_openrouter_models(free_only=free_only)
        model_name = st.selectbox(
            "Model name",
            models,
            index=0,
            help="OpenRouter model identifier (provider/model).",
            key="model_name",
        )
    else:
        model_name = st.text_input(
            "Model name (provider-specific)",
            value=st.session_state.get("model_name", "dummy-echo"),
            help="For dummy-local you can leave this as-is.",
            key="model_name",
        )

    method_name = st.selectbox(
        "Method",
        [
            "FullContext",
            "SlidingWindow",
            "RAG-BM25",
            "Map-Reduce",
            "Re-Ranking",
            "Query-Summarization",
            "Hybrid-RAG"
        ],
        index=2,
        help="How to process long contexts: full pass, sliding windows, retrieval + rank/summarization, etc.",
        key="method_name",
    )
    top_k = st.slider(
        "top_k / windows", 1, 10, st.session_state.get("top_k", 3),
        help="For retrieval-based methods: number of chunks to keep. For SlidingWindow: number of windows.",
        key="top_k",
    )
    window_size = st.slider(
        "Window size (tokens, approx)", 200, 4000, st.session_state.get("window_size", 800), step=100,
        help="Approximate token size per chunk/window (coarse heuristic).",
        key="window_size",
    )

    st.header("Dataset")
    n_docs = st.slider(
        "# docs", 10, 300, st.session_state.get("n_docs", 50), step=10,
        help="Number of synthetic documents to generate.",
        key="n_docs",
    )
    metric_choice = st.selectbox(
        "Metric to visualize",
        ["EM", "precision", "recall", "f1", "bleu"],
        index=0,
        help="Choose which evaluation metric to plot."
    )
    context_len = st.slider(
        "Context length (tokens, approx)", 500, 12000, st.session_state.get("context_len", 3000), step=500,
        help="Approximate size of each synthetic document.",
        key="context_len",
    )
    positions = st.multiselect(
        "Answer positions", ["start", "middle", "end"],
        default=st.session_state.get("positions", ["start","middle","end"]),
        help="Where the ground-truth answer is placed in the document.",
        key="positions",
    )

    st.header("Run options")
    seed = st.number_input("Random seed", value=st.session_state.get("seed", 42), help="For reproducible dataset generation.", key="seed")
    show_cost_latency = st.checkbox("Show cost/latency metrics", value=st.session_state.get("show_cost_latency", True), help="Display cost and latency estimates after the run.", key="show_cost_latency")
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
    # Update UI state from last config and rerun to reflect in widgets
    st.session_state.update({
        "provider": cfg.get("provider", st.session_state.get("provider")),
        "model_name": cfg.get("model_name", st.session_state.get("model_name")),
        "method_name": cfg.get("method_name", st.session_state.get("method_name")),
        "n_docs": cfg.get("n_docs", st.session_state.get("n_docs")),
        "context_len": cfg.get("context_len", st.session_state.get("context_len")),
        "positions": cfg.get("positions", st.session_state.get("positions")),
        "seed": cfg.get("seed", st.session_state.get("seed")),
        "show_cost_latency": cfg.get("show_cost_latency", st.session_state.get("show_cost_latency", True)),
    })
    st.rerun()

if run_btn:
    # store last config
    st.session_state["last_config"] = {
        "provider": provider,
        "model_name": model_name,
        "method_name": method_name,
        "n_docs": n_docs,
        "context_len": context_len,
        "positions": positions,
        "seed": seed,
        "show_cost_latency": bool(show_cost_latency),
    }

    random.seed(seed)
    with st.spinner("Generating dataset..."):
        ds = make_dataset(n_docs=n_docs, context_tokens=context_len, positions=positions)
    st.success(f"Generated {len(ds)} synthetic items.")

    st.subheader("Running evaluation...")
    col_prog, col_log = st.columns([1, 1])
    with col_prog:
        progress = st.progress(0, text="Starting...")
    with col_log:
        st.caption("Live log")
        log_box = st.empty()
    rows = []
    logs = []
    start_run = time.time()
    for i, item in enumerate(ds):
        # live log update before and after
        logs.append(f"→ Processing {i + 1}/{len(ds)} | position={item.get('position')} | tokens≈{item.get('context_tokens')}")
        log_box.text("\n".join(logs[-25:]))
        row = run_single(item, method, provider_key=provider, index=i)
        rows.append(row)
        progress.progress((i + 1) / len(ds), text=f"Processed {i + 1}/{len(ds)} docs")
        logs.append(f"✓ Done {i + 1}/{len(ds)} | EM={row.get('EM')} | edit_distance={row.get('edit_distance')}")
        log_box.text("\n".join(logs[-25:]))
    total_time = time.time() - start_run

    results = pd.DataFrame(rows)
    results.attrs["duration_s"] = total_time

    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_accuracy_by_position(results, metric=metric_choice))
    with col2:
        st.pyplot(plot_accuracy_by_context(results, metric=metric_choice))

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
