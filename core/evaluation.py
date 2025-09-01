from typing import List, Dict
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

from core.utils import (normalize_for_em, strict_exact_match, edit_distance,
                        approx_tokens_from_text, estimate_cost_usd)

# --- helpers ---------------------------------------------------------------
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n == 0:
        return (0.0, 0.0, 0.0)
    m = np.mean(a)
    se = np.std(a, ddof=1) / math.sqrt(n) if n > 1 else 0.0
    # t-critical for large n ~ z
    from scipy.stats import t
    h = se * t.ppf((1 + confidence) / 2., n-1) if n > 1 else 0.0
    return m, m - h, m + h

# --- run functions ---------------------------------------------------------
def run_experiment(dataset: List[Dict], method, provider_key: str = "unknown") -> pd.DataFrame:
    """Run an experiment and return rows including per-call logs."""
    rows = []
    start = time.time()
    for i, item in enumerate(dataset):
        row = run_single(item, method, provider_key=provider_key, index=i)
        rows.append(row)
    dur = time.time() - start
    df = pd.DataFrame(rows)
    df.attrs["duration_s"] = dur
    # add aggregate fields
    df["EM"] = df["EM"].astype(int)
    return df

def run_single(item: Dict, method, provider_key: str = "unknown", index: int = 0) -> Dict:
    """Evaluate a single item; calls method.answer and tries to extract trace."""
    question = item["question"]
    doc = item.get("document") or item.get("doc")
    gold = item.get("answer")
    position = item.get("position")
    context_tokens_approx = item.get("context_tokens", approx_tokens_from_text(doc or ""))

    if not doc:
        return {
            "i": index,
            "position": position,
            "context_tokens": 0,
            "answer": gold,
            "pred": "[ERROR: empty doc]",
            "EM": 0,
            "edit_distance": edit_distance("", gold or ""),
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
            "cost_usd": 0.0,
            "provider": provider_key,
            "model": getattr(method, "model", getattr(method, "name", "unknown"))
        }

    # Prefer ask_with_trace if available via underlying model
    # many methods call model.ask(question, context); to keep compatibility
    # we prefer to call method.answer and then, if the model supports ask_with_trace,
    # call it directly to get a trace (best effort).
    # First call method.answer as before (it may already perform internal calls).
    pred = method.answer(question, doc) or ""

    # Attempt to get a single best per-call trace:
    input_tokens = None
    output_tokens = None
    latency_ms = None
    cost_usd = None
    provider = provider_key
    model_name = getattr(method, "model", getattr(method, "name", method.__class__.__name__))

    # If the method exposes a .model and that model supports ask_with_trace, call it with
    # the constructed context used by the method when answering (best-effort).
    try:
        the_model = getattr(method, "model", None)
        if hasattr(the_model, "ask_with_trace"):
            # Use doc as context (best-effort)
            trace = the_model.ask_with_trace(question, doc)
            if isinstance(trace, dict):
                input_tokens = trace.get("input_tokens")
                output_tokens = trace.get("output_tokens")
                latency_ms = trace.get("latency_ms")
                cost_usd = trace.get("cost_usd")
                provider = trace.get("provider", provider)
                model_name = trace.get("model", model_name)
    except Exception:
        # ignore trace failures; we'll estimate below
        pass

    # If trace missing, make crude estimates
    if input_tokens is None:
        input_tokens = approx_tokens_from_text(question + "\n" + doc)
    if output_tokens is None:
        output_tokens = approx_tokens_from_text(pred)
    if latency_ms is None:
        latency_ms = -1  # unknown
    if cost_usd is None:
        cost_usd = estimate_cost_usd(provider.lower(), input_tokens, output_tokens)

    # strict EM and edit distance
    em = 1 if strict_exact_match(pred, gold) else 0
    ed = edit_distance(normalize_for_em(pred), normalize_for_em(gold))

    # false positive: predicted a code-like token but gold is missing or empty
    import re
    predicted_codes = re.findall(r"ANSWER-\d{4}", pred or "")
    gold_codes = re.findall(r"ANSWER-\d{4}", gold or "")
    false_positive = 1 if (predicted_codes and not gold_codes) else 0

    # decoy confusion: predicted some valid code token that is present in doc but not the gold
    # i.e., model returned an ANSWER-####, that appears in doc but differs from gold
    decoy_confusion = 0
    if predicted_codes:
        for pc in predicted_codes:
            if pc != gold and pc in doc:
                decoy_confusion = 1
                break

    row = {
        "i": index,
        "position": position,
        "context_tokens": context_tokens_approx,
        "answer": gold,
        "pred": pred,
        "EM": em,
        "edit_distance": ed,
        "false_positive": false_positive,
        "decoy_confusion": decoy_confusion,
        "calls": 1,  # methods generally make 1 external call (methods may internally make several; if you need exact calls, instrument methods)
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "latency_ms": int(latency_ms) if latency_ms is not None else -1,
        "cost_usd": float(cost_usd),
        "provider": provider,
        "model": model_name,
    }
    return row

# --- plotting helpers ------------------------------------------------------
def plot_accuracy_by_position(df: pd.DataFrame):
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig
    acc = df.groupby("position")["EM"].mean().reindex(["start", "middle", "end"]).dropna()
    fig, ax = plt.subplots(figsize=(6,4))
    acc.plot(kind="bar", ax=ax)
    ax.set_ylim(0,1)
    ax.set_title("Strict EM by answer position")
    ax.set_ylabel("EM")
    ax.set_xlabel("Position")
    for i, v in enumerate(acc.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    return fig

def plot_accuracy_by_context(df: pd.DataFrame):
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig
    # group by binned context token sizes
    bins = sorted(set(min(df.context_tokens),) | set(df.context_tokens)) if False else None
    # simpler: bin into 6 quantiles
    df = df.copy()
    df['ctx_bin'] = pd.qcut(df['context_tokens'].replace(0,1), q=6, duplicates='drop')
    acc = df.groupby("ctx_bin")["EM"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(range(len(acc)), acc["EM"], marker='o')
    ax.set_ylim(0,1)
    ax.set_title("EM vs. context length (binned)")
    ax.set_ylabel("EM")
    ax.set_xlabel("Context bins (increasing)")
    plt.tight_layout()
    return fig

# --- aggregate summary -----------------------------------------------------
def aggregate_summary(df: pd.DataFrame, by = ["method","position"]):
    """
    Return mean EM + 95% CI and cost/latency aggregates grouped by 'by'.
    """
    groups = []
    gb = df.groupby(by)
    for name, g in gb:
        ems = g["EM"].tolist()
        mean_em, lo, hi = mean_confidence_interval(ems)
        total_cost = g["cost_usd"].sum()
        total_latency = g["latency_ms"].replace(-1, np.nan).sum(skipna=True)
        mean_latency = g["latency_ms"].replace(-1, np.nan).mean()
        groups.append({
            **({"group": name} if not isinstance(name, tuple) else {"group": name}),
            "n": len(g),
            "mean_em": mean_em,
            "95ci_lo": lo,
            "95ci_hi": hi,
            "total_cost_usd": total_cost,
            "mean_latency_ms": mean_latency if not math.isnan(mean_latency) else None,
            "total_latency_ms": total_latency if not math.isnan(total_latency) else None,
            "mean_input_tokens": g["input_tokens"].mean(),
            "mean_output_tokens": g["output_tokens"].mean(),
            "false_positive_rate": g["false_positive"].mean(),
            "decoy_confusion_rate": g["decoy_confusion"].mean()
        })
    return pd.DataFrame(groups)
