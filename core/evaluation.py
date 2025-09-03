import math
import re
import time
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.utils import (normalize_for_em, strict_exact_match, edit_distance,
                        precision_recall_f1, bleu_score, ensure_trace)


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
    question = item["question"]
    doc = item.get("document") or item.get("doc")
    gold = item.get("answer")
    position = item.get("position")

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
            "model": "unknown",
        }

    pred = method.answer(question, doc) or ""
    model_obj = getattr(method, "model", None)
    model_name_str = "unknown"
    if hasattr(model_obj, 'model_name'):
        model_name_str = model_obj.model_name
    elif model_obj is not None:
        if model_obj.__class__.__name__ == 'DummyModel':
            model_name_str = "dummy-echo"
        else:
            model_name_str = str(model_obj)
    else:
        model_name_str = getattr(method, "name", method.__class__.__name__)

    trace = ensure_trace(
        text=pred,
        prompt=question + "\n" + doc,
        provider=provider_key,
        model=model_name_str,
    )

    em = 1 if strict_exact_match(pred, gold) else 0
    ed = edit_distance(normalize_for_em(pred), normalize_for_em(gold))
    prec, rec, f1 = precision_recall_f1(pred, gold)
    bleu = bleu_score(pred, gold)

    row = {
        "i": index,
        "position": position,
        "context_tokens": item.get("context_tokens", trace["input_tokens"]),
        "answer": gold,
        "pred": pred,
        "EM": em,
        "edit_distance": ed,
        "false_positive": int(bool(re.findall(r"ANSWER-\d{4}", pred)) and not re.findall(r"ANSWER-\d{4}", gold or "")),
        "decoy_confusion": int(any(pc != gold and pc in doc for pc in re.findall(r"ANSWER-\d{4}", pred))),
        "calls": 1,
        **trace,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "bleu": bleu,
    }
    return row

# --- plotting helpers ------------------------------------------------------
def plot_accuracy_by_position(df: pd.DataFrame, metric: str = "EM"):
    if df.empty or metric not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data for {metric}", ha="center")
        return fig
    acc = df.groupby("position")[metric].mean().reindex(["start", "middle", "end"]).dropna()
    fig, ax = plt.subplots(figsize=(6,4))
    acc.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1 if metric in ["EM", "precision", "recall", "f1"] else None)
    ax.set_title(f"{metric} by answer position")
    ax.set_ylabel(metric)
    ax.set_xlabel("Position")
    for i, v in enumerate(acc.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    return fig

def plot_accuracy_by_context(df: pd.DataFrame, metric: str = "EM"):
    if df.empty or metric not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data for {metric}", ha="center")
        return fig
    df = df.copy()
    df['ctx_bin'] = pd.qcut(df['context_tokens'].replace(0, 1), q=6, duplicates='drop')
    acc = df.groupby("ctx_bin", observed=False)[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(acc)), acc[metric], marker='o')
    ax.set_ylim(0, 1 if metric in ["EM", "precision", "recall", "f1"] else None)
    ax.set_title(f"{metric} vs. context length (binned)")
    ax.set_ylabel(metric)
    ax.set_xlabel("Context bins (increasing)")
    plt.tight_layout()
    return fig

# --- aggregate summary -----------------------------------------------------
def aggregate_summary(df: pd.DataFrame, by = ["method","position"]):
    """
    Return mean EM + 95% CI and cost/latency aggregates grouped by 'by'.
    """
    groups = []
    grouping_keys = [key for key in by]
    gb = df.groupby(grouping_keys)
    for name, g in gb:
        ems = g["EM"].tolist()
        mean_em, lo, hi = mean_confidence_interval(ems)
        total_cost = g["cost_usd"].sum()
        total_latency = g["latency_ms"].replace(-1, np.nan).sum(skipna=True)
        mean_latency = g["latency_ms"].replace(-1, np.nan).mean()
        group_name = name if not isinstance(name, tuple) else dict(zip(grouping_keys, name))
        groups.append({
            **group_name,
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
            "decoy_confusion_rate": g["decoy_confusion"].mean(),
            "mean_precision": g["precision"].mean(),
            "mean_recall": g["recall"].mean(),
            "mean_f1": g["f1"].mean(),
            "mean_bleu": g["bleu"].mean(),
        })
    return pd.DataFrame(groups)
