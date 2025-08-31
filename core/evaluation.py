from typing import List, Dict
import time
import pandas as pd
import matplotlib.pyplot as plt

def run_experiment(dataset: List[Dict], method) -> pd.DataFrame:
    """Run an experiment on a dataset with a given method."""
    rows = []
    start = time.time()
    for i, item in enumerate(dataset):
        question = item["question"]
        doc = item.get("document") or item.get("doc")  # support both keys
        answer = item["answer"]

        if not doc:
            pred = "[ERROR: empty doc]"
            correct = False
        else:
            pred = method.answer(question=question, document=doc) or ""
            correct = answer.strip().lower() in pred.strip().lower()

        rows.append({
            "i": i,
            "position": item.get("position"),
            "context_tokens": item.get("context_tokens", len(doc.split()) if doc else 0),
            "answer": answer,
            "pred": pred,
            "correct": int(bool(correct)),
            "method": getattr(method, "name", method.__class__.__name__),
        })
    dur = time.time() - start
    df = pd.DataFrame(rows)
    df.attrs["duration_s"] = dur
    return df

def plot_accuracy_by_position(df: pd.DataFrame):
    """Bar chart: accuracy grouped by answer position (start/middle/end)."""
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig
    acc = df.groupby("position")["correct"].mean().reindex(["start", "middle", "end"]).dropna()
    fig, ax = plt.subplots(figsize=(6,4))
    acc.plot(kind="bar", ax=ax)
    ax.set_ylim(0,1)
    ax.set_title("Accuracy by answer position")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Position")
    for i, v in enumerate(acc.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    return fig

def plot_accuracy_by_context(df: pd.DataFrame):
    """Line chart: accuracy as a function of context length."""
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig

    acc = (
        df.groupby("context_tokens")["correct"]
        .mean()
        .reset_index()
        .sort_values("context_tokens")
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(acc["context_tokens"], acc["correct"], marker='o')
    ax.set_ylim(0,1)
    ax.set_title("Accuracy vs. context length")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Approx. tokens")
    plt.tight_layout()
    return fig

def run_single(item, method):
    """Evaluate a single dataset item with a method."""
    question = item["question"]
    doc = item.get("document") or item.get("doc")
    answer = item["answer"]

    if not doc:
        return {
            "question": question,
            "answer": answer,
            "pred": "[ERROR: empty doc]",
            "correct": 0,
            "position": item.get("position"),
            "context_tokens": 0,
        }

    pred = method.answer(question, doc) or ""
    correct = answer.strip().lower() in pred.strip().lower()

    return {
        "question": question,
        "answer": answer,
        "pred": pred,
        "correct": int(correct),
        "position": item.get("position"),
        "context_tokens": item.get("context_tokens", len(doc.split())),
    }
