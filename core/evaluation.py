from typing import List, Dict
import time
import pandas as pd
import matplotlib.pyplot as plt

def run_experiment(dataset: List[Dict], method) -> pd.DataFrame:
    rows = []
    start = time.time()
    for i, item in enumerate(dataset):
        pred = method.answer(question=item["question"], document=item["doc"]) or ""
        # naive exact-match check; gold designed to be unique
        correct = str(item["gold"]).strip() in pred
        rows.append({
            "i": i,
            "position": item["position"],
            "context_tokens": item["context_tokens"],
            "gold": item["gold"],
            "pred": pred,
            "correct": int(bool(correct)),
            "method": getattr(method, "name", method.__class__.__name__),
        })
    dur = time.time() - start
    df = pd.DataFrame(rows)
    df.attrs["duration_s"] = dur
    return df

def plot_accuracy_by_position(df: pd.DataFrame):
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
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig
    acc = df.groupby("context_tokens")["correct"].mean().reset_index().sort_values("context_tokens")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(acc["context_tokens"], acc["correct"], marker='o')
    ax.set_ylim(0,1)
    ax.set_title("Accuracy vs. context length")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Approx. tokens")
    plt.tight_layout()
    return fig
