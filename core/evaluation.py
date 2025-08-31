from typing import List, Dict
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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
            f1, precision, recall, bleu = 0, 0, 0, 0
        else:
            pred = method.answer(question=question, document=doc) or ""
            correct = answer.strip().lower() in pred.strip().lower()

            # Prepare for more complex metrics
            pred_tokens = pred.strip().lower().split()
            answer_tokens = answer.strip().lower().split()

            # F1, Precision, Recall
            # This is a simplified calculation for our specific use case (ANSWER-####).
            # It treats the problem as a binary classification: is the correct token present?
            # For more complex answers, you would need a more robust token-based comparison.
            correct_token_found = int(bool(correct))
            f1 = f1_score([1], [correct_token_found], average='binary', zero_division=0)
            precision = precision_score([1], [correct_token_found], average='binary', zero_division=0)
            recall = recall_score([1], [correct_token_found], average='binary', zero_division=0)

            # BLEU Score
            # Use a smoothing function for short sentences to avoid a zero score.
            chencherry = SmoothingFunction()
            bleu = sentence_bleu([answer_tokens], pred_tokens, smoothing_function=chencherry.method1)

        rows.append({
            "i": i,
            "position": item.get("position"),
            "context_tokens": item.get("context_tokens", len(doc.split()) if doc else 0),
            "answer": answer,
            "pred": pred,
            "correct": int(bool(correct)),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "bleu": bleu,
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
    """Evaluate a single dataset item with a method, returning more metrics."""
    question = item["question"]
    doc = item.get("document") or item.get("doc")
    answer = item["answer"]

    if not doc:
        return {
            "question": question,
            "answer": answer,
            "pred": "[ERROR: empty doc]",
            "correct": 0,
            "f1": 0,
            "precision": 0,
            "recall": 0,
            "bleu": 0,
            "position": item.get("position"),
            "context_tokens": 0,
        }

    pred = method.answer(question, doc) or ""
    correct = answer.strip().lower() in pred.strip().lower()

    # Prepare for more complex metrics
    pred_tokens = pred.strip().lower().split()
    answer_tokens = answer.strip().lower().split()

    # F1, Precision, Recall
    correct_token_found = int(bool(correct))
    f1 = f1_score([1], [correct_token_found], average='binary', zero_division=0)
    precision = precision_score([1], [correct_token_found], average='binary', zero_division=0)
    recall = recall_score([1], [correct_token_found], average='binary', zero_division=0)

    # BLEU Score
    chencherry = SmoothingFunction()
    bleu = sentence_bleu([answer_tokens], pred_tokens, smoothing_function=chencherry.method1)

    return {
        "question": question,
        "answer": answer,
        "pred": pred,
        "correct": int(correct),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "bleu": bleu,
        "position": item.get("position"),
        "context_tokens": item.get("context_tokens", len(doc.split())),
    }
