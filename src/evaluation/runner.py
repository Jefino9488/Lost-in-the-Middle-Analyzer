import time
import re
import math
import pandas as pd
import numpy as np
from typing import List, Dict
from src.utils import (normalize_for_em, strict_exact_match, edit_distance,
                        precision_recall_f1, bleu_score, ensure_trace)
from src.evaluation.metrics import mean_confidence_interval

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

    result = method.answer(question, doc)
    if isinstance(result, dict):
        pred = result.get("answer", "") or ""
        retrieved = result.get("retrieved_chunks", [])
    else:
        pred = result or ""
        retrieved = []
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
    
    # Retrieval metrics
    hit_rate = 0
    mrr = 0.0
    if retrieved and gold:
        # Check if we have offsets for robust verification
        answer_start = item.get("answer_start_index", -1)
        answer_end = item.get("answer_end_index", -1)
        
        rank = -1
        for idx, chunk in enumerate(retrieved):
            is_hit = False
            
            # Case 1: Chunk is a dict with metadata (New VectorRAG)
            if isinstance(chunk, dict) and "metadata" in chunk:
                meta = chunk["metadata"]
                chunk_start = meta.get("start", -1)
                chunk_end = meta.get("end", -1)
                
                if answer_start != -1 and chunk_start != -1:
                    # Robust offset check: Answer must overlap with chunk
                    # Ideally answer is fully contained, but overlap is enough for a "hit" in RAG
                    # Let's say if the answer starts within the chunk
                    if chunk_start <= answer_start < chunk_end:
                        is_hit = True
                else:
                    # Fallback to string check on chunk text
                    chunk_text = chunk.get("text", "")
                    if gold in chunk_text:
                        is_hit = True
            
            # Case 2: Chunk is a string (Legacy methods)
            elif isinstance(chunk, str):
                if gold in chunk:
                    is_hit = True
            
            if is_hit:
                rank = idx + 1
                break
        
        if rank > 0:
            hit_rate = 1
            mrr = 1.0 / rank

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
        "hit_rate": hit_rate,
        "mrr": mrr,
        "answer_start_index": item.get("answer_start_index", -1),  # Add for positional decay
        "document": doc,  # Keep for positional decay calculation
    }
    return row

def aggregate_summary(df: pd.DataFrame, by = ["model", "method", "position"]):
    """
    Return mean EM + 95% CI and cost/latency aggregates grouped by 'by'.
    Default grouping: model, method, position (changed from method, position for app compatibility).
    """
    df = df.copy()
    # Normalize column names if needed
    if 'method' not in df.columns and 'method_name' in df.columns:
        df = df.rename(columns={'method_name': 'method'})
    if 'model' not in df.columns and 'provider' in df.columns:
        # Use provider as model if model column missing
        df['model'] = df['provider']
    
    groups = []
    grouping_keys = [key for key in by if key in df.columns]
    
    if not grouping_keys:
        # Fallback if no keys found
        grouping_keys = ['position'] if 'position' in df.columns else []
    
    if not grouping_keys:
        # No valid grouping, return overall stats
        grouping_keys = [None]
    
    gb = df.groupby(grouping_keys) if grouping_keys != [None] else [(None, df)]
    
    for name, g in gb:
        ems = g["EM"].tolist()
        mean_em, lo, hi = mean_confidence_interval(ems)
        total_cost = g["cost_usd"].sum()
        total_latency = g["latency_ms"].replace(-1, np.nan).sum(skipna=True)
        mean_latency = g["latency_ms"].replace(-1, np.nan).mean()
        
        if grouping_keys != [None]:
            group_name = name if not isinstance(name, tuple) else dict(zip(grouping_keys, name))
        else:
            group_name = {}
        
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
            "mean_hit_rate": g["hit_rate"].mean(),
            "mean_mrr": g["mrr"].mean(),
        })
    return pd.DataFrame(groups)


def compute_positional_decay(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """
    Compute EM as function of answer position ratio - THE CORE LOST-IN-THE-MIDDLE METRIC.
    
    This calculates EM for different positions within documents (0-5%, 5-10%, ..., 95-100%)
    to show how retrieval accuracy varies with answer position.
    
    Args:
        df: Results DataFrame with answer_start_index and document columns
        n_bins: Number of position bins (default 20 for 5% intervals)
    
    Returns:
        DataFrame with columns: position_bin, mean_em, count, mean_position_ratio
    """
    df = df.copy()
    
    # Calculate position ratio for each item
    def calc_position_ratio(row):
        start_idx = row.get('answer_start_index', -1)
        doc = row.get('document', '')
        if start_idx >= 0 and len(doc) > 0:
            return start_idx / len(doc)
        return -1
    
    df['position_ratio'] = df.apply(calc_position_ratio, axis=1)
    
    # Filter valid ratios
    valid = df[df['position_ratio'] >= 0].copy()
    
    if len(valid) == 0:
        # No valid position data
        return pd.DataFrame(columns=['position_bin', 'mean_em', 'count', 'mean_position_ratio'])
    
    # Bin into n_bins
    bins = np.linspace(0, 1, n_bins + 1)
    labels = [f"{int(i*100/n_bins)}-{int((i+1)*100/n_bins)}%" for i in range(n_bins)]
    
    valid['position_bin'] = pd.cut(
        valid['position_ratio'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    
    # Compute EM per bin
    decay_df = valid.groupby('position_bin', observed=True).agg({
        'EM': ['mean', 'count'],
        'position_ratio': 'mean'
    }).reset_index()
    
    # Flatten column names
    decay_df.columns = ['position_bin', 'mean_em', 'count', 'mean_position_ratio']
    
    return decay_df
