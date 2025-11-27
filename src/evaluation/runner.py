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
    }
    return row

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
            "mean_hit_rate": g["hit_rate"].mean(),
            "mean_mrr": g["mrr"].mean(),
        })
    return pd.DataFrame(groups)
