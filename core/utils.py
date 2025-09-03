import re
import time
import warnings

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
try:
    import tiktoken  # optional: better token counting if available
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

# --- Token counting ---------------------------------------------------------
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


def count_tokens(text: str, model: str = None) -> int:
    """
    Return an estimated token count for given text and model.
    Priority:
    1. Use tiktoken if available and model supported.
    2. Use Hugging Face tokenizer if available.
    3. Fallback to character-length heuristic (~4 chars/token).
    """
    if not text:
        return 0
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass

    # --- Hugging Face path ---
    if TRANSFORMERS_AVAILABLE and model:
        try:
            # Try loading a tokenizer for the model family
            if "gemini" in model.lower():
                tok = AutoTokenizer.from_pretrained("google/gemma-2b")
            elif "llama" in model.lower():
                tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            else:
                tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            return len(tok.encode(text))
        except Exception:
            pass

    # --- crude fallback ---
    warnings.warn("Using crude character-based token estimate (no tokenizer available).")
    return max(1, int(len(text) / 4))


# alias for compatibility
approx_tokens_from_text = count_tokens


# --- Normalization / EM -----------------------------------------------------
def normalize_for_em(s: str) -> str:
    """Normalize string: strip, lower, collapse whitespace."""
    if s is None:
        return ""
    s = s.strip().lower()
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    return s

def strict_exact_match(pred: str, gold: str) -> bool:
    """Strict EM using normalized strings and exact equality (not containment)."""
    return normalize_for_em(pred) == normalize_for_em(gold)

# --- edit distance ----------------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance (classic DP)."""
    if a is None: a = ""
    if b is None: b = ""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i-1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,     # deletion
                dp[i][j-1] + 1,     # insertion
                dp[i-1][j-1] + cost # substitution
            )
    return dp[n][m]

# --- simple pricing tables (placeholders) -----------------------------------
# NOTE: these are *approximate* placeholders. Replace with accurate per-provider pricing.
PRICING = {
    "gemini": {"input_per_1k_tokens_usd": 0.002, "output_per_1k_tokens_usd": 0.004},
    "openrouter": {"input_per_1k_tokens_usd": 0.003, "output_per_1k_tokens_usd": 0.003},
    "dummy-local": {"input_per_1k_tokens_usd": 0.0, "output_per_1k_tokens_usd": 0.0},
    "claude-cli": {"input_per_1k_tokens_usd": 0.0, "output_per_1k_tokens_usd": 0.0},
}

def estimate_cost_usd(provider_key: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a single call using PRICING table."""
    p = PRICING.get(provider_key.lower(), PRICING["openrouter"])
    in_cost = (input_tokens / 1000.0) * p["input_per_1k_tokens_usd"]
    out_cost = (output_tokens / 1000.0) * p["output_per_1k_tokens_usd"]
    return round(in_cost + out_cost, 8)

# --- timing helper ----------------------------------------------------------
def timed_call(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), return (result, latency_ms)."""
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return res, int((t1 - t0) * 1000)


def precision_recall_f1(pred: str, gold: str):
    """
    Compute token-level precision, recall, and F1 between prediction and gold.
    """
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    pred_tokens = pred.strip().split()
    gold_tokens = gold.strip().split()
    pred_set, gold_set = set(pred_tokens), set(gold_tokens)

    true_pos = len(pred_set & gold_set)
    precision = true_pos / len(pred_set) if pred_set else 0.0
    recall = true_pos / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def bleu_score(pred: str, gold: str):
    """
    Compute BLEU score between predicted and gold text.
    """
    if not pred or not gold:
        return 0.0
    smoothing = SmoothingFunction().method1
    return sentence_bleu([gold.split()], pred.split(), smoothing_function=smoothing)
