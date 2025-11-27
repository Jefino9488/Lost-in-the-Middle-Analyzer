import re
import time
import warnings

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
try:
    import tiktoken
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
    if not pred or not gold: # optional: better token counting if available
        return 0.0
    smoothing = SmoothingFunction().method1
    return sentence_bleu([gold.split()], pred.split(), smoothing_function=smoothing)


def truncate_text_center(text: str, max_tokens: int, model: str = None) -> tuple[str, bool]:
    """
    Truncate text using center-preserving strategy if it exceeds max_tokens.
    Keeps first 40% and last 40% of tokens, drops middle 20%.
    
    Args:
        text: Text to potentially truncate
        max_tokens: Maximum allowed tokens
        model: Optional model hint for token counting
    
    Returns:
        (truncated_text, was_truncated): Tuple of result text and boolean flag
    """
    if not text:
        return text, False
    
    current_tokens = count_tokens(text, model)
    
    if current_tokens <= max_tokens:
        return text, False
    
    # Need to truncate - keep first 40%, last 40%, drop middle 20%
    # Approximate character positions based on token ratio
    chars_per_token = len(text) / current_tokens if current_tokens > 0 else 4
    target_chars = int(max_tokens * chars_per_token)
    
    # Split into first 40% and last 40%
    first_chunk_size = int(target_chars * 0.4)
    last_chunk_size = int(target_chars * 0.4)
    
    first_part = text[:first_chunk_size]
    last_part = text[-last_chunk_size:] if last_chunk_size > 0 else ""
    
    truncated = f"{first_part}\n\n[...middle section truncated due to token limit...]\n\n{last_part}"
    
    return truncated, True


def chunk_text_by_tokens(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int = 0,
    model: str = None
) -> tuple[list[str], list[dict]]:
    """
    Chunk text using token-aware boundaries instead of character counts.
    
    Args:
        text: Text to chunk
        chunk_tokens: Target tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        model: Optional model hint for tokenizer selection
    
    Returns:
        (chunks, metadatas): List of text chunks and corresponding metadata dicts
            Metadata includes: start, end (char positions), estimated_tokens, chunk_index
    """
    if not text:
        return [], []
    
    chunks = []
    metadatas = []
    
    # Try to use tiktoken for accurate token splitting
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
            
            # Encode entire text
            tokens = enc.encode(text)
            total_tokens = len(tokens)
            
            # Calculate step size (accounting for overlap)
            step = max(1, chunk_tokens - overlap_tokens)
            
            chunk_idx = 0
            position = 0
            
            while position < total_tokens:
                # Extract chunk tokens
                end_position = min(position + chunk_tokens, total_tokens)
                chunk_tokens_slice = tokens[position:end_position]
                
                # Decode back to text
                chunk_text = enc.decode(chunk_tokens_slice)
                
                # Find character positions (approximate via search)
                # This is imperfect but gives reasonable boundaries
                if chunk_idx == 0:
                    char_start = 0
                else:
                    # Find where this chunk starts in the original text
                    # Use the previous chunk end as a hint
                    search_start = metadatas[-1]["end"] if metadatas else 0
                    char_start = text.find(chunk_text[:min(50, len(chunk_text))], search_start)
                    if char_start == -1:
                        char_start = search_start
                
                char_end = char_start + len(chunk_text)
                
                chunks.append(chunk_text)
                metadatas.append({
                    "start": char_start,
                    "end": char_end,
                    "estimated_tokens": len(chunk_tokens_slice),
                    "chunk_index": chunk_idx
                })
                
                position += step
                chunk_idx += 1
            
            return chunks, metadatas
            
        except Exception as e:
            # Fall through to character-based chunking
            warnings.warn(f"Token-based chunking failed ({e}), using character fallback")
    
    # Fallback: character-based chunking with token estimation
    # Approximate: 4 characters ≈ 1 token
    chars_per_token = 4
    chunk_chars = chunk_tokens * chars_per_token
    overlap_chars = overlap_tokens * chars_per_token
    step_chars = max(1, chunk_chars - overlap_chars)
    
    chunk_idx = 0
    position = 0
    text_len = len(text)
    
    while position < text_len:
        end_position = min(position + chunk_chars, text_len)
        chunk_text = text[position:end_position]
        
        chunks.append(chunk_text)
        metadatas.append({
            "start": position,
            "end": end_position,
            "estimated_tokens": count_tokens(chunk_text, model),
            "chunk_index": chunk_idx
        })
        
        position += step_chars
        chunk_idx += 1
    
    return chunks, metadatas



def ensure_trace(
    text: str,
    prompt: str,
    provider: str,
    model: str,
    latency_ms: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
) -> dict:
    """
    Normalize a trace dictionary for a model call.
    Fills in missing token counts, cost, and latency.
    """
    # Token counts
    model_hint = model or provider
    if input_tokens is None:
        input_tokens = count_tokens(prompt, model_hint)
    if output_tokens is None:
        output_tokens = count_tokens(text, model_hint)

    # Latency
    if latency_ms is None:
        latency_ms = -1

    # Cost
    if cost_usd is None:
        cost_usd = estimate_cost_usd(provider.lower(), input_tokens, output_tokens)

    return {
        "text": text.strip() if isinstance(text, str) else str(text),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "latency_ms": int(latency_ms),
        "cost_usd": float(cost_usd),
        "provider": provider,
        "model": model or provider,
    }

