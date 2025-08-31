import random
import textwrap
from typing import List, Dict

SEED_SENTENCES = [
    "Quantum cats dance on probabilistic pianos.",
    "The API gateway logs were rotated at midnight.",
    "A gentle breeze carried the scent of pine.",
    "Embeddings improve retrieval when tuned for domain.",
    "The quick brown fox jumps over the lazy dog.",
    "Serverless functions scaled during the influx of traffic.",
]

def _filler(n: int) -> str:
    # n ~ number of tokens (approx). We'll approximate by words.
    pool = SEED_SENTENCES
    words = []
    for _ in range(max(1, n // 6)):
        words.extend(random.choice(pool).split())
    # trim to n words max
    return " ".join(words)[: max(1, n * 4)]  # crude char-length safety

def make_item(context_tokens: int, position: str) -> Dict:
    """
    Create a synthetic doc where an 'ANSWER-####' token sits at start/middle/end.
    context_tokens is approximate token count (used for filler sizing).
    """
    answer = f"ANSWER-{random.randint(1000,9999)}"
    # Build parts to put answer at requested position
    total_words = max(200, context_tokens)  # coarse
    third = total_words // 3

    if position == "start":
        parts = [answer, _filler(third), _filler(third)]
    elif position == "end":
        parts = [_filler(third), _filler(third), answer]
    else:  # middle
        parts = [_filler(third), answer, _filler(third)]

    doc = " ".join(parts)
    # make pretty paragraph-wrapped text
    doc = textwrap.fill(doc, width=120)
    question = "What is the hidden code? Respond with the exact code."
    return {
        "document": doc,
        "question": question,
        "answer": answer,
        "position": position,
        "context_tokens": context_tokens,
    }

def make_dataset(n_docs: int, context_tokens: int, positions: List[str]):
    ds = [make_item(context_tokens=context_tokens, position=random.choice(positions)) for _ in range(n_docs)]
    return ds
