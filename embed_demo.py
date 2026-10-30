#!/usr/bin/env python3
"""
Embedding demo — turn text into vectors, compare meaning.

Uses your existing OPENROUTER_API_KEY from ~/.hermes/.env
No extra installs needed (stdlib only).
"""
import json
import math
import os
import urllib.request

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.hermes/.env"))

API_KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "openai/text-embedding-3-small"  # 1536 dims, very cheap


def embed(text: str) -> list[float]:
    """Send text, get back a vector (list of floats)."""
    req = urllib.request.Request(
        URL,
        data=json.dumps({"model": MODEL, "input": text}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.loads(r.read())
    return payload["data"][0]["embedding"]


def cosine(a: list[float], b: list[float]) -> float:
    """Similarity between two vectors: 1.0 = same meaning, 0.0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


if __name__ == "__main__":
    sentence = "i love this country because of its national football club is amazing"

    vec = embed(sentence)

    print("INPUT:", sentence)
    print(f"\nMODEL:  {MODEL}")
    print(f"DIMS:   {len(vec)}   <- the output IS this vector, nothing else")
    print(f"FIRST 8 NUMBERS: {[round(n, 4) for n in vec[:8]]}")
    print(f"MIN/MAX:         {min(vec):.4f} / {max(vec):.4f}")

    # What embeddings are FOR: comparing meaning
    print("\n--- SIMILARITY TO OTHER SENTENCES ---")
    others = [
        "this nation has a wonderful soccer team",
        "the country's football squad is fantastic",
        "i enjoy living here due to the great team",
        "quantum chromodynamics perturbative expansion",
    ]
    base = vec
    for text in others:
        print(f"{cosine(base, embed(text)):.4f}  {text}")
