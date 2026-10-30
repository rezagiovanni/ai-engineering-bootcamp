#!/usr/bin/env python3
"""
embedding_demo.py — cheapest embedding model on the Nous Portal.

Model : perplexity/pplx-embed-v1-0.6b
Cost  : $0.004 per 1M tokens  (cheapest of the 37 embedding models on Portal)
Dims  : 1024
Ctx   : 32,000 tokens
Auth  : reuses your existing Hermes Nous Portal agent key — no new signup

What it does:
  1. embeds your sentence -> prints the raw vector
  2. embeds a few comparison sentences -> prints cosine similarity
  3. shows real token cost from the API response

Run: python3 embedding_demo.py
"""

import json
import math
import os
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
AUTH_FILE = os.path.expanduser("~/.hermes/auth.json")
MODEL = "perplexity/pplx-embed-v1-0.6b"   # $0.004 / 1M tokens — cheapest available

TARGET = "i love this country because of its national football club is amazing"

COMPARE = [
    "this country has an bad football club, that's why its people really dont love this country",
    "this nation has a wonderful soccer team",
    "the country's football squad is fantastic",
    "i enjoy living here due to the great team",
    "the referee awarded a penalty in the final minute",
    "quantum chromodynamics perturbative expansion",
]


# --------------------------------------------------------------------------
# Auth — read the Portal agent key Hermes already stores
# --------------------------------------------------------------------------
def load_credentials() -> tuple[str, str]:
    """Return (base_url, api_key) from the Hermes auth store."""
    if not os.path.exists(AUTH_FILE):
        raise SystemExit(
            f"Auth store not found: {AUTH_FILE}\n"
            "Run `hermes setup` first, or set NOUS_API_KEY / NOUS_BASE_URL env vars."
        )

    with open(AUTH_FILE) as fh:
        providers = json.load(fh).get("providers", {})

    nous = providers.get("nous") or {}
    key = nous.get("agent_key") or os.environ.get("NOUS_API_KEY")
    base = nous.get("inference_base_url") or os.environ.get(
        "NOUS_BASE_URL", "https://inference-api.nousresearch.com/v1"
    )

    if not key:
        raise SystemExit("No Nous Portal agent key found. Run `hermes setup`.")
    return base.rstrip("/"), key


# --------------------------------------------------------------------------
# Core: text -> vector
# --------------------------------------------------------------------------
def embed(texts: list[str], base: str, key: str, model: str = MODEL,
          retries: int = 3) -> tuple[list[list[float]], dict]:
    """
    Embed one or more strings. Returns (vectors, usage).

    The API accepts a list, so batching many chunks in ONE request is far
    cheaper and faster than one call per string.
    """
    payload = json.dumps({"model": model, "input": texts}).encode()

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            f"{base}/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
            # API returns items with an "index" — sort to keep input order
            items = sorted(body["data"], key=lambda d: d.get("index", 0))
            return [it["embedding"] for it in items], body.get("usage", {})

        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")[:300]
            if err.code in (429, 500, 502, 503) and attempt < retries:
                time.sleep(2 ** attempt)          # backoff
                continue
            raise SystemExit(f"HTTP {err.code}: {detail}") from err
        except urllib.error.URLError as err:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise SystemExit(f"Network error: {err.reason}") from err

    raise SystemExit("Embedding failed after retries.")


# --------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------
def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite."""
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    base, key = load_credentials()

    # ---- 1. embed target + all comparison sentences in ONE request ----
    batch = [TARGET, *COMPARE]
    vectors, usage = embed(batch, base, key)

    target_vec = vectors[0]
    compare_vecs = vectors[1:]

    tokens = usage.get("prompt_tokens", 0)
    cost = usage.get("cost", 0.0)

    # ---- 2. show what the output actually is ----
    print(f"MODEL   : {MODEL}")
    print(f"DIMS    : {len(target_vec)}")
    print(f"TOKENS  : {tokens} (one batched request)")
    print(f"COST    : ${cost:.10f}  (${cost / max(tokens, 1) * 1_000_000:.4f} per 1M tokens)")
    print()
    print(f"INPUT   : {TARGET!r}")
    print(f"VECTOR  : [{', '.join(f'{x:+.4f}' for x in target_vec[:10])}, ...]")
    print(f"RANGE   : min={min(target_vec):+.4f}  max={max(target_vec):+.4f}")
    print()
    print("The 'output' of an embedding is the vector above — numbers, not text.")
    print("Its value is in comparing vectors. Cosine similarity:")

    # ---- 3. compare meaning ----
    print()
    scored = sorted(
        ((cosine_similarity(target_vec, v), text)
         for v, text in zip(compare_vecs, COMPARE)),
        reverse=True,
    )
    for score, text in scored:
        bar = "#" * max(0, int(score * 40))
        print(f"  {score:+.4f}  {bar:<40}  {text}")

    print()
    best = scored[0]
    worst = scored[-1]
    print(f"Most similar : {best[0]:.4f}  -> {best[1]!r}")
    print(f"Least similar: {worst[0]:.4f}  -> {worst[1]!r}")
    print()
    print("Note how 'soccer team' / 'football squad' score high despite sharing")
    print("almost no words with the input — that is the entire point of embeddings.")


if __name__ == "__main__":
    main()
