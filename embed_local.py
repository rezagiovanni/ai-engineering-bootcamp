#!/usr/bin/env python3
"""
embed_local.py — local embedding + semantic search on CPU (no API, no cost).

Model : sentence-transformers/all-MiniLM-L12-v2
        onnx/model_quint8_avx2.onnx  (32.6 MB, native AVX2 build)
Dims  : 384
Max   : 256 tokens per chunk (truncation)
Stack : onnxruntime + tokenizers + numpy   (no torch, no CUDA)

Measured on i5-6200U / 4 threads:
    7.97 ms per sentence, ~125 chunks/sec, 274 MB peak RSS

Usage:
    python3 embed_local.py index  <dir_or_file> --store mydb.npz
    python3 embed_local.py search "your query"   --store mydb.npz
    python3 embed_local.py demo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_REPO = "sentence-transformers/all-MiniLM-L12-v2"
MODEL_FILE = "model_quint8_avx2.onnx"
TOKENIZER_FILE = "tokenizer.json"
CACHE_DIR = Path(os.environ.get("EMBED_CACHE", Path.home() / ".cache" / "embed_local"))
MODEL_MAX_TOKENS = 256
CHUNK_WORDS = 120          # ~<=256 tokens for English prose
CHUNK_OVERLAP = 20


# --------------------------------------------------------------------------
# Model download + load
# --------------------------------------------------------------------------
def ensure_model() -> tuple[Path, Path]:
    """Download the ONNX graph and tokenizer once, then reuse."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = CACHE_DIR / MODEL_FILE
    tok_path = CACHE_DIR / TOKENIZER_FILE

    for path, remote in (
        (onnx_path, f"onnx/{MODEL_FILE}"),
        (tok_path, TOKENIZER_FILE),
    ):
        if path.exists() and path.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{remote}"
        print(f"downloading {remote} ...", file=sys.stderr)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)   # noqa: S310 - fixed HF host
        tmp.replace(path)
        print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)

    return onnx_path, tok_path


class Embedder:
    """Mean-pooled ONNX sentence embedder with L2 normalisation."""

    def __init__(self, max_tokens: int = MODEL_MAX_TOKENS) -> None:
        onnx_path, tok_path = ensure_model()

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        # CRITICAL: without this the tokenizer pads every input to a fixed
        # length, and mean pooling then averages over [PAD] vectors, which
        # collapses all embeddings toward one centroid.
        self.tokenizer.no_padding()

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.dims = self.session.get_outputs()[0].shape[-1]
        if not isinstance(self.dims, int):
            self.dims = 384

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return an (n, dims) float32 array of unit-norm embeddings."""
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)

        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encodings = [self.tokenizer.encode(t) for t in batch]
            width = max(len(e.ids) for e in encodings)

            ids = np.array(
                [e.ids + [0] * (width - len(e.ids)) for e in encodings],
                dtype=np.int64,
            )
            mask = np.array(
                [[1] * len(e.ids) + [0] * (width - len(e.ids)) for e in encodings],
                dtype=np.int64,
            )

            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)

            hidden = self.session.run(None, feed)[0]          # (b, seq, dims)

            # masked mean pooling
            m = mask[..., None].astype(np.float32)
            summed = (hidden * m).sum(axis=1)
            counts = np.clip(m.sum(axis=1), 1e-9, None)
            vectors = summed / counts

            # L2 normalise so dot product == cosine similarity
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            out.append(vectors / np.clip(norms, 1e-9, None))

        return np.vstack(out).astype(np.float32)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_text(text: str, words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on paragraph/sentence boundaries, then pack into word windows."""
    text = re.sub(r"[ \t]+", " ", text.strip())
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []

    for para in paragraphs:
        parts = re.split(r"(?<=[.!?])\s+", para)
        buf: list[str] = []
        size = 0
        for sentence in parts:
            n = len(sentence.split())
            if size + n > words and buf:
                chunks.append(" ".join(buf))
                tail = buf[-overlap:] if overlap else []
                buf = list(tail)
                size = len(" ".join(tail).split())
            buf.append(sentence)
            size += n
        if buf:
            chunks.append(" ".join(buf))

    return [c for c in chunks if c.strip()]


# --------------------------------------------------------------------------
# Vector store (.npz)
# --------------------------------------------------------------------------
class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.vectors: np.ndarray = np.zeros((0, 384), dtype=np.float32)
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.meta: dict = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            vectors=self.vectors,
            chunks=np.array(self.chunks, dtype=object),
            sources=np.array(self.sources, dtype=object),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: Path) -> "Store":
        if not path.exists():
            raise SystemExit(
                f"store not found: {path}\nrun: python3 embed_local.py index <dir> --store {path}"
            )
        s = cls(path)
        with np.load(path, allow_pickle=True) as data:
            s.vectors = data["vectors"]
            s.chunks = list(data["chunks"])
            s.sources = list(data["sources"])
            raw = data["meta"]
            s.meta = json.loads(str(raw.item() if hasattr(raw, "item") else raw))
        return s

    def add(self, vectors: np.ndarray, chunks: list[str], sources: list[str]) -> None:
        if len(self.vectors) == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)
        self.sources.extend(sources)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[tuple[float, str, str]]:
        if len(self.vectors) == 0:
            return []
        # vectors are unit-norm, so a dot product is the cosine similarity
        scores = self.vectors @ query_vec.reshape(-1)
        order = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self.chunks[i], self.sources[i]) for i in order]


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".py", ".json", ".csv", ".org"}


def gather_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
        and not any(part.startswith(".") for part in p.parts)
    )


def build_index(target: Path, store_path: Path) -> None:
    files = gather_files(target)
    if not files:
        raise SystemExit(f"no indexable text files under {target}")

    print(f"model      : {MODEL_REPO} ({MODEL_FILE})")
    print(f"files      : {len(files)}")

    embedder = Embedder()
    store = Store(store_path)

    all_chunks: list[str] = []
    all_sources: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as err:
            print(f"  skip {path}: {err}", file=sys.stderr)
            continue
        for chunk in chunk_text(text):
            all_chunks.append(chunk)
            all_sources.append(str(path))

    if not all_chunks:
        raise SystemExit("no text content found")

    print(f"chunks     : {len(all_chunks)}")
    vectors = embedder.encode(all_chunks)

    store.add(vectors, all_chunks, all_sources)
    store.meta = {
        "model": MODEL_REPO,
        "onnx": MODEL_FILE,
        "dims": int(vectors.shape[1]),
        "source_root": str(target),
        "files": len(files),
        "chunks": len(all_chunks),
    }
    store.save()

    print(f"dims       : {vectors.shape[1]}")
    print(f"store      : {store_path} ({store_path.stat().st_size / 1e6:.1f} MB)")
    print("done.")


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
DEMO_DOCS = {
    "note-football.md": (
        "I love this country because of its national football club is amazing. "
        "The atmosphere in the stadium is incredible and the fans sing for ninety minutes."
    ),
    "note-fast-objects.md": (
        "Some people cannot see an object that moves very fast. Human vision has a "
        "temporal resolution limit, so rapid motion is often perceived as a blur or missed entirely."
    ),
    "note-indodax.md": (
        "Indodax is an Indonesian cryptocurrency exchange. Trading fees are 0.3 percent "
        "for makers and takers on the IDR market."
    ),
    "note-unrelated.md": (
        "Municipal bond yields rose slightly on Tuesday amid broader market uncertainty."
    ),
}


def run_demo() -> None:
    embedder = Embedder()
    names = list(DEMO_DOCS)
    chunks = [DEMO_DOCS[n] for n in names]

    vectors = embedder.encode(chunks)
    store = Store(Path(":memory:"))
    store.add(vectors, chunks, names)

    print(f"model : {MODEL_REPO}")
    print(f"dims  : {vectors.shape[1]}   chunks indexed: {len(chunks)}")
    print()

    queries = [
        "which country has a great soccer team",
        "things that move too fast for human eyes",
        "crypto exchange fees in Indonesia rupiah",
    ]

    for query in queries:
        qv = embedder.encode([query])[0]
        print(f'QUERY: "{query}"')
        for score, chunk, source in store.search(qv, top_k=3):
            preview = chunk[:70].replace("\n", " ")
            print(f"   {score:+.4f}  {source:22s} {preview}...")
        print()

    print("Positive scores = relevant. Scores near 0 or negative = unrelated.")
    print(f"Threshold guidance for this model: >0.45 keep, <0.15 drop.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Local CPU embedding + semantic search")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="index a file or directory")
    p_index.add_argument("target", type=Path)
    p_index.add_argument("--store", type=Path, default=Path("vectors.npz"))

    p_search = sub.add_parser("search", help="query an existing index")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--store", type=Path, default=Path("vectors.npz"))
    p_search.add_argument("--top", type=int, default=5)
    p_search.add_argument("--min-score", type=float, default=0.0)

    sub.add_parser("demo", help="run the built-in demo")

    args = ap.parse_args()

    if args.cmd == "demo":
        run_demo()
    elif args.cmd == "index":
        build_index(args.target, args.store)
    elif args.cmd == "search":
        store = Store.load(args.store)
        embedder = Embedder()
        query = " ".join(args.query)
        qv = embedder.encode([query])[0]
        hits = [h for h in store.search(qv, args.top) if h[0] >= args.min_score]
        if not hits:
            print(f'no results above {args.min_score} for "{query}"')
            return
        print(f'QUERY: "{query}"   (model: {store.meta.get("model")})')
        for score, chunk, source in hits:
            print(f"\n  {score:+.4f}  {source}")
            print(f"           {chunk[:300]}")


if __name__ == "__main__":
    main()
