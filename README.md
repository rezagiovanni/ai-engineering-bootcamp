# ai-engineering-bootcamp

Working repo for the AI engineering bootcamp — local-first, low-cost tooling.
Everything here runs on CPU; no GPU, no paid API required for the core path.

## Contents

| File | What it demonstrates | Stack |
|---|---|---|
| `embed_local.py` | Local embedding + semantic search over a directory, with chunking and a persistent vector store | `onnxruntime`, `tokenizers`, `numpy` |
| `embedding_demo.py` | Hosted embedding via Nous Portal, batched in one request, prints real token cost | stdlib only |
| `embed_demo.py` | Hosted embedding via OpenRouter, cosine similarity between sentences | stdlib only |
| `et.py` | Minimal HTTP + JSON exercise | `requests` |

### `embed_local.py`

Self-contained CPU embedding pipeline. Downloads a quantised ONNX model once
(`sentence-transformers/all-MiniLM-L12-v2`, `model_quint8_avx2.onnx`, 32.6 MB,
384 dims, 256-token truncation), then indexes and queries text.

```bash
python3 embed_local.py demo
python3 embed_local.py index  <dir_or_file> --store mydb.npz
python3 embed_local.py search "your query"  --store mydb.npz
```

Reference measurement on an i5-6200U (4 threads): ~8 ms per sentence,
~125 chunks/sec, 274 MB peak RSS.

Two details worth keeping in mind if you refactor it:

- The tokenizer must run with `no_padding()`. With padding left on, mean pooling
  averages over `[PAD]` vectors and every embedding drifts toward one centroid.
- Vectors are L2-normalised on write, so search is a plain dot product.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install onnxruntime tokenizers numpy requests python-dotenv
```

`embedding_demo.py` reads the Nous Portal key Hermes already stores in
`~/.hermes/auth.json`; `embed_demo.py` reads `OPENROUTER_API_KEY` from
`~/.hermes/.env`.

## Excluded from version control

`.gitignore` keeps out the virtualenv (6.4 GB), local credentials, caches, and
one tutorial clone (`Hello-World/`, an untouched checkout of
`octocat/Hello-World` with its own `.git`, used as an HTTP/git exercise).

Before any commit that touches configuration, confirm with:

```bash
git check-ignore -v key.txt
git diff --cached --name-only   # eyeball what is actually staged
```
