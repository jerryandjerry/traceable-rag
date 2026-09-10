# Vendored RAGFlow

This directory contains selected third-party code derived from
[RAGFlow](https://github.com/infiniflow/ragflow), distributed under Apache-2.0.

Imports are namespaced under `visionagent.vendor.ragflow` so the code resolves
from an installed package. Narrow compatibility modules bridge RAGFlow's
interfaces to this project's configured providers and storage boundary.

| Directory | What it is |
|---|---|
| `deepdoc/` | PDF/DOCX/PPT parsing, OCR, layout and table-structure recognition |
| `rag/` | chunking strategies, the text analyzer, Elasticsearch access |
| `rag/res/` | Active DeepDoc ONNX/XGBoost models, the `huqie` trie, and dictionaries (~157 MB) |
| `9b5ad71b2ce5302211f9c61530b329a4922fc6a4` | Offline `cl100k_base.tiktoken` cache; its name is tiktoken's SHA-1 URL cache key |
| `api/` | a small compatibility shim RAGFlow's modules import |
| `conf/mapping.json` | the Elasticsearch index mapping |

The bundled DeepDoc assets come from immutable model revisions recorded in the
root `THIRD_PARTY_NOTICES.md`. Parsing never downloads or replaces models at
runtime; a missing or invalid asset fails explicitly.

**The hash-named tiktoken cache is deliberate.** `rag/utils/__init__.py` sets
`TIKTOKEN_CACHE_DIR` to this directory before loading `cl100k_base`. With the
locked tiktoken 0.14.0, the official source URL maps to
`9b5ad71b2ce5302211f9c61530b329a4922fc6a4`; the tracked bytes must retain
SHA-256 `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`.
Do not rename or regenerate this file without updating the dependency lock,
third-party notice, and golden asset manifest.

## Operational constraints

**Asset paths resolve from `__file__`, not from config.**
`api/utils/file_utils.py:get_project_base_directory()` walks two levels up from
itself and everything else is relative to that — `rag/res/deepdoc/*.onnx`,
`conf/mapping.json`, `rag/res/huqie.txt.trie`. Moving any directory *within*
`vendor/ragflow/` breaks model loading with no import error, only a missing-file
failure at first parse. `RAG_PROJECT_BASE` overrides the base if you need it.

**The repository root owns the Git LFS rules for model assets.** The upstream
HuggingFace `.gitattributes` is ignored because its broad patterns affect files
outside this directory. A clean checkout must run `git lfs pull` before parsing
documents or executing the offline golden suite.

## Excluded from tooling

`ruff` and `mypy` skip this tree (see `backend/pyproject.toml`). Lint findings
here are upstream's, not ours, and fixing them would create a diff against
upstream that has to be carried forever.
