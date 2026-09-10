# Third-party notices

This file records third-party code and assets incorporated into Traceable RAG.
It does not grant a license for the project's original code.

## RAGFlow / DeepDoc

Selected parser, search, dictionary, and model assets under
`backend/src/visionagent/vendor/ragflow/` are derived from
[RAGFlow](https://github.com/infiniflow/ragflow).

The unmodified parser fixtures in `test/fixtures/ragflow/` come from
RAGFlow commit
[`c178cdd5b91c1c0adc2ee7c5451841565da59790`](https://github.com/infiniflow/ragflow/tree/c178cdd5b91c1c0adc2ee7c5451841565da59790/test/benchmark/test_docs):

- `Doc1.pdf`: SHA-256 `e9642829132ab617b158271fd841abe83d2d4a948ee01d97f3a2fc14b836570f`
- `Doc2.pdf`: SHA-256 `546bf0c9840c4ec3e6e8f77c4c26e6f1dc65fa40fbfc0d33baeba7e5bec87a55`
- `Doc3.pdf`: SHA-256 `f956e3c7204d072de501309754689a08c6db4a67d83e591b199e330f89745cc2`

Copyright 2024-2025 The InfiniFlow Authors. All Rights Reserved.

RAGFlow is distributed under the Apache License 2.0. A copy is included at
`backend/src/visionagent/vendor/ragflow/LICENSE`. This project namespaces its
imports and provides narrow compatibility adapters for its own providers and
storage boundaries; modified files therefore differ from upstream.

Bundled inference assets are pinned to these Apache-2.0 model snapshots:

- [`InfiniFlow/deepdoc`](https://huggingface.co/InfiniFlow/deepdoc/tree/9c7aa2c730d7a242d7f04cf6109a6a239aefc717), revision `9c7aa2c730d7a242d7f04cf6109a6a239aefc717`
- [`InfiniFlow/text_concat_xgb_v1.0`](https://huggingface.co/InfiniFlow/text_concat_xgb_v1.0/tree/722ed09a54f23f14fe0279ce6b74ce18e1960f54), revision `722ed09a54f23f14fe0279ce6b74ce18e1960f54`

## OpenAI tiktoken `cl100k_base` encoding

The file
`backend/src/visionagent/vendor/ragflow/9b5ad71b2ce5302211f9c61530b329a4922fc6a4`
is an offline cache of OpenAI's
[`cl100k_base.tiktoken`](https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken).
The opaque name is intentional: tiktoken 0.14.0 uses the SHA-1 of that source
URL as its cache key. The bundled file is 1,681,126 bytes with SHA-256
`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`,
which is the integrity hash declared by tiktoken. RAGFlow commit
[`c178cdd5b91c1c0adc2ee7c5451841565da59790`](https://github.com/infiniflow/ragflow/blob/c178cdd5b91c1c0adc2ee7c5451841565da59790/Dockerfile#L28)
uses the same cache placement for offline operation.

The [tiktoken project](https://github.com/openai/tiktoken/tree/0.14.0) is
distributed under the MIT License, Copyright (c) 2022 OpenAI, Shantanu Jain.
Its official sources do not state a separate license for the remotely hosted
encoding table; this notice records the table's provenance and integrity and
does not assert a separate data license.

MIT License

Copyright (c) 2022 OpenAI, Shantanu Jain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## LightRAG

The graph prompt, merge behavior, identifiers, and local graph/vector-storage
primitives in `backend/src/visionagent/service/vectorstore/graphstore/` and
`backend/src/visionagent/database/graph/lightrag_storage.py` are adapted from
[LightRAG](https://github.com/HKUDS/LightRAG).

MIT License

Copyright (c) 2025 LightRAG Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
