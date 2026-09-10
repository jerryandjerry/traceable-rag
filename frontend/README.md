# Traceable RAG frontend

React 18 + TypeScript + Vite 8 + antd. This is the web client for the
Traceable RAG backend; see the repository root `README.md` for the whole system.

## Running

```bash
cp .env.example .env
npm install
npm run dev          # http://localhost:5181
```

The dev server port is pinned to **5181** in `vite.config.ts`, not Vite's
default 5173. It proxies `VITE_API_BASE` to `VITE_API_PROXY`, stripping the
prefix, so the backend must be up on :8000.

Start from `.env.example`; `.env` is local and ignored:

```env
VITE_API_BASE = /ai-search
VITE_API_PROXY = http://localhost:8000/
```

## Scripts

| Script | Does |
|---|---|
| `npm run dev` | dev server on :5181 |
| `npm run build` | Typecheck, build `dist/`, then enforce initial-JS and per-chunk budgets |
| `npm run preview` | serve the built `dist/` |
| `npm run lint` | eslint |
| `npx vitest run` | unit tests from `tests/unit/` |

There is no `npm test` script. See `test/README.md` for the full suite.

## Third-party bundles

The 3D graph viewer imports `react-force-graph-3d`, `three`, and
`three-spritetext` from npm. Vite bundles them with the lazy-loaded graph route;
the application does not inject runtime script tags or depend on UMD globals.
The viewer uses WebGL, so its dependency's unused optional WebGPU backend is
excluded from the bundle.
