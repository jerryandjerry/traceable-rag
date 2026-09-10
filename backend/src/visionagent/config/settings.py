from __future__ import annotations

import logging
import os
import secrets
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - fallback if dotenv not present at runtime
    load_dotenv = None  # type: ignore


def _load_env_files() -> None:
    """Load the first available local dotenv file without overriding the process.

    Precedence is the existing process environment, then the first file found
    in this order: ``backend/.env``, repository ``.env``, package ``.env``,
    and the current working directory's ``.env``.
    """
    if load_dotenv is None:
        return

    resolved = Path(__file__).resolve()
    package_dir = resolved.parents[1]  # .../backend/src/visionagent
    backend_dir = resolved.parents[3]  # .../backend
    repo_root = resolved.parents[4]    # repo root

    candidate_paths = [
        backend_dir / ".env",
        repo_root / ".env",
        package_dir / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in candidate_paths:
        try:
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=False)
                break
        except Exception:
            # Do not crash app if dotenv load fails
            continue


_load_env_files()


@dataclass(frozen=True)
class Settings:
    """Validated runtime configuration from environment variables and dotenv."""

    # --- deployment ------------------------------------------------------
    app_env: str
    """"production" unless APP_ENV says otherwise. Only "development" and
    "test" relax the checks below; any other value is treated as production."""

    jwt_secret_key: str
    """Bearer-token signing key. Production requires at least 32 characters."""

    allowed_tools: tuple[str, ...]
    """The retrieval tools server policy grants an authenticated caller,
    as the ToolName values. Parsed and validated at API start-up, so an
    unknown name refuses to start rather than silently granting nothing or
    everything."""

    # Provider API keys; required only by the selected hosted providers.
    dashscope_api_key: str | None
    serper_api_key: str | None

    # Provider base URLs (required)
    dashscope_base_url: str

    # Model selections (required)
    embedding_model: str
    embedding_dimensions: int
    rerank_model: str
    ner_model:str
    chat_model: str
    chat_model_turbo: str

    # Mutable runtime state. Paths are absolute and share one mountable root.
    state_dir: Path     # the root
    storage_dir: Path   # <state>/uploads   -- user files and session context
    graph_dir: Path     # <state>/graph     -- graph_*.graphml, vdb_*.json

    # --- retrieval -------------------------------------------------------
    max_question_chars: int
    """Longest question the API accepts. A bound on what reaches the model
    and the prompt log, not a UX limit."""

    retrieval_top_k: int
    """Chunks fetched per tool before reranking."""

    rerank_top_n: int
    """Chunks kept after reranking, and fed to the answer prompt."""

    vector_similarity_weight: float
    """Vector vs BM25 balance in the Elasticsearch hybrid search. 0 is pure
    keyword, 1 pure vector."""

    web_search_results: int
    """Hits requested from the web search provider, before re-ranking."""

    web_search_top_k: int
    """Web snippets kept after re-ranking, and fed to the answer prompt."""

    # --- agent -----------------------------------------------------------
    max_rounds: int
    """Retrieval rounds per turn, counting the first. 2 means one initial
    search and at most one reflection round; 1 disables reflection."""

    sufficient_threshold: float
    """Reflection runs when the evaluator's score is at or below this."""

    tool_timeout_s: float
    """Per-tool budget. gather waits for its slowest member, so without this
    one hung provider holds the whole round."""

    turn_timeout_s: float
    """Whole-turn budget for the retrieval loop, on top of the per-tool one.
    Two rounds of four tools each under their own timeout can still add up to
    longer than any client waits."""

    parse_timeout_s: float
    """Per-document parse budget in the ingest worker."""

    max_upload_files: int
    """Files accepted in one upload request. The frontend picker has its own
    limit; this is the one a direct HTTP caller meets."""

    max_upload_bytes: int
    """Size accepted per uploaded file. The request spool is scanned in
    chunks and refused once exceeded; it is never joined into one bytes value."""

    max_upload_total_bytes: int
    """Aggregate file-content bytes accepted in one multipart upload.

    This is independent of the per-file and file-count limits. It bounds the
    request-scoped spool and prevents a valid batch from multiplying the
    per-file allowance by ``max_upload_files``.
    """

    max_concurrent_uploads: int
    """Upload bodies admitted concurrently by each API worker. Excess
    requests are rejected before their body is read or spooled."""

    max_upload_workers: int
    """Live ingest child reservations allowed per runtime node. PostgreSQL
    serializes this capacity across API processes sharing the node id."""

    max_upload_attempts: int
    """Maximum durable worker incarnations before an upload is dead-lettered."""

    upload_retry_backoff_s: float
    """Base delay for exponential recovery backoff."""

    # --- graphrag --------------------------------------------------------
    graph_embedding_dimensions: int
    """Width of the entity/relation vectors. Defaults to `embedding_dimensions`.

    Stored graph vectors have a fixed width. A width change requires
    ``scripts/migrate_graph_embeddings.py``; a provider or model change at the
    same width requires the same command with ``--force``.
    """

    graph_entity_top_k: int
    graph_relation_top_k: int

    @property
    def is_dashscope_configured(self) -> bool:
        return bool(self.dashscope_api_key)

    @property
    def is_serper_configured(self) -> bool:
        return bool(self.serper_api_key)

_WEAK_SECRETS = {"default_secret_key", "secret", "changeme", "password", "jwt_secret_key"}


def _read_development_secret(keyfile: Path) -> str | None:
    """Read one safely-published local key without following filesystem links."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(keyfile, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"development JWT secret cannot be opened safely: {keyfile}") from exc

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"development JWT secret must be a regular file: {keyfile}")
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise RuntimeError(f"development JWT secret has the wrong owner: {keyfile}")
        # Repair permissions through the already-open, no-follow descriptor so
        # an older development file cannot remain group/world readable.
        os.fchmod(descriptor, 0o600)
        raw = os.read(descriptor, 4097)
        if len(raw) > 4096:
            raise RuntimeError(f"development JWT secret is unexpectedly large: {keyfile}")
    finally:
        os.close(descriptor)

    try:
        kept = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"development JWT secret is not valid UTF-8: {keyfile}") from exc
    if not kept or kept.lower() in _WEAK_SECRETS or len(kept) < 32:
        # Never silently rotate an existing key: that would invalidate every
        # local session and concurrent workers could briefly sign differently.
        raise RuntimeError(
            f"development JWT secret is corrupt or too weak; remove or replace {keyfile}"
        )
    return kept


def _create_development_secret(keyfile: Path) -> tuple[str, bool]:
    """Atomically publish a complete mode-0600 key; return (key, created)."""
    keyfile.parent.mkdir(parents=True, exist_ok=True)
    generated = secrets.token_hex(32)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=keyfile.parent,
        prefix=".jwt_secret.",
        suffix=".tmp",
    )
    try:
        os.fchmod(descriptor, 0o600)
        encoded = generated.encode("utf-8")
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("development JWT secret write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1

        try:
            # A hard-link publish is create-if-absent and exposes only the
            # already-complete inode. Concurrent workers therefore either win
            # with this value or read the winner; nobody can observe an empty
            # O_EXCL target while it is still being written.
            os.link(temporary_name, keyfile, follow_symlinks=False)
            created = True
        except FileExistsError:
            created = False
        finally:
            os.unlink(temporary_name)

        directory_descriptor = os.open(
            keyfile.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

        if created:
            return generated, True
        winner = _read_development_secret(keyfile)
        if winner is None:  # A concurrent external deletion; fail closed.
            raise RuntimeError(f"development JWT secret disappeared during creation: {keyfile}")
        return winner, False
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _jwt_secret(app_env: str, state_dir: Path) -> str:
    """The signing secret, or refuse to start.

    Missing or weak is fatal in production. In development a random secret is
    generated once and kept under the state directory (gitignored, mode 0600),
    so a local checkout runs without a committed secret and a code reload does
    not sign everyone out -- uvicorn --reload restarts the process on every
    save, and a per-process secret made each save a forced logout and turned
    every open session's next request into an error. In test it is random per
    process: nothing there outlives the run.
    """
    raw = (os.getenv("JWT_SECRET_KEY") or "").strip()
    weak = not raw or raw.lower() in _WEAK_SECRETS or len(raw) < 32
    if not weak:
        return raw
    if app_env == "test":
        return secrets.token_hex(32)
    if app_env == "development":
        keyfile = state_dir / "jwt_secret.dev"
        kept = _read_development_secret(keyfile)
        if kept is not None:
            return kept
        generated, created = _create_development_secret(keyfile)
        if created:
            logging.getLogger(__name__).warning(
                "JWT_SECRET_KEY is %s; APP_ENV=development so a generated secret is kept at %s",
                "missing" if not raw else "too weak", keyfile,
            )
        return generated
    raise RuntimeError(
        "JWT_SECRET_KEY is missing or too weak (need 32+ characters). Set it in "
        "the environment, or set APP_ENV=development for a local run."
    )


def load_settings() -> Settings:
    """Construct settings from environment variables and documented defaults."""
    app_env = (os.getenv("APP_ENV") or "production").strip().lower()

    # Secrets (can be empty at startup)
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    serper_api_key = os.getenv("SERPER_API_KEY")

    # Required strings
    dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    rerank_model = os.getenv("RERANK_MODEL")
    ner_model = os.getenv("CHAT_MODEL_TURBO")
    chat_model = os.getenv("CHAT_MODEL")
    chat_model_turbo = os.getenv("CHAT_MODEL_TURBO")

    missing = [
        name for name, val in [
            ("DASHSCOPE_BASE_URL", dashscope_base_url),
            ("EMBEDDING_MODEL", embedding_model),
            ("RERANK_MODEL", rerank_model),
            ("CHAT_MODEL", chat_model),
            ("CHAT_MODEL_TURBO", chat_model_turbo),
        ] if val is None or str(val).strip() == ""
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    # Past this point every required variable is a str. Asserting it rather
    # than leaving the reader (and the type checker) to re-derive it from the
    # `missing` list above.
    assert dashscope_base_url and embedding_model and rerank_model
    assert ner_model and chat_model and chat_model_turbo

    def _int(name: str, default: int) -> int:
        raw = os.getenv(name)
        return int(raw) if raw else default

    def _positive_int(name: str, default: int) -> int:
        value = _int(name, default)
        if value <= 0:
            raise RuntimeError(f"{name} must be a positive integer")
        return value

    def _float(name: str, default: float) -> float:
        raw = os.getenv(name)
        return float(raw) if raw else default

    def _positive_float(name: str, default: float) -> float:
        value = _float(name, default)
        if value <= 0:
            raise RuntimeError(f"{name} must be positive")
        return value

    # Runtime directories. Default to backend/, beside the package rather than
    # inside it, so an install never writes into its own source tree.
    backend_dir = Path(__file__).resolve().parents[3]
    state_dir = Path(os.getenv("STATE_DIR") or backend_dir / "var")
    storage_dir = Path(os.getenv("STORAGE_DIR") or state_dir / "uploads")
    graph_dir = Path(os.getenv("GRAPH_DIR") or state_dir / "graph")
    for d in (state_dir, storage_dir, graph_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Required integer
    embedding_dim_raw = os.getenv("EMBEDDING_DIM")
    if embedding_dim_raw is None or str(embedding_dim_raw).strip() == "":
        raise RuntimeError("Missing required environment variable: EMBEDDING_DIM")
    try:
        embedding_dim = int(embedding_dim_raw)
    except ValueError as exc:
        raise RuntimeError("EMBEDDING_DIM must be an integer") from exc

    allowed_tools = tuple(
        name.strip() for name in
        (os.getenv("ALLOWED_TOOLS") or "RAG,GraphRAG,web_search,LLM").split(",")
        if name.strip()
    )

    return Settings(
        app_env=app_env,
        jwt_secret_key=_jwt_secret(app_env, state_dir),
        allowed_tools=allowed_tools,
        dashscope_api_key=dashscope_api_key,
        serper_api_key=serper_api_key,
        dashscope_base_url=dashscope_base_url,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dim,
        rerank_model=rerank_model,
        ner_model = ner_model,
        chat_model=chat_model,
        chat_model_turbo=chat_model_turbo,
        state_dir=state_dir,
        storage_dir=storage_dir,
        graph_dir=graph_dir,
        max_question_chars=_int("MAX_QUESTION_CHARS", 4000),
        retrieval_top_k=_int("RETRIEVAL_TOP_K", 5),
        rerank_top_n=_int("RERANK_TOP_N", 5),
        vector_similarity_weight=_float("VECTOR_SIMILARITY_WEIGHT", 0.6),
        web_search_results=_int("WEB_SEARCH_RESULTS", 10),
        web_search_top_k=_int("WEB_SEARCH_TOP_K", 3),
        max_rounds=max(1, _int("MAX_ROUNDS", 2)),
        sufficient_threshold=_float("SUFFICIENT_THRESHOLD", 0.5),
        tool_timeout_s=_float("TOOL_TIMEOUT_S", 120.0),
        turn_timeout_s=_float("TURN_TIMEOUT_S", 600.0),
        parse_timeout_s=_positive_float("PARSE_TIMEOUT_S", 1800.0),
        max_upload_files=_positive_int("MAX_UPLOAD_FILES", 10),
        max_upload_bytes=_positive_int("MAX_UPLOAD_BYTES", 500 * 1024 * 1024),
        max_upload_total_bytes=_positive_int(
            "MAX_UPLOAD_TOTAL_BYTES", 500 * 1024 * 1024
        ),
        max_concurrent_uploads=_positive_int("MAX_CONCURRENT_UPLOADS", 2),
        max_upload_workers=_positive_int("MAX_UPLOAD_WORKERS", 2),
        max_upload_attempts=_positive_int("MAX_UPLOAD_ATTEMPTS", 3),
        upload_retry_backoff_s=_positive_float("UPLOAD_RETRY_BACKOFF_S", 5.0),
        graph_embedding_dimensions=_int("GRAPH_EMBEDDING_DIM", 0) or embedding_dim,
        graph_entity_top_k=_int("GRAPH_ENTITY_TOP_K", 5),
        graph_relation_top_k=_int("GRAPH_RELATION_TOP_K", 5),
    )

# Module-level instance for convenient imports: `from ...config.settings import settings`
settings: Settings = load_settings()
