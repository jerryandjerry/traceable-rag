"""The Pydantic contracts: domain vocabulary and every HTTP body.

SQLAlchemy tables are in `visionagent.database.postgres`.

One deliberate exception: a slot's Protocol may define its own value type
beside it (`vectorstore.VectorHit`, `websearch.WebResult`), because that type
is part of the interface's signature and travels with whoever implements it.
Everything else belongs here.

Re-exported so callers write `from visionagent.models import RetrievedChunk`
regardless of which file a contract happens to live in.
"""
from visionagent.models.answer import Answer, AnswerChunk
from visionagent.models.api import (
    AddDocsRequest,
    AddDocsResponse,
    ChangePasswordRequest,
    ChatRequest,
    DeleteAccountRequest,
    DeleteFileRequest,
    DocumentResponse,
    ExploreRequest,
    ExploreResponse,
    FilestResponse,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    SessionCreatedResponse,
    SessionListResponse,
    SessionResponse,
)
from visionagent.models.chunk import (
    Citation,
    ParsedChunk,
    RetrievedChunk,
    SourceType,
)
from visionagent.models.context import (
    AuthorizationScope,
    DeleteAccountCommand,
    IngestJob,
    JobIdentity,
    QueryJob,
    ToolContext,
    TurnOptions,
    WebSearchMode,
)
from visionagent.models.graph import Entity, GraphHit, Relation
from visionagent.models.query import (
    Evaluation,
    Intent,
    Plan,
    Scenario,
    ToolCall,
    ToolName,
    ToolResult,
)
from visionagent.models.run import AgentState, IngestRun
from visionagent.models.trace import Evidence, EvidenceKind, StepStatus, TraceStep
from visionagent.models.upload import (
    StagedUpload,
    UploadItemState,
    UploadJobRecord,
    UploadJobState,
    UploadProgressEvent,
)

__all__ = [
    # chunk
    "Citation", "ParsedChunk", "RetrievedChunk", "SourceType",
    # query
    "Evaluation", "Intent", "Plan", "Scenario", "ToolCall", "ToolContext", "ToolName",
    "ToolResult",
    # answer
    "Answer", "AnswerChunk",
    # graph
    "Entity", "GraphHit", "Relation",
    # trace
    "Evidence", "EvidenceKind", "StepStatus", "TraceStep",
    # run
    "IngestRun", "AgentState", "AuthorizationScope", "DeleteAccountCommand",
    "IngestJob", "JobIdentity", "QueryJob", "TurnOptions", "WebSearchMode",
    "StagedUpload", "UploadItemState", "UploadJobRecord", "UploadJobState",
    "UploadProgressEvent",
    # api
    "AddDocsRequest", "AddDocsResponse", "ChangePasswordRequest", "ChatRequest",
    "DeleteAccountRequest", "DeleteFileRequest", "DocumentResponse",
    "ExploreRequest", "ExploreResponse", "FilestResponse", "LoginRequest",
    "MessageResponse", "RegisterRequest", "SessionCreatedResponse",
    "SessionListResponse", "SessionResponse",
]
