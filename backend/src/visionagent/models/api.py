"""Pydantic request, response, and shared domain models."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import UUID4, BaseModel, ConfigDict


class ChatRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message: str
    # True requests FORCE mode; False requests AUTO mode.
    web_search: bool = False
    deep_research: bool = False
    chat_id: str | int | None = None
    attachments: list[str] | None = None


class SessionResponse(BaseModel):
    session_id: str
    session_name: str
    user_id: str
    created_at: str
    updated_at: str


class SessionCreatedResponse(BaseModel):
    """Response to POST /create_session/ (distinct shape from SessionResponse)."""

    session_id: str
    status: str
    message: str


class SessionListResponse(BaseModel):
    user_id: str
    sessions: list[SessionResponse]


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message_id: UUID4
    session_id: str
    user_question: str
    model_answer: str
    created_at: datetime
    documents: list[Any] | dict[str, Any] | None = None
    recommended_questions: list[Any] | dict[str, Any] | None = None
    think: str | None


class FilestResponse(BaseModel):
    user_id: str
    file_name: str
    created_at: str
    updated_at: str
    total_chunks: int | None = None
    process_time: float | None = None
    error: str | None = None
    """Why ingest did not fully complete, or None. A document with chunks
    but no graph is searchable and still reported here as partial."""


class DeleteFileRequest(BaseModel):
    file_name: str


class ExploreRequest(BaseModel):
    user_message: str


class DocumentResponse(BaseModel):
    document_id: str
    document_name: str
    preview: str
    create_time: int
    update_time: int


class ExploreResponse(BaseModel):
    documents: list[DocumentResponse]
    message: str
    status: str


class AddDocsRequest(BaseModel):
    document_id: list[str]


class AddDocsResponse(BaseModel):
    status: str
    message: str


# --- auth and account ------------------------------------------------------
# HTTP request bodies for authentication and account operations.


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class DeleteAccountRequest(BaseModel):
    password: str
    confirm: str = ""
