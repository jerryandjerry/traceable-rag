import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from visionagent.config.settings import settings
from visionagent.database.postgres.engine import get_db
from visionagent.database.postgres.repositories import AccountWriteUnavailable
from visionagent.models import RetrievedChunk
from visionagent.providers.llm import _llm

logger = logging.getLogger(__name__)

def _serialize_retrieved_chunks(
    chunks: Sequence[RetrievedChunk],
    *,
    citation_ids: Mapping[str, str] | None = None,
    related_questions: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Serialize typed evidence at the existing SSE/Postgres boundary.

    Elasticsearch field names are a storage and wire compatibility contract;
    they are deliberately reconstructed here and never leak back into the
    pipeline. A missing citation map is the casual path, whose persisted rows
    historically had no ``citation_id``.
    """
    payload: list[dict[str, Any]] = []
    for chunk in chunks:
        row: dict[str, Any] = {
            "chunk_id": chunk.id,
            "id": chunk.id,
            "content_with_weight": chunk.content,
            "docnm": chunk.doc_name or "",
            "docnm_kwd": chunk.doc_name or "",
            "doc_id": chunk.url or "",
            "kb_id": "",
            "important_kwd": [],
            "sim": chunk.score,
            "similarity": chunk.score,
            "ref_images": list(chunk.images),
            "source_type": chunk.source_type.value,
        }
        if chunk.page_num is not None:
            row["page_num"] = chunk.page_num
        if chunk.url:
            row.update({
                "url": chunk.url,
                "title": chunk.doc_name or "",
                "related_questions": list(related_questions),
                "image_urls": [],
                "video_urls": [],
            })
        if citation_ids is not None:
            row["citation_id"] = citation_ids[chunk.id]
        payload.append(row)
    return payload


def _unique(values: Sequence[str]) -> list[str]:
    """First-seen deduplication for the recommended-question frame."""
    return list(dict.fromkeys(values))


async def generate_recommended_questions(
    user_question: Any, retrieved_content: Any
) -> Any:
    """Generate follow-up questions in the query's language."""
    if not retrieved_content:
        formatted_references = "No relevant knowledge-base evidence was retrieved."
    else:
        formatted_references = "\n".join([f"[{ref['id']}] {ref['content_with_weight']}" for ref in retrieved_content])

    prompt = f"""
    Generate exactly three useful follow-up questions from the user's question
    and the retrieved evidence. Write every question in the same language as
    the user's question.

    User question:
    {user_question}

    Retrieved evidence:
    {formatted_references}

    Return only this JSON shape:
    {{
      "recommended_questions": [
        "first question",
        "second question",
        "third question"
      ]
    }}
    """
    
    response = await _llm().complete(
        prompt=prompt, model=settings.chat_model_turbo, json_object=True
    )

    if response:
        try:
            response_json = json.loads(response)
            recommended_questions = response_json.get("recommended_questions")
            logger.debug(
                "recommended questions generated count=%s",
                len(recommended_questions) if isinstance(recommended_questions, list) else 0,
            )
            return recommended_questions
        except json.JSONDecodeError:
            logger.warning("recommended-question response was not valid JSON")
            return []
    return []

async def generate_session_name(user_question: str) -> str:
    """Generate a concise session name in the query's language."""
    prompt = f"""
    Generate a concise session name that summarizes the user's question. Write
    the name in the same language as the question.

    User question:
    {user_question}

    Return only this JSON shape:
    {{
      "session_name": "concise name"
    }}
    """
    
    try:
        response = await _llm().complete(
            prompt=prompt, model=settings.chat_model_turbo, json_object=True
        )

        if response:
            try:
                response_json = json.loads(response)
                session_name = response_json.get("session_name")
                logger.debug("session name generated")
                return str(session_name or user_question)
            except json.JSONDecodeError:
                logger.warning("session-name response was not valid JSON")
                return user_question
    except Exception:
        logger.exception("session-name generation failed")
        return user_question
    return user_question


def write_chat_to_db(
    session_id: str,
    user_id: str,
    user_question: str,
    model_answer: str,
    retrieval_content: Any,
    recommended_questions: Any,
    think: Any,
) -> None:
    """Persist a turn only while its tenant-owned session remains writable."""
    db = next(get_db())
    try:
        documents_json = json.dumps(retrieval_content, ensure_ascii=False)

        inserted = db.execute(
            text(
                """
                INSERT INTO messages (session_id, user_question, model_answer, documents, recommended_questions, think )
                SELECT :session_id, :user_question, :model_answer, :documents,
                       :recommended_questions, :think
                FROM sessions AS session
                JOIN users AS account ON account.id::text = session.user_id
                WHERE session.session_id = :session_id
                  AND session.user_id = :user_id
                  AND account.deletion_requested = FALSE
                RETURNING message_id
                """
            ),
            {
                "session_id": session_id,
                "user_id": user_id,
                "user_question": user_question,
                "model_answer": model_answer,
                "documents": documents_json,
                "recommended_questions": recommended_questions,
                "think": think,
            }
        ).fetchone()
        if inserted is None:
            raise AccountWriteUnavailable("chat write lost its account/session fence")
        db.commit()
        logger.info("chat turn persisted")
    except Exception as exc:
        db.rollback()
        raise RuntimeError("chat persistence failed") from exc
    finally:
        db.close()

def _get_session_name(session_id: str) -> str | None:
    """Blocking PostgreSQL read kept behind the answer slot's DB boundary."""
    db = next(get_db())
    try:
        row = db.execute(
            text("SELECT session_name FROM sessions WHERE session_id = :session_id"),
            {"session_id": session_id},
        ).fetchone()
        return str(row[0]) if row and row[0] is not None else None
    except SQLAlchemyError:
        logger.exception(
            "session-name lookup failed",
            extra={"session_id": session_id},
        )
        return None
    finally:
        db.close()


async def check_and_update_session_name(
    session_id: str, question: str, user_id: str
) -> None:
    """
    Check if session needs a name update and only update if session_name is NULL or empty
    """
    existing_name = await asyncio.to_thread(_get_session_name, session_id)
    if existing_name and existing_name.strip():
        logger.debug("session already named", extra={"session_id": session_id})
        return
    await update_session_name(session_id, question, user_id)


def _store_session_name(
    session_id: str, user_id: str, session_name: str
) -> None:
    """Persist a generated name only for the existing, owned session."""
    db = next(get_db())
    try:
        result = db.execute(
            text(
                """
                UPDATE sessions
                SET session_name = :session_name, updated_at = NOW()
                WHERE session_id = :session_id
                  AND user_id = :user_id
                  AND (session_name IS NULL OR session_name = '')
                """
            ),
            {
                "session_id": session_id,
                "user_id": user_id,
                "session_name": session_name,
            },
        )
        if result.rowcount == 0:
            # Session creation is its own authorized workflow. In particular,
            # never recreate a session that account deletion just removed.
            logger.info("session name update skipped")
        else:
            logger.info("session name updated")
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise RuntimeError("session naming persistence failed") from exc
    finally:
        db.close()


async def update_session_name(session_id: str, question: str, user_id: str) -> None:
    """Name an existing owned session when its name is still empty."""
    if not question:
        logger.warning(
            "cannot name session without a question",
            extra={"session_id": session_id},
        )
        return
    session_name = await generate_session_name(question)
    logger.debug("updating session name", extra={"session_id": session_id})
    await asyncio.to_thread(_store_session_name, session_id, user_id, session_name)


def _failure_frame(what: str, run_id: str) -> str:
    """The error frame a client sees. The exception itself goes to the log.

    Provider, host, path, and SQL details stay in logs because the frontend
    renders this frame as assistant-visible content.
    """
    logger.exception(
        "answer generation failed",
        extra={"run_id": run_id, "answer_type": what},
    )
    return "event: error\ndata: " + json.dumps(
        {
            "role": "error",
            "content": f"The {what} could not be completed. Reference: {run_id}",
            "run_id": run_id,
        }
    ) + "\n\n"


async def _persist(
    session_id: str,
    question: str,
    model_answer: str,
    documents: Any,
    related_questions: Any,
    think: Any,
    user_id: str,
    *,
    run_id: str,
) -> list[str]:
    """Write the turn and name the session; an error frame if that failed.

    Returns nothing on success and one terminal error frame on failure, so
    the caller yields it instead of [DONE]: the user is not told the turn
    completed when history does not have it.
    """
    try:
        await asyncio.to_thread(
            write_chat_to_db,
            session_id,
            user_id,
            question,
            model_answer,
            documents,
            related_questions,
            think,
        )
        await check_and_update_session_name(session_id, question, user_id)
    except Exception:
        logger.exception(
            "chat turn persistence failed",
            extra={"run_id": run_id, "session_id": session_id},
        )
        return ["event: error\ndata: " + json.dumps(
            {
                "role": "error",
                "content": f"The answer was generated but could not be saved. Reference: {run_id}",
                "run_id": run_id,
            }
        ) + "\n\n"]
    return []


async def casual_chat_completion(
    session_id: str,
    question: str,
    user_id: str,
    final_prompt: str,
    web_context: list[RetrievedChunk] | None = None,
    *,
    run_id: str,
    related_questions: list[str] | None = None,
) -> AsyncIterator[str]:
    """Stream a casual answer and persist it before reporting completion."""

    try:
        model_answer = ""
        async for content, _reasoning in _llm().stream(
            prompt=final_prompt, model=settings.chat_model_turbo
        ):
            if not content:
                continue
            model_answer += content
            message = {
                "role": "assistant",
                "content": content,
                "thinking": False,
            }
            json_message = json.dumps(message)
            yield f"event: message\ndata: {json_message}\n\n"

        # Finalise on stream exhaustion rather than finish_reason == "stop";
        # length and content-filter stops still need persistence and [DONE].
        #
        # The write comes BEFORE the terminal frame. A client that closes on
        # [DONE] stops this generator being advanced, so anything after the
        # last yield may never run -- and a turn the user was told completed
        # would be missing from history. If the write fails, the terminal
        # frame is an error, never a success.
        documents = _serialize_retrieved_chunks(
            web_context or [],
            related_questions=related_questions or [],
        )
        for frame in await _persist(
            session_id,
            question,
            model_answer,
            documents,
            [],
            "",
            user_id,
            run_id=run_id,
        ):
            yield frame
            return
        yield "event: end\ndata: [DONE]\n\n"

    except Exception:
        yield _failure_frame("casual answer", run_id)

async def get_chat_completion(
    session_id: str,
    question: str,
    context_list: list[RetrievedChunk],
    user_id: str,
    final_prompt: str,
    related_questions: list[str],
    snippets: list[RetrievedChunk],
    *,
    run_id: str,
    citation_ids: Mapping[str, str],
    media: dict[str, list[dict[str, Any]]] | None = None,
) -> AsyncIterator[str]:
    """Stream a grounded answer and its evidence frames.

    ``media`` is ``{"images": [...], "videos": [...]}`` from an authorized
    web-tool execution, or ``None`` when that tool did not run.
    """

    try:
        context_payload = _serialize_retrieved_chunks(
            context_list,
            citation_ids=citation_ids,
            related_questions=related_questions,
        )
        snippets_payload = _serialize_retrieved_chunks(
            snippets,
            citation_ids=citation_ids,
            related_questions=related_questions,
        )
        recommended_questions = _unique(related_questions)

        message: dict[str, Any] = {
            "documents": context_payload,
        }
        json_message = json.dumps(message)
        yield f"event: message\ndata: {json_message}\n\n"

        message = {
            "web_search": snippets_payload,
        }
        json_message = json.dumps(message)
        yield f"event: message\ndata: {json_message}\n\n"

        model_answer = ""
        think = ""
        async for content, _reasoning in _llm().stream(
            prompt=final_prompt, model=settings.chat_model, temperature=0
        ):
            if not content:
                continue
            model_answer += content
            message = {
                "role": "assistant",
                "content": content,
                "thinking": False,
            }
            json_message = json.dumps(message)
            yield f"event: message\ndata: {json_message}\n\n"

        # Finalise on stream exhaustion so every provider stop reason persists
        # the answer, emits citations, and terminates the client stream.
        message = {
            "recommended_questions": recommended_questions,
        }
        json_message = json.dumps(message)
        yield f"event: message\ndata: {json_message}\n\n"

        # Return media only when the authorized web tool supplied it. The wire
        # shape remains what chat/component/result.tsx consumes.
        media = media or {}
        message = {
            "image_results": {"images": list(media.get("images") or [])},
        }
        json_message = json.dumps(message)
        yield f"event: message\ndata: {json_message}\n\n"
        message = {
            "video_results": {"videos": list(media.get("videos") or [])},
        }
        json_message = json.dumps(message)
        yield f"event: message\ndata: {json_message}\n\n"

        ref_images = [image for context in context_list for image in context.images]
        logger.debug(
            "answer references ready",
            extra={
                "session_id": session_id,
                "citation_count": len(context_payload),
                "image_count": len(ref_images),
            },
        )

        message = {
            "citations": context_payload,
            "ref_images": ref_images,
        }
        json_message = json.dumps(message)
        yield f"event: message\ndata: {json_message}\n\n"

        # Persist before the terminal frame, for the reason given in
        # casual_chat_completion.
        logger.debug(
            "answer stream complete",
            extra={
                "session_id": session_id,
                "character_count": len(model_answer),
            },
        )
        for frame in await _persist(
            session_id,
            question,
            model_answer,
            context_payload,
            recommended_questions,
            think,
            user_id,
            run_id=run_id,
        ):
            yield frame
            return

        yield "event: end\ndata: [DONE]\n\n"

    except Exception:
        yield _failure_frame("answer", run_id)
