"""Exercises the core SQLAlchemy models against an in-memory SQLite engine.

SQLite is used here purely as a lightweight, dependency-free way to exercise
column definitions/relationships for unit coverage — the models are also
verified against a real PostgreSQL instance via
`uv run alembic upgrade head`/`downgrade base` (see backend README), which is
the actual source of truth for schema correctness across DB_ENGINE targets.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import (
    Chunk,
    Citation,
    Conversation,
    Document,
    Element,
    ErrorCode,
    IngestionJob,
    Message,
    Page,
)


def make_sqlite_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_core_tables_roundtrip() -> None:
    session = make_sqlite_session()

    document = Document(
        title="PUHY-P Service Manual",
        manufacturer="Mitsubishi Electric",
        model_family="PUHY-P_YKB-A1",
        filename="puhy-p.pdf",
        source_hash="abc123",
        status="ready",
        page_count=400,
    )
    session.add(document)
    session.flush()

    page = Page(document_id=document.id, page_number=1, text_layer_present=True)
    session.add(page)
    session.flush()

    parent_element = Element(
        document_id=document.id,
        page_id=page.id,
        element_type="paragraph",
        text="Press the button to reset.",
        source_hash="elem-hash-1",
    )
    session.add(parent_element)
    session.flush()

    child_icon = Element(
        document_id=document.id,
        page_id=page.id,
        element_type="icon",
        text=None,
        image_uri="s3://bucket/icon.png",
        source_hash="elem-hash-2",
        parent_id=parent_element.id,
    )
    session.add(child_icon)
    session.flush()

    chunk = Chunk(
        document_id=document.id,
        chunk_type="text",
        content_text="Press the button to reset.",
        content_hash="chunk-hash-1",
        element_ids=[parent_element.id, child_icon.id],
    )
    session.add(chunk)
    session.flush()

    child_chunk = Chunk(
        document_id=document.id,
        chunk_type="entity",
        content_text="Reset button",
        content_hash="chunk-hash-2",
        parent_chunk_id=chunk.id,
    )
    session.add(child_chunk)

    error_code = ErrorCode(
        document_id=document.id,
        code="P8",
        model_family="PUHY-P_YKB-A1",
        description="Water flow abnormality",
        element_id=parent_element.id,
    )
    session.add(error_code)

    ingestion_job = IngestionJob(
        document_id=document.id,
        stage="docling",
        status="done",
        stage_metrics={"duration_ms": 1234, "pages_processed": 400},
    )
    session.add(ingestion_job)

    conversation = Conversation(title="Troubleshooting P8")
    session.add(conversation)
    session.flush()

    message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content="Check the water flow sensor.",
        structured_answer={"answer": "Check the water flow sensor."},
    )
    session.add(message)
    session.flush()

    citation = Citation(
        message_id=message.id,
        document_id=document.id,
        page=page.page_number,
        element_id=parent_element.id,
        quote="Press the button to reset.",
        rank=1,
    )
    session.add(citation)
    session.commit()

    assert document.pages[0].id == page.id
    assert parent_element.children[0].id == child_icon.id
    assert child_icon.parent_id == parent_element.id
    assert chunk.children[0].id == child_chunk.id
    assert conversation.messages[0].id == message.id
    assert message.citations[0].id == citation.id

    session.close()
