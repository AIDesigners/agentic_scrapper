from datetime import datetime, timezone
import os
import psycopg
from pgvector.psycopg import register_vector

DB_DSN = os.getenv(
    "RAG_DB_DSN",
    "postgresql://postgres@localhost:5432/rag_db",
)
EMBEDDING_DIM = 1024  # Match your embedding model output dimension

DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;

CREATE TABLE IF NOT EXISTS docs (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    file_name   TEXT NOT NULL UNIQUE,
    date        DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    doc_id          BIGINT NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    prev_chunk_id   BIGINT REFERENCES chunks(id) ON DELETE SET NULL,
    next_chunk_id   BIGINT REFERENCES chunks(id) ON DELETE SET NULL,
    "offset"        INTEGER NOT NULL CHECK ("offset" >= 0),
    len             INTEGER NOT NULL CHECK (len > 0),
    embedding       vector({EMBEDDING_DIM}) NOT NULL,
    raw_text        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_offset ON chunks (doc_id, "offset");

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_raw_text_bm25
    ON chunks USING bm25 (raw_text)
    WITH (text_config = 'english');
"""


def init_schema(dsn: str = DB_DSN) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch;")
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute(DDL)


def insert_doc_from_metadata(
    conn: psycopg.Connection,
    title: str,
    file_name: str,
    meta_date: str | datetime,
) -> int:
    if isinstance(meta_date, str):
        parsed_dt = datetime.fromisoformat(meta_date)
    else:
        parsed_dt = meta_date

    if parsed_dt.tzinfo is None:
        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)

    row = conn.execute(
        """
        INSERT INTO docs (title, file_name, timestamp)
        VALUES (%s, %s, %s)
        RETURNING id;
        """,
        (title, file_name, parsed_dt),
    ).fetchone()
    assert row is not None
    return int(row[0])


if __name__ == "__main__":
    init_schema()
