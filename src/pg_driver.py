"""
Postgres driver module.

This module provides a unified Postgres client for the RAG index (the
docs + chunks tables: the story documents and their embedded chunks).
The driver owns the RAG schema: ensure_schema() runs the idempotent
DDL (the extensions, the tables and the HNSW / BM25 indexes) and is
the source of truth for it. In debug mode (DEBUG env var), readonly is
enforced and the client still connects to the real server: the reads
check the real tables (the docs.url dedup stays honest) while the DDL
and the writes are buffered in internal storage so the server remains
unchanged (the session is additionally pinned READ ONLY as a
backstop).

Usage:
    # Production (default)
    pg_client = PostgresDBClient(host="localhost", port=5432, dbname="rag_db")

    # Debug mode (readonly enforced)
    pg_client = PostgresDBClient(readonly=True)

    # Or use environment variable DEBUG
    import os
    os.environ['DEBUG'] = 'true'  # Will automatically enable readonly mode
"""

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

logger = logging.getLogger(__name__)
logger.propagate = False
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('(%(funcName)s:%(lineno)d) %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def _num(value: Any) -> Optional[float]:
    """
    A numeric field of a queue message (the monitoring data) as a
    float (None when missing or not numeric - a failed measurement
    comes out null in the database).
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    """
    An integer field of a queue message (the monitoring data) as an
    int (None when missing or not integral).
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _num_neg(value: Any) -> Optional[float]:
    """
    A numeric field stored with the pg_textsearch sign convention (the
    BM25 scorer returns NEGATIVE scores - the more negative, the better
    match, 0.0 when the doc does not match) flipped to positive (higher
    = better - the vec_score semantics); None when missing.
    """
    v = _num(value)
    return None if v is None else -v


class ChunkRow(NamedTuple):
    """
    One chunk row of the chunks table (the embedding precomputed).

    Attributes:
        offset: The first token of the chunk in the story's OWN
                tokenization (the embedding model's token space;
                resolved later by re-tokenizing the story file and
                taking tokens[offset : offset + length]).
        length: The length of the chunk in tokens of the story's own
                tokenization.
        raw_text: The exact text the embedding was computed on (stored
                  verbatim for BM25).
        embedding: The precomputed embedding (EMBEDDING_DIM floats).
    """
    offset : int
    length : int
    raw_text : str
    embedding : List[float]


class PostgresDBClient:
    """
    Unified Postgres client for the RAG index (the docs + chunks
    tables).

    Always connects to a real Postgres server. In readonly mode the
    real tables are still checked by the reads (the docs.url dedup
    stays honest) while the DDL and the writes are buffered in
    internal storage so the server remains unchanged.

    The schema is owned by ensure_schema() (idempotent DDL: the
    extensions, docs, chunks and the HNSW / BM25 indexes) - the
    driver is the source of truth for the RAG schema.

    Example:
        # Production
        async with PostgresDBClient() as client:
            await client.connect()
            await client.ensure_schema()
            doc_id = await client.insert_story(url, title, date, ppl, chunks)

        # Debug mode
        async with PostgresDBClient(readonly=True) as client:
            await client.connect()
            await client.ensure_schema()
            doc_id = await client.insert_story(url, title, date, ppl, chunks)
    """

    EMBEDDING_DIM = 1024  # the jina-v3 embedding dimension (vector(1024))
    # The advisory lock key of the RAG index maintenance. The ingester
    # holds it ONLY around the post-batch BM25 merge (the index-
    # rewriting operation - milliseconds) so a retrieval never overlaps
    # the merge: the readers are NEVER blocked during the data ingestion
    # itself (their reads are snapshot-consistent by MVCC).
    RAG_INDEX_LOCK_KEY = 0x52414749  # "RAGI"

    # The expected shape of the RAG schema (ensure_schema is the source
    # of truth; the checks below detect a legacy table layout).
    _EXPECTED_DOCS_COLS = frozenset({
        "id", "url", "title", "date",
        "ppl_per_word", "ppl_per_token", "ppl_tokens", "ppl_words",
        "ppls", "ppl_frag_p90", "ppl_error",
    })
    _EXPECTED_CHUNKS_COLS = frozenset({
        "id", "doc_id", "prev_chunk_id", "next_chunk_id",
        "offset", "len", "embedding", "raw_text",
    })
    # The canonical DDL (idempotent - the driver owns the RAG schema, the
    # legacy scripts/postgre_init.py is not part of the pipeline).
    # docs.url is the story file location (the queue's file_path) and the
    # unique idempotency key of the index; the ppl_* columns are the
    # story's perplexity metrics (the monitoring data the scrapper
    # measured pre-publish - stored as-is, a failed measurement comes
    # out null).
    _DDL = (
        "CREATE EXTENSION IF NOT EXISTS vector;",
        "CREATE EXTENSION IF NOT EXISTS pg_textsearch;",
        """
        CREATE TABLE IF NOT EXISTS docs (
            id              BIGSERIAL PRIMARY KEY,
            url             TEXT NOT NULL UNIQUE,
            title           TEXT NOT NULL,
            date            DATE NOT NULL,
            ppl_per_word    DOUBLE PRECISION,
            ppl_per_token   DOUBLE PRECISION,
            ppl_tokens      INTEGER,
            ppl_words       INTEGER,
            ppls            JSONB,
            ppl_frag_p90    DOUBLE PRECISION,
            ppl_error       TEXT
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id              BIGSERIAL PRIMARY KEY,
            doc_id          BIGINT NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
            prev_chunk_id   BIGINT NOT NULL DEFAULT 0,
            next_chunk_id   BIGINT NOT NULL DEFAULT 0,
            "offset"        INTEGER NOT NULL CHECK ("offset" >= 0),
            len             INTEGER NOT NULL CHECK (len > 0),
            embedding       vector({EMBEDDING_DIM}) NOT NULL,
            raw_text        TEXT NOT NULL
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);",
        'CREATE INDEX IF NOT EXISTS idx_chunks_doc_offset ON chunks (doc_id, "offset");',
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw "
        "ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_raw_text_bm25 "
        "ON chunks USING bm25 (raw_text) WITH (text_config = 'english');",
    )

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "rag_db",
        user: str = "postgres",
        password: Optional[str] = None,
        readonly: Optional[bool] = None
    ) -> None:
        """
        Initialize the Postgres client.

        Args:
            host: Postgres server hostname.
            port: Postgres server port.
            dbname: The RAG database name.
            user: The Postgres user.
            password: The Postgres password (None for trust auth).
            readonly: If True, only checks the real tables and buffers
                      the writes (the server is not modified). If not
                      specified, checks the DEBUG environment variable.
        """
        if readonly is None:
            import os
            readonly = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')

        self.readonly = readonly
        self._conn = None
        self._host = host
        self._port = port
        self._dbname = dbname
        self._user = user
        self._password = password
        self._connected = False
        # Readonly-mode state: url -> the synthetic doc id buffered in
        # this run (the docs.url dedup is still checked against the real
        # tables; only the writes buffer). The synthetic ids start past
        # the real docs (see connect) so they can never collide.
        self._buffered_docs: Dict[str, int] = {}
        self._synthetic_doc_id = 1

    @property
    def host(self) -> str:
        """Postgres server hostname."""
        return self._host

    @host.setter
    def host(self, value: str) -> None:
        if self._conn is not None:
            raise RuntimeError("Cannot change host while connected. Close connection first.")
        self._host = value

    @property
    def port(self) -> int:
        """Postgres server port."""
        return self._port

    @port.setter
    def port(self, value: int) -> None:
        if self._conn is not None:
            raise RuntimeError("Cannot change port while connected. Close connection first.")
        self._port = value

    @property
    def dbname(self) -> str:
        """The RAG database name."""
        return self._dbname

    @dbname.setter
    def dbname(self, value: str) -> None:
        if self._conn is not None:
            raise RuntimeError("Cannot change dbname while connected. Close connection first.")
        self._dbname = value

    @property
    def user(self) -> str:
        """The Postgres user."""
        return self._user

    @user.setter
    def user(self, value: str) -> None:
        if self._conn is not None:
            raise RuntimeError("Cannot change user while connected. Close connection first.")
        self._user = value

    async def _table_counts(self) -> Tuple[int, int]:
        """
        (The number of docs, the number of chunks) of the RAG tables,
        (-1, -1) when either table is missing.
        """
        try:
            cursor = await self._conn.execute("SELECT COUNT(*) FROM docs")
            docs = (await cursor.fetchone())[0]
            cursor = await self._conn.execute("SELECT COUNT(*) FROM chunks")
            chunks = (await cursor.fetchone())[0]
            return (docs, chunks)
        except Exception:
            return (-1, -1)

    async def _table_columns(self) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        """
        The column sets of the docs / chunks tables (None for a table
        that is missing or unreadable).
        """
        try:
            cursor = await self._conn.execute(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name IN ('docs', 'chunks')")
            rows = await cursor.fetchall()
        except Exception as e:
            logger.error(f"Postgres error: {e}")
            return (None, None)
        docs_cols = {r[1] for r in rows if r[0] == "docs"}
        chunks_cols = {r[1] for r in rows if r[0] == "chunks"}
        return ((docs_cols if docs_cols else None), (chunks_cols if chunks_cols else None))

    async def connect(self) -> int:
        """
        Connect to the Postgres server.

        The connection is opened in autocommit mode: every read is a
        standalone transaction, and the single write path
        (insert_story) wraps its statements in one explicit
        transaction (the atomic doc + chunks + links settle). In
        readonly mode the session is additionally pinned READ ONLY
        (SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY) so the
        server can never be modified.

        Returns:
            0 on success, -1 on failure.
        """
        if self._connected:
            logger.error("Postgres client is already opened!")
            return -1

        try:
            import psycopg
            from pgvector.psycopg import register_vector_async
            conninfo = f"host={self._host} port={self._port} dbname={self._dbname} user={self._user}"
            if self._password:
                conninfo += f" password={self._password}"
            self._conn = await psycopg.AsyncConnection.connect(conninfo, autocommit=True)
            await self._conn.execute("SELECT 1")  # the ping
            (docs_count, chunks_count) = await self._table_counts()
            if self.readonly:
                # Pin the session: every transaction on this session is
                # read-only (a backstop on top of the driver's own
                # readonly branching).
                await self._conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
                # The synthetic doc ids start past the real docs, so a
                # readonly buffered id can never collide with a real one.
                self._buffered_docs = {}
                self._synthetic_doc_id = max(0, docs_count) + 1
            # The pgvector codec (best effort: on a fresh database the
            # vector type does not exist until ensure_schema() runs -
            # the codec is (re)registered there).
            try:
                await register_vector_async(self._conn)
            except Exception as e:
                logger.debug(f"Postgres: the vector type is not present yet ({e}); "
                             "the codec will be registered by ensure_schema().")
            self._connected = True
            if docs_count < 0:
                logger.info(f"Postgres connected to {self._host}:{self._port}/{self._dbname} "
                            f"(no RAG tables yet - run ensure_schema)"
                            + (", readonly: reads real, writes buffered" if self.readonly else ""))
            else:
                logger.info(f"Postgres connected to {self._host}:{self._port}/{self._dbname}, "
                            f"{docs_count} docs, {chunks_count} chunks"
                            + (", readonly: reads real, writes buffered" if self.readonly else ""))
            return 0
        except Exception as e:
            logger.error(f"Failed to connect Postgres, error: {e}")
            try:
                if self._conn is not None:
                    await self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._connected = False
            return -1

    async def disconnect(self) -> None:
        """
        Gracefully disconnect and clean up all resources.

        This method closes the connection and clears internal state
        (including the readonly write buffer).
        """
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception as e:
                logger.error(f"Postgres close failed, error: {e}")
            self._conn = None
        self._connected = False
        self._buffered_docs = {}
        self._synthetic_doc_id = 1
        logger.info("Postgres connection closed.")

    async def restart(self) -> int:
        """
        Restart the Postgres connection by disconnecting and reconnecting.

        Returns:
            0 on success, -1 on failure.
        """
        await self.disconnect()
        return await self.connect()

    async def close(self) -> None:
        """
        Close the Postgres connection (alias for disconnect).
        """
        await self.disconnect()

    async def ensure_schema(self) -> int:
        """
        Ensure the RAG schema is present and current (idempotent):
        the extensions (vector, pg_textsearch), docs(id, url UNIQUE,
        title, date, ppl_per_word, ppl_per_token, ppl_tokens,
        ppl_words, ppls, ppl_frag_p90, ppl_error),
        chunks(id, doc_id FK, prev_chunk_id / next_chunk_id NOT NULL
        DEFAULT 0, offset / len (tokens of the story's own
        tokenization), embedding vector(EMBEDDING_DIM), raw_text) and
        the indexes ((doc_id), (doc_id, offset), the HNSW embedding
        index (vector_cosine_ops, m=16,
        ef_construction=64) and the BM25 full-text index on raw_text
        (text_config='english')).

        The driver owns the schema: a table that exists with a
        different (legacy) layout is adopted in place only when both
        tables are EMPTY (drop + recreate); a non-empty legacy layout
        is a migration decision, not a bootstrap - the check fails
        loudly. In readonly mode no DDL is issued at all: the presence
        of the current schema is only checked.

        Returns:
            0 on success, -1 on failure.
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return -1

        docs_cols, chunks_cols = await self._table_columns()
        current = (docs_cols is not None and self._EXPECTED_DOCS_COLS.issubset(docs_cols)
                   and chunks_cols is not None and self._EXPECTED_CHUNKS_COLS.issubset(chunks_cols))
        if current:
            if self.readonly:
                logger.info("Postgres schema is present and current (readonly: existence check only).")
            else:
                logger.info("Postgres schema is already present (idempotent no-op).")
            return 0

        if self.readonly:
            logger.error(f"Postgres schema check (readonly): the RAG schema is missing or stale "
                         f"(docs columns: {sorted(docs_cols) if docs_cols else None}, "
                         f"chunks columns: {sorted(chunks_cols) if chunks_cols else None}) - "
                         "run once in production mode to create/migrate it.")
            return -1

        try:
            legacy = (docs_cols is not None and "url" not in docs_cols) \
                     or (chunks_cols is not None and "raw_text" not in chunks_cols)
            if legacy:
                (docs_count, chunks_count) = await self._table_counts()
                if docs_count > 0 or chunks_count > 0:
                    logger.error(f"Postgres schema check: the legacy RAG tables are not empty "
                                 f"({docs_count} docs, {chunks_count} chunks) - migrate them "
                                 "manually before running the pipeline.")
                    return -1
                logger.info("Postgres schema check: the legacy RAG tables are empty - dropping and recreating them.")
                await self._conn.execute("DROP TABLE IF EXISTS chunks;")
                await self._conn.execute("DROP TABLE IF EXISTS docs;")
            for statement in self._DDL:
                await self._conn.execute(statement)
            # The vector type is guaranteed now - (re)register the codec
            # (it was skipped in connect on a database without it).
            from pgvector.psycopg import register_vector_async
            await register_vector_async(self._conn)
            logger.info(f"Postgres schema ensured (extensions, docs, chunks, HNSW + BM25 indexes; {len(self._DDL)} statements).")
            return 0
        except Exception as e:
            logger.error(f"Failed to ensure the Postgres RAG schema, error: {e}")
            return -1

    async def doc_exists(self, url: str) -> int:
        """
        Check whether a story (by its url - the story file location,
        the docs.url unique key) is already indexed.

        In readonly mode the internal buffer (this run's buffered
        docs) is checked first - the dedup stays honest across the
        non-destructive re-reads of the queue head.

        Returns:
            1 if the url is already indexed
            0 if the url is not indexed yet
            -1 on error
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return -1

        if self.readonly and url in self._buffered_docs:
            logger.debug(f"URL already indexed (internal storage): {url}")
            return 1
        try:
            cursor = await self._conn.execute("SELECT 1 FROM docs WHERE url = %s", (url,))
            row = await cursor.fetchone()
        except Exception as e:
            logger.error(f"Postgres error: {e}")
            return -1
        if row is not None:
            logger.debug(f"URL already indexed (Postgres): {url}")
        return 1 if row is not None else 0

    async def get_doc(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch one document row.

        Returns:
            {"id", "url", "title", "date", "ppl_per_word",
            "ppl_per_token", "ppl_tokens", "ppl_words", "ppls",
            "ppl_frag_p90", "ppl_error"} (the ppl_* fields are the
            monitoring data stored as-is - null when the measurement
            failed) - or None when the doc does not exist (or on
            error, logged).
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return None
        try:
            cursor = await self._conn.execute(
                """
                SELECT id, url, title, date,
                       ppl_per_word, ppl_per_token, ppl_tokens, ppl_words,
                       ppls, ppl_frag_p90, ppl_error
                FROM docs WHERE id = %s
                """, (doc_id,))
            row = await cursor.fetchone()
        except Exception as e:
            logger.error(f"Postgres error: {e}")
            return None
        if row is None:
            return None
        return { "id"          : row[0],
                 "url"         : row[1],
                 "title"       : row[2],
                 "date"        : row[3],
                 "ppl_per_word" : row[4],
                 "ppl_per_token" : row[5],
                 "ppl_tokens"  : row[6],
                 "ppl_words"   : row[7],
                 "ppls"        : row[8],
                 "ppl_frag_p90" : row[9],
                 "ppl_error"   : row[10] }

    async def get_doc_id_by_url(self, url: str) -> Optional[int]:
        """
        Get the doc id of an indexed story by its url (the story file
        location, the docs.url unique key).

        Returns:
            The doc id (> 0), or None when the url is not indexed (or
            on error, logged).
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return None
        try:
            cursor = await self._conn.execute("SELECT id FROM docs WHERE url = %s", (url,))
            row = await cursor.fetchone()
        except Exception as e:
            logger.error(f"Postgres error: {e}")
            return None
        return row[0] if row is not None else None

    async def get_chunks(self, doc_id: int) -> List[Dict[str, Any]]:
        """
        Fetch every chunk of a document in document order ("offset").

        Returns:
            A list of {"id", "doc_id", "prev_chunk_id", "next_chunk_id"
            (0 = no prev/next chunk), "offset", "len", "raw_text",
            "embedding"} (the embedding is a list of EMBEDDING_DIM
            floats) - [] when the doc has no chunks (or on error,
            logged).
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return []
        try:
            cursor = await self._conn.execute(
                """
                SELECT id, doc_id, prev_chunk_id, next_chunk_id, "offset", len, raw_text, embedding
                FROM chunks WHERE doc_id = %s
                ORDER BY "offset", id
                """, (doc_id,))
            rows = await cursor.fetchall()
        except Exception as e:
            logger.error(f"Postgres error: {e}")
            return []
        return [ { "id" : r[0], "doc_id" : r[1], "prev_chunk_id" : r[2],
                   "next_chunk_id" : r[3], "offset" : r[4], "len" : r[5],
                   "raw_text" : r[6], "embedding" : r[7].to_list() } for r in rows ]

    async def get_doc_count(self) -> int:
        """
        Get the total number of indexed documents.

        Returns:
            Number of docs, or -1 if not connected.
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return -1
        try:
            cursor = await self._conn.execute("SELECT COUNT(*) FROM docs")
            return (await cursor.fetchone())[0]
        except Exception as e:
            logger.error(f"Postgres error: {e}")
            return -1

    async def get_chunk_count(self) -> int:
        """
        Get the total number of indexed chunks.

        Returns:
            Number of chunks, or -1 if not connected.
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return -1
        try:
            cursor = await self._conn.execute("SELECT COUNT(*) FROM chunks")
            return (await cursor.fetchone())[0]
        except Exception as e:
            logger.error(f"Postgres error: {e}")
            return -1

    async def search_chunks(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query (the RAG
        retrieval step - the read-only counterpart of insert_story):
        the chunks are ranked by reciprocal rank fusion (RRF,
        k = 60) over the two retrieval paths:
          - the BM25 full-text rank over raw_text (the
            idx_chunks_raw_text_bm25 index, the pg_textsearch
            scoring - a row without a positive score does not
            appear in this rank; if the extension's background
            scoring is not current the whole rank is empty and the
            fusion degrades to the other path),
          - the vector rank (the cosine distance, the HNSW index)
            when query_embedding (the query embedded with the SAME
            embedding model as the corpus) is given.
        The RRF ranks are computed over the FULL chunk ranking (no
        pre-limit) so the fusion does not distort the ranks.

        Read-only: a pure SELECT (safe in readonly mode, where the
        session is pinned READ ONLY).

        Args:
            query_text: The query (the BM25 side, analyzed with the
                        index's text_config).
            query_embedding: The query's embedding (EMBEDDING_DIM
                             floats) for the vector path (None = the
                             BM25 path only).
            top_k: The number of chunks to return (>= 1).

        Returns:
            A list (at most top_k) of {"chunk_id", "doc_id",
            "offset", "len", "raw_text", "title", "date", "rrf"
            (the fused score), "bm25_rank" / "bm25_score" (the BM25
            path's 1-based rank / score - positive, the extension's raw
            score is negative so it is flipped here; higher = better -
            None when the BM25 path did not match), "vec_rank" /
            "vec_score" (the vector path's 1-based rank / cosine
            similarity, None when the vector path was not used)}
            ordered by relevance - [] on error (logged) or when nothing
            matches.
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return []
        if not query_text or not query_text.strip():
            logger.error("Postgres search_chunks: no query text given.")
            return []
        top_k = max(1, int(top_k))

        # NOTE: the extension returns the BM25 score NEGATIVE (the sign
        # flip is its ASC-ordering convention) - a match scores < 0 (the
        # more negative, the better match), a non-match scores 0.0. So
        # the filter is < 0 and the rank orders ASC (best = rank 1); the
        # returned score is flipped to positive (see _num_neg).
        bm25_cte = """
        bm25_rank AS (
            SELECT c.id,
                   bm25_text_bm25query_score(
                       c.raw_text,
                       to_bm25query(%s, 'idx_chunks_raw_text_bm25')) AS score,
                   RANK() OVER (
                       ORDER BY bm25_text_bm25query_score(
                           c.raw_text,
                           to_bm25query(%s, 'idx_chunks_raw_text_bm25')) ASC
                   ) AS rank
            FROM chunks c
            WHERE bm25_text_bm25query_score(
                      c.raw_text,
                      to_bm25query(%s, 'idx_chunks_raw_text_bm25')) < 0
        )
        """
        if query_embedding:
            sql = f"""
            WITH {bm25_cte},
            vec_rank AS (
                SELECT c.id,
                       1.0 - (c.embedding <=> %s::vector) AS score,
                       RANK() OVER (ORDER BY c.embedding <=> %s::vector) AS rank
                FROM chunks c
            ),
            fused AS (
                SELECT COALESCE(b.id, v.id) AS id,
                       COALESCE(1.0 / (60 + b.rank), 0.0)
                     + COALESCE(1.0 / (60 + v.rank), 0.0) AS rrf
                FROM bm25_rank b
                FULL JOIN vec_rank v ON b.id = v.id
            )
            SELECT c.id, c.doc_id, c."offset", c.len, c.raw_text,
                   d.title, d."date",
                   f.rrf, b.rank, b.score, v.rank, v.score
            FROM fused f
            JOIN chunks c ON c.id = f.id
            JOIN docs d ON d.id = c.doc_id
            LEFT JOIN bm25_rank b ON b.id = f.id
            LEFT JOIN vec_rank v ON v.id = f.id
            ORDER BY f.rrf DESC, c.id
            LIMIT %s
            """
            params = (query_text, query_text, query_text, query_embedding,
                      query_embedding, top_k)
        else:
            sql = f"""
            WITH {bm25_cte}
            SELECT c.id, c.doc_id, c."offset", c.len, c.raw_text,
                   d.title, d."date",
                   0.0, b.rank, b.score, NULL, NULL
            FROM bm25_rank b
            JOIN chunks c ON c.id = b.id
            JOIN docs d ON d.id = c.doc_id
            ORDER BY b.rank ASC
            LIMIT %s
            """
            params = (query_text, query_text, query_text, top_k)
        try:
            cursor = await self._conn.execute(sql, params)
            rows = await cursor.fetchall()
        except Exception as e:
            logger.error(f"Postgres error: {e}")
            return []
        return [ { "chunk_id" : r[0], "doc_id" : r[1], "offset" : r[2], "len" : r[3],
                   "raw_text" : r[4], "title" : r[5], "date" : r[6],
                   "rrf" : _num(r[7]), "bm25_rank" : _int(r[8]),
                   "bm25_score" : _num_neg(r[9]),
                   "vec_rank" : _int(r[10]), "vec_score" : _num(r[11]) }
                 for r in rows ]

    async def try_lock_index(self) -> bool:
        """
        Try to take the RAG index maintenance lock (held by the ingester
        ONLY around the post-batch BM25 merge - see RAG_INDEX_LOCK_KEY):
        the session-level advisory lock RAG_INDEX_LOCK_KEY, taken
        non-blocking (pg_try_advisory_lock). It is held until
        unlock_index() or until the session ends (released automatically
        on disconnect / crash - a wedged run can never leave the index
        locked).

        Write mode only: in readonly mode the lock is not taken and
        False is returned.

        Returns:
            True when the lock was taken, False when another run is
            merging the index right now (or in readonly mode).
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return False
        if self.readonly:
            logger.warning("Postgres try_lock_index: readonly mode - the lock is not taken.")
            return False
        try:
            cursor = await self._conn.execute(
                "SELECT pg_try_advisory_lock(%s)", (self.RAG_INDEX_LOCK_KEY,))
            row = await cursor.fetchone()
            return bool(row[0])
        except Exception as e:
            logger.error(f"Postgres error: {e}")
            return False

    async def unlock_index(self) -> bool:
        """
        Release the RAG index maintenance lock taken by this session
        (pg_advisory_unlock - a no-op when this session does not hold
        it; the session end releases it anyway).

        Returns:
            True when released (or not held in readonly mode), False on
            error.
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return False
        if self.readonly:
            return True  # readonly never takes the lock
        try:
            cursor = await self._conn.execute(
                "SELECT pg_advisory_unlock(%s)", (self.RAG_INDEX_LOCK_KEY,))
            row = await cursor.fetchone()
            return bool(row[0])
        except Exception as e:
            logger.error(f"Postgres error: {e}")
            return False

    async def is_index_locked(self) -> bool:
        """
        Probe (a plain SELECT on pg_locks - safe in readonly mode)
        whether the RAG index maintenance lock is currently held by ANY
        session (a BM25 merge in flight - the lock is held only for the
        merge itself, never during the data ingestion; see
        RAG_INDEX_LOCK_KEY).

        Returns:
            True when the lock is held, False otherwise (also when not
            connected).
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return False
        try:
            # The bigint advisory lock shows up in pg_locks as
            # locktype='advisory', classid = the high 32 bits of the key,
            # objid = the low 32 bits, objsubid = 1.
            cursor = await self._conn.execute(
                "SELECT count(*) > 0 FROM pg_locks"
                " WHERE locktype = 'advisory' AND objsubid = 1"
                "   AND classid = %s AND objid = %s",
                (self.RAG_INDEX_LOCK_KEY >> 32, self.RAG_INDEX_LOCK_KEY & 0xFFFFFFFF))
            row = await cursor.fetchone()
            return bool(row[0])
        except Exception as e:
            logger.error(f"Postgres error: {e}")
            return False

    async def compact_bm25_index(self) -> bool:
        """
        The post-batch maintenance pass of the BM25 index:
        bm25_force_merge collapses the memtable into the on-disk
        segment (the index-only LSM merge - the table data is
        untouched, milliseconds at this corpus scale). Run it after an
        indexing batch, under the index maintenance lock (see
        RAG_INDEX_LOCK_KEY), so the corpus is fully consolidated (one
        uniform index state for the BM25 scorer) before the next reader
        comes.

        Write mode only: in readonly mode the pass is skipped.

        Returns:
            True when the pass ran (or was skipped in readonly), False
            on error (logged).
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return False
        if self.readonly:
            logger.debug("Postgres compact_bm25_index: readonly mode - the pass is skipped.")
            return True
        try:
            await self._conn.execute("SELECT bm25_force_merge('idx_chunks_raw_text_bm25')")
            logger.info("BM25 index maintenance: the memtable was merged into the on-disk segment.")
            return True
        except Exception as e:
            logger.error(f"Postgres error (BM25 index maintenance): {e}")
            return False

    async def insert_story(
        self,
        url: str,
        title: str,
        date: str,
        ppl: Dict[str, Any],
        chunks: List[ChunkRow]
    ) -> int:
        """
        Atomically index one story: the docs row (url, title, date +
        the story's perplexity metrics from the queue message - the
        monitoring data) + every chunk row + the prev/next link pass
        in ONE transaction (so a crash can never leave a document
        without its chunks - which the url dedup would then skip
        forever).

        The docs row is inserted with ON CONFLICT (url) DO NOTHING -
        the unique url is the idempotency key of the whole run: a
        story is indexed exactly once no matter how many workers /
        runs see it. The chunks are inserted in row order (RETURNING
        id), then linked from that order: prev_chunk_id[i] = id[i-1]
        (0 for the first), next_chunk_id[i] = id[i+1] (0 for the
        last) - 0 means no prev/next chunk.

        In readonly mode the url dedup is still checked against the
        real tables (and the internal buffer from this run), then the
        doc + chunks are buffered with a synthetic id (the links are
        resolved against the synthetic ids); the server is not
        modified.

        Args:
            url: The story file location (docs.url, unique - the
                 queue's file_path).
            title: The story title (docs.title).
            date: The crawl date, "YYYY-MM-DD" (docs.date - the
                  queue's timestamp, the news/<date> folder name).
            ppl: The story's perplexity metrics from the queue message
                 (ppl_per_word / ppl_per_token / ppl_tokens /
                 ppl_words / ppls / ppl_frag_p90 / ppl_error) - the
                 monitoring data, stored as-is (a missing / failed
                 field comes out null).
            chunks: The chunk rows (offset, length, raw_text, the
                    precomputed embedding) in document order -
                    offset/length are tokens of the story's own
                    tokenization (the embedding model's token space),
                    raw_text the exact text the embedding was
                    computed on.

        Returns:
            The new doc id (> 0) on success, 0 if the url is already
            indexed (nothing inserted), -1 on error (the whole
            transaction is rolled back).
        """
        if self._conn is None:
            logger.error("Postgres client is not yet opened!")
            return -1
        if not url:
            logger.error("Postgres insert_story: no url given.")
            return -1

        if self.readonly:
            return await self._insert_story_buffered(url, title, date, ppl, chunks)
        return await self._insert_story_real(url, title, date, ppl, chunks)

    async def _insert_story_real(
        self,
        url: str,
        title: str,
        date: str,
        ppl: Dict[str, Any],
        chunks: List[ChunkRow]
    ) -> int:
        """
        Production implementation of insert_story: the doc + the
        chunks + the prev/next link pass commit in ONE Postgres
        transaction (a crash can never leave a document without its
        chunks).
        """
        from psycopg.types.json import Jsonb
        try:
            async with self._conn.transaction():
                # The doc row (ON CONFLICT (url) DO NOTHING - the unique
                # url is the idempotency key of the whole run).
                cursor = await self._conn.execute(
                    """
                    INSERT INTO docs (url, title, date, ppl_per_word, ppl_per_token,
                                      ppl_tokens, ppl_words, ppls, ppl_frag_p90, ppl_error)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                    RETURNING id
                    """,
                    (url, title, date,
                     _num(ppl.get("ppl_per_word")), _num(ppl.get("ppl_per_token")),
                     _int(ppl.get("ppl_tokens")), _int(ppl.get("ppl_words")),
                     Jsonb(ppl["ppls"]) if ppl.get("ppls") is not None else None,
                     _num(ppl.get("ppl_frag_p90")),
                     str(ppl["ppl_error"]) if ppl.get("ppl_error") else None),
                )
                row = await cursor.fetchone()
                if row is None:
                    logger.debug(f"Story already indexed (url): {url}")
                    return 0
                doc_id = row[0]
                # The chunk rows, in document order (the ids are
                # collected for the prev/next link pass).
                ids = []
                for chunk in chunks:
                    cursor = await self._conn.execute(
                        """
                        INSERT INTO chunks (doc_id, prev_chunk_id, next_chunk_id,
                                            "offset", len, embedding, raw_text)
                        VALUES (%s, 0, 0, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (doc_id, chunk.offset, chunk.length, chunk.embedding, chunk.raw_text),
                    )
                    chunk_id = (await cursor.fetchone())[0]
                    ids.append(chunk_id)
                # The link pass (0 = no prev/next chunk): prev[i] =
                # id[i-1], next[i] = id[i+1].
                for i, chunk_id in enumerate(ids):
                    prev_id = ids[i - 1] if i > 0 else 0
                    next_id = ids[i + 1] if i < len(ids) - 1 else 0
                    await self._conn.execute(
                        "UPDATE chunks SET prev_chunk_id = %s, next_chunk_id = %s WHERE id = %s",
                        (prev_id, next_id, chunk_id))
            logger.info(f"Postgres inserted story doc {doc_id}: {len(chunks)} chunks (url: {url})")
            return doc_id
        except Exception as e:
            logger.error(f"Postgres insert_story failed (the transaction was rolled back), error: {e}")
            return -1

    async def _insert_story_buffered(
        self,
        url: str,
        title: str,
        date: str,
        ppl: Dict[str, Any],
        chunks: List[ChunkRow]
    ) -> int:
        """
        Readonly implementation of insert_story: the url dedup is
        still checked against the real tables (and the internal
        buffer from this run), then the doc is buffered with a
        synthetic id; the server is not modified.
        """
        if url in self._buffered_docs:
            logger.debug(f"URL already indexed (internal storage): {url}")
            return 0
        try:
            cursor = await self._conn.execute("SELECT 1 FROM docs WHERE url = %s", (url,))
            row = await cursor.fetchone()
        except Exception as e:
            logger.error(f"Postgres error: {e}")
            return -1
        if row is not None:
            logger.debug(f"URL already indexed (Postgres): {url}")
            return 0
        doc_id = self._synthetic_doc_id
        self._synthetic_doc_id += 1
        self._buffered_docs[url] = doc_id
        logger.debug(f"Readonly Postgres buffered story: doc {doc_id} ({len(chunks)} chunks, "
                     f"url: {url}, title: {title}, date: {date})")
        return doc_id

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager, disconnecting Postgres on completion."""
        await self.disconnect()
