import os
import sqlite3
from typing import Iterable, Any, Optional

from .config import settings

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        original_name TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        bytes INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        page_start INTEGER,
        page_end INTEGER,
        section TEXT,
        text TEXT NOT NULL,
        text_len INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(doc_id) REFERENCES documents(id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)""",
    """CREATE INDEX IF NOT EXISTS idx_chunks_sha ON chunks(sha256)""",
]

def _connect():
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

_conn = None

def get_conn():
    global _conn
    if _conn is None:
        _conn = _connect()
        init_db(_conn)
    return _conn

def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    for stmt in SCHEMA:
        cur.execute(stmt)
    conn.commit()

def execute(stmt: str, params: Iterable[Any] = ()):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(stmt, tuple(params))
    conn.commit()
    return cur

def executemany(stmt: str, seq: Iterable[Iterable[Any]]):
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany(stmt, [tuple(x) for x in seq])
    conn.commit()
    return cur

def fetchone(stmt: str, params: Iterable[Any] = ()):
    cur = get_conn().cursor()
    cur.execute(stmt, tuple(params))
    return cur.fetchone()

def fetchall(stmt: str, params: Iterable[Any] = ()):
    cur = get_conn().cursor()
    cur.execute(stmt, tuple(params))
    return cur.fetchall()

def scalar(stmt: str, params: Iterable[Any] = ()):
    row = fetchone(stmt, params)
    if row is None:
        return None
    vals = list(row)
    return vals[0] if vals else None
