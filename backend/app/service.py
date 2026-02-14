import os
import uuid
import datetime
from typing import List, Dict, Any, Optional, Tuple

from .config import settings
from .db import execute, executemany, fetchall, fetchone, scalar
from .utils.files import sha256_bytes, safe_filename, ensure_dir
from .utils.text import chunk_by_sentences, soft_dedup, normalize_text
from .ingest.parsers import parse_pdf, parse_text
from .retrieval.service import RetrievalService
from .llm.providers import make_llm

class AppService:
    def __init__(self):
        ensure_dir(settings.data_dir)
        ensure_dir(os.path.join(settings.data_dir, "uploads"))
        self.retrieval = RetrievalService(settings.data_dir)
        self.llm = make_llm()
        self.retrieval.load_or_build()

    def list_documents(self) -> List[Dict[str, Any]]:
        rows = fetchall(
            """SELECT d.*, (SELECT COUNT(1) FROM chunks c WHERE c.doc_id = d.id) as chunks
               FROM documents d ORDER BY created_at DESC"""
        )
        return [dict(r) for r in rows]

    def upload_and_index(self, original_name: str, mime_type: str, content: bytes) -> Dict[str, Any]:
        doc_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat() + "Z"
        digest = sha256_bytes(content)
        fname = safe_filename(original_name)
        stored_name = f"{doc_id}_{fname}"
        upath = os.path.join(settings.data_dir, "uploads", stored_name)
        with open(upath, "wb") as f:
            f.write(content)

        execute(
            """INSERT INTO documents (id, filename, original_name, mime_type, bytes, sha256, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, stored_name, original_name, mime_type, len(content), digest, created_at)
        )

        chunks = self._extract_chunks(upath, mime_type, original_name)
        self._store_chunks(doc_id, chunks)
        self.retrieval.load_or_build()
        out = fetchone("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return dict(out) if out else {"id": doc_id}

    def _extract_chunks(self, path: str, mime_type: str, original_name: str):
        lower = original_name.lower()
        items = []
        if lower.endswith(".pdf") or mime_type == "application/pdf":
            pages = parse_pdf(path)
            for text, page_no in pages:
                text = normalize_text(text)
                if not text:
                    continue
                parts = chunk_by_sentences(text)
                parts = soft_dedup(parts)
                for p in parts:
                    items.append({"text": p, "page_start": page_no, "page_end": page_no, "section": None})
        else:
            text = normalize_text(parse_text(path))
            if text:
                parts = chunk_by_sentences(text)
                parts = soft_dedup(parts)
                for p in parts:
                    items.append({"text": p, "page_start": None, "page_end": None, "section": None})
        return items

    def _store_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]):
        created_at = datetime.datetime.utcnow().isoformat() + "Z"
        rows = []
        for i, ch in enumerate(chunks):
            cid = str(uuid.uuid4())
            txt = normalize_text(ch["text"])
            if not txt:
                continue
            sh = sha256_bytes(txt.encode("utf-8"))
            rows.append((
                cid, doc_id, i,
                ch.get("page_start"), ch.get("page_end"),
                ch.get("section"),
                txt, len(txt), sh, created_at
            ))
        if rows:
            executemany(
                """INSERT INTO chunks (id, doc_id, chunk_index, page_start, page_end, section, text, text_len, sha256, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows
            )

    def search(self, query: str, top_k: int):
        hits = self.retrieval.search(query, top_k)
        return hits

    async def chat(self, query: str, top_k: int, style: str):
        hits = self.retrieval.search(query, top_k)
        if not hits:
            return {"answer": "Kaynaklarda bu soruya dair içerik bulamadım.", "sources": [], "refused": True, "reason": "no_sources"}
        context = self._make_context(hits)
        if len(context) > settings.max_context_chars:
            context = context[:settings.max_context_chars]
        ans = await self.llm.generate(query, context)
        ans2 = self._postprocess_answer(ans, hits)
        refused = "bulamad" in ans2.lower() and len(hits) == 0
        return {"answer": ans2, "sources": hits, "refused": refused, "reason": ""}

    def _make_context(self, hits: List[Dict[str, Any]]):
        lines = []
        for i, h in enumerate(hits, start=1):
            header = f"[{i}] doc={h['original_name']} chunk={h['chunk_index']}"
            if h.get("page_start"):
                header += f" pages={h['page_start']}-{h.get('page_end') or h['page_start']}"
            lines.append(header)
            lines.append(h["text"].strip())
            lines.append("")
        return "\n".join(lines).strip()

    def _postprocess_answer(self, answer: str, hits: List[Dict[str, Any]]):
        a = (answer or "").strip()
        if not a:
            return "Kaynaklarda bu soruya dair içerik bulamadım."
        a = a.replace("\u2022", "-")
        return a

    def build_eval_metrics(self, items: List[Dict[str, Any]], top_k: int):
        per = []
        p_sum = 0.0
        r_docs_sum = 0.0
        r_chunks_sum = 0.0
        for it in items:
            q = it["question"]
            exp_docs = set(it.get("expected_doc_ids") or [])
            exp_chunks = set(it.get("expected_chunk_ids") or [])
            hits = self.retrieval.search(q, top_k)
            got_docs = [h["doc_id"] for h in hits]
            got_chunks = [h["chunk_id"] for h in hits]
            got_docs_set = set(got_docs)
            got_chunks_set = set(got_chunks)

            correct = 0
            if exp_chunks:
                correct = len(exp_chunks_set & got_chunks_set)
                denom = max(1, top_k)
                prec = correct / denom
                rec_chunks = correct / max(1, len(exp_chunks))
            else:
                correct = len(exp_docs & got_docs_set)
                denom = max(1, top_k)
                prec = correct / denom
                rec_chunks = 0.0

            if exp_docs:
                rec_docs = len(exp_docs & got_docs_set) / max(1, len(exp_docs))
            else:
                rec_docs = 0.0

            p_sum += prec
            r_docs_sum += rec_docs
            r_chunks_sum += rec_chunks
            per.append({
                "question": q,
                "precision_at_k": prec,
                "recall_at_k_docs": rec_docs,
                "recall_at_k_chunks": rec_chunks,
                "top_docs": got_docs[:min(5, len(got_docs))],
                "top_chunks": got_chunks[:min(5, len(got_chunks))]
            })
        n = max(1, len(items))
        return {
            "top_k": top_k,
            "count": len(items),
            "precision_at_k": p_sum / n,
            "recall_at_k_docs": r_docs_sum / n,
            "recall_at_k_chunks": r_chunks_sum / n,
            "per_item": per
        }
