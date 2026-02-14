from typing import List, Dict, Any, Tuple, Optional
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import settings
from ..db import fetchall
from ..utils.text import normalize_text
from .faiss_store import FaissStore
from .bm25 import BM25Index
from .rerank import Reranker, CrossEncoderReranker, OllamaReranker

class RetrievalService:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.index_dir = os.path.join(self.data_dir, "index")
        self.faiss_path = os.path.join(self.index_dir, "chunks.faiss")
        self.embedder = SentenceTransformer(settings.embed_model)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.faiss = FaissStore(self.dim, self.faiss_path)
        self.bm25 = BM25Index()
        self.reranker = self._make_reranker()

    def _make_reranker(self) -> Reranker:
        if settings.llm_provider.strip().lower() == "ollama":
            return OllamaReranker(settings.ollama_base_url, settings.ollama_model)
        return Reranker()

    def load_or_build(self):
        rows = fetchall("SELECT id, doc_id, chunk_index, page_start, page_end, section, text FROM chunks ORDER BY created_at ASC")
        texts = [normalize_text(r["text"]) for r in rows]
        meta = []
        for r in rows:
            meta.append({
                "chunk_id": r["id"],
                "doc_id": r["doc_id"],
                "chunk_index": int(r["chunk_index"]),
                "page_start": r["page_start"],
                "page_end": r["page_end"],
                "section": r["section"]
            })
        if not rows:
            self.faiss.index = None
            self.faiss.meta = []
            self.bm25.build([], [])
            return

        if self.faiss.exists():
            try:
                self.faiss.load()
                if len(self.faiss.meta) != len(meta):
                    raise RuntimeError("meta mismatch")
            except Exception:
                self._rebuild(texts, meta)
        else:
            self._rebuild(texts, meta)

        self.bm25.build(texts, meta)

    def _rebuild(self, texts: List[str], meta: List[Dict[str, Any]]):
        vecs = self.embedder.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        vecs = vecs.astype(np.float32)
        self.faiss.build(vecs, meta)
        self.faiss.save()

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query = normalize_text(query)
        if not query:
            return []
        if self.faiss.index is None:
            return []

        qv = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        vec_hits = self.faiss.search(qv, max(top_k * 4, top_k))
        bm_hits = self.bm25.search(query, max(top_k * 4, top_k))

        vec_map = {i: s for i, s in vec_hits}
        bm_map = {i: s for i, s in bm_hits}

        if vec_map:
            vmin, vmax = min(vec_map.values()), max(vec_map.values())
        else:
            vmin, vmax = 0.0, 1.0
        if bm_map:
            bmin, bmax = min(bm_map.values()), max(bm_map.values())
        else:
            bmin, bmax = 0.0, 1.0

        def norm(x, a, b):
            if b - a < 1e-9:
                return 0.0
            return (x - a) / (b - a)

        cand = set(vec_map.keys()) | set(bm_map.keys())
        scored = []
        for idx in cand:
            v = vec_map.get(idx, None)
            b = bm_map.get(idx, None)
            vs = norm(v, vmin, vmax) if v is not None else 0.0
            bs = norm(b, bmin, bmax) if b is not None else 0.0
            score = settings.hybrid_alpha * vs + (1.0 - settings.hybrid_alpha) * bs
            scored.append((idx, float(score), float(v or 0.0), float(b or 0.0)))
        scored.sort(key=lambda x: x[1], reverse=True)

        keep = scored[:max(top_k * 4, top_k)]
        ids = [self.faiss.meta[i]["chunk_id"] for i, _, _, _ in keep]
        rows = fetchall(
            """SELECT c.id as chunk_id, c.doc_id, c.chunk_index, c.page_start, c.page_end, c.section, c.text,
                       d.original_name, d.filename
                FROM chunks c JOIN documents d ON d.id = c.doc_id
                WHERE c.id IN (%s)
            """ % ",".join(["?"] * len(ids)),
            ids
        )
        row_map = {r["chunk_id"]: r for r in rows}
        out = []
        for idx, hs, vs, bs in keep:
            cid = self.faiss.meta[idx]["chunk_id"]
            r = row_map.get(cid)
            if not r:
                continue
            out.append({
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "original_name": r["original_name"],
                "filename": r["filename"],
                "chunk_index": int(r["chunk_index"]),
                "page_start": r["page_start"],
                "page_end": r["page_end"],
                "section": r["section"],
                "score": float(hs),
                "vec_score": float(vs),
                "bm25_score": float(bs),
                "text": r["text"]
            })

        out = self._dedup_results(out)
        out = self.reranker.rerank(query, out)
        return out[:top_k]

    def _dedup_results(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for it in items:
            key = (it["doc_id"], it["chunk_index"])
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out
