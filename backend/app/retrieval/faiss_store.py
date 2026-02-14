import os
import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import faiss

class FaissStore:
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = index_path + ".meta.json"
        self.index = None
        self.meta = []

    def exists(self) -> bool:
        return os.path.exists(self.index_path) and os.path.exists(self.meta_path)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.dim = self.index.d

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

    def build(self, vectors: np.ndarray, meta: List[Dict[str, Any]]):
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        self.dim = vectors.shape[1]
        idx = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(vectors)
        idx.add(vectors)
        self.index = idx
        self.meta = meta

    def search(self, q: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if self.index is None:
            raise RuntimeError("index not loaded")
        if q.dtype != np.float32:
            q = q.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        qq = q.copy()
        faiss.normalize_L2(qq)
        scores, ids = self.index.search(qq, top_k)
        out = []
        for i, s in zip(ids[0].tolist(), scores[0].tolist()):
            if i == -1:
                continue
            out.append((i, float(s)))
        return out
