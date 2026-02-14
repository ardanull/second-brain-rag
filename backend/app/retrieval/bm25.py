from typing import List, Tuple, Dict, Any
import re
from rank_bm25 import BM25Okapi

_tok = re.compile(r"[\w\-]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _tok.findall(text)]

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.meta = []
        self.corpus = []

    def build(self, texts: List[str], meta: List[Dict[str, Any]]):
        self.corpus = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.corpus)
        self.meta = meta

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            return []
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        idxs = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        out = []
        for i in idxs:
            out.append((i, float(scores[i])))
        return out
