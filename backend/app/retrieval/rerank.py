from typing import List, Dict, Any, Tuple
import os
import httpx

class Reranker:
    def rerank(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return items

class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pairs = [(query, it["text"]) for it in items]
        scores = self.model.predict(pairs)
        out = []
        for it, s in zip(items, scores):
            it2 = dict(it)
            it2["rerank_score"] = float(s)
            out.append(it2)
        out.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return out

class OllamaReranker(Reranker):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def rerank(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not items:
            return items
        prompt = self._prompt(query, items)
        url = self.base_url + "/api/generate"
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(url, json={"model": self.model, "prompt": prompt, "stream": False})
                r.raise_for_status()
                txt = r.json().get("response", "")
        except Exception:
            return items
        order = self._parse(txt, len(items))
        if not order:
            return items
        ordered = []
        used = set()
        for idx in order:
            if 0 <= idx < len(items) and idx not in used:
                ordered.append(items[idx])
                used.add(idx)
        for i in range(len(items)):
            if i not in used:
                ordered.append(items[i])
        return ordered

    def _prompt(self, query: str, items: List[Dict[str, Any]]) -> str:
        lines = []
        lines.append("You are ranking passages for a question. Return ONLY a JSON array of indices from best to worst.")
        lines.append("Question:")
        lines.append(query)
        lines.append("Passages:")
        for i, it in enumerate(items):
            t = it["text"].replace("\n", " ").strip()
            if len(t) > 900:
                t = t[:900]
            lines.append(f"[{i}] {t}")
        lines.append("Return JSON array now.")
        return "\n".join(lines)

    def _parse(self, s: str, n: int):
        import json
        s2 = s.strip()
        try:
            arr = json.loads(s2)
            if isinstance(arr, list):
                out = []
                for x in arr:
                    try:
                        out.append(int(x))
                    except Exception:
                        pass
                return out
        except Exception:
            pass
        nums = []
        for m in __import__("re").findall(r"\b\d+\b", s2):
            try:
                v = int(m)
                if 0 <= v < n:
                    nums.append(v)
            except Exception:
                pass
        if nums:
            seen = set()
            out = []
            for v in nums:
                if v not in seen:
                    out.append(v)
                    seen.add(v)
            return out
        return []
