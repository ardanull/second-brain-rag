import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

_ws = re.compile(r"\s+")
_sentence = re.compile(r"(?<=[.!?])\s+")

@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None

def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = _ws.sub(" ", s).strip()
    return s

def split_sentences(text: str) -> List[str]:
    t = normalize_text(text)
    if not t:
        return []
    parts = _sentence.split(t)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out

def chunk_by_sentences(
    text: str,
    chunk_size: int = 900,
    overlap: int = 120,
    hard_limit: int = 1400
) -> List[str]:
    sents = split_sentences(text)
    if not sents:
        return []
    chunks = []
    buf = []
    buf_len = 0
    for sent in sents:
        sl = len(sent)
        if buf and buf_len + 1 + sl > chunk_size:
            chunk = " ".join(buf).strip()
            if chunk:
                chunks.append(chunk[:hard_limit])
            keep = []
            keep_len = 0
            if overlap > 0:
                for prev in reversed(buf):
                    if keep_len + 1 + len(prev) > overlap:
                        break
                    keep.append(prev)
                    keep_len += 1 + len(prev)
                keep = list(reversed(keep))
            buf = keep
            buf_len = sum(len(x) + 1 for x in buf)
        buf.append(sent)
        buf_len += sl + 1
    last = " ".join(buf).strip()
    if last:
        chunks.append(last[:hard_limit])
    return chunks

def soft_dedup(texts: List[str], threshold: float = 0.92) -> List[str]:
    try:
        from rapidfuzz.fuzz import ratio
    except Exception:
        return texts
    kept = []
    for t in texts:
        nt = normalize_text(t)
        if not nt:
            continue
        is_dup = False
        for k in kept[-50:]:
            if ratio(nt, k) / 100.0 >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(nt)
    return kept
