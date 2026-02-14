import os
import hashlib
from typing import Tuple

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def safe_filename(name: str) -> str:
    name = name.replace("\\", "/").split("/")[-1]
    out = []
    for ch in name:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    if not s:
        s = "file"
    return s[:180]

def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
