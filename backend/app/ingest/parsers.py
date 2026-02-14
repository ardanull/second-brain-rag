from typing import List, Tuple, Optional
import fitz

def parse_pdf(path: str) -> List[Tuple[str, int]]:
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text")
        pages.append((text, i + 1))
    doc.close()
    return pages

def parse_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
