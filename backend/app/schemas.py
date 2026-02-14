from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class DocumentOut(BaseModel):
    id: str
    filename: str
    original_name: str
    mime_type: str
    bytes: int
    sha256: str
    created_at: str
    chunks: int = 0

class SourceSpan(BaseModel):
    chunk_id: str
    doc_id: str
    original_name: str
    filename: str
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    score: float
    text: str

class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = 8

class SearchResponse(BaseModel):
    query: str
    top_k: int
    sources: List[SourceSpan]

class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = 8
    style: str = "concise"
    include_sources: bool = True

class ChatAnswer(BaseModel):
    answer: str
    sources: List[SourceSpan]
    refused: bool = False
    reason: str = ""

class EvalItem(BaseModel):
    question: str
    expected_doc_ids: List[str] = []
    expected_chunk_ids: List[str] = []

class EvalRequest(BaseModel):
    items: List[EvalItem]
    top_k: int = 8

class EvalMetrics(BaseModel):
    top_k: int
    count: int
    precision_at_k: float
    recall_at_k_docs: float
    recall_at_k_chunks: float
    per_item: List[Dict[str, Any]]
