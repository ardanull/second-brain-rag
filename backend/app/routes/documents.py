from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from ..service import AppService

router = APIRouter(prefix="/documents", tags=["documents"])

def get_service() -> AppService:
    from ..main import service
    return service

@router.get("")
def list_docs():
    return get_service().list_documents()

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    doc = get_service().upload_and_index(file.filename or "file", file.content_type or "application/octet-stream", content)
    return doc
