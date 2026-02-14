from fastapi import APIRouter
from ..schemas import ChatRequest
from ..service import AppService

router = APIRouter(tags=["chat"])

def get_service() -> AppService:
    from ..main import service
    return service

@router.post("/chat")
async def chat(req: ChatRequest):
    out = await get_service().chat(req.query, req.top_k, req.style)
    return out
