from fastapi import APIRouter
from ..schemas import SearchRequest
from ..service import AppService

router = APIRouter(tags=["search"])

def get_service() -> AppService:
    from ..main import service
    return service

@router.post("/search")
def search(req: SearchRequest):
    hits = get_service().search(req.query, req.top_k)
    return {"query": req.query, "top_k": req.top_k, "sources": hits}
