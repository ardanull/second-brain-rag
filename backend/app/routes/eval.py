from fastapi import APIRouter, HTTPException
from ..schemas import EvalRequest
from ..service import AppService

router = APIRouter(prefix="/eval", tags=["eval"])

def get_service() -> AppService:
    from ..main import service
    return service

@router.post("/run")
def run_eval(req: EvalRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="no items")
    items = [it.model_dump() for it in req.items]
    m = get_service().build_eval_metrics(items, req.top_k)
    return m
