from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .service import AppService
from .routes.documents import router as documents_router
from .routes.search import router as search_router
from .routes.chat import router as chat_router
from .routes.eval import router as eval_router

service = AppService()

app = FastAPI(title="Second Brain RAG", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(eval_router)

@app.get("/health")
def health():
    return {"ok": True}
