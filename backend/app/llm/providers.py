from typing import List, Dict, Any, Optional
import os
import json
import httpx

from ..config import settings

class LLM:
    async def generate(self, query: str, context: str) -> str:
        return ""

class ExtractiveLLM(LLM):
    async def generate(self, query: str, context: str) -> str:
        lines = [x.strip() for x in context.splitlines() if x.strip()]
        if not lines:
            return "Kaynaklarda bu soruya dair yeterli içerik bulamadım."
        picked = []
        total = 0
        for ln in lines:
            if len(ln) < 40:
                continue
            if total + len(ln) > 1200:
                break
            picked.append(ln)
            total += len(ln)
            if len(picked) >= 8:
                break
        if not picked:
            picked = lines[:6]
        return "\n".join(["- " + p for p in picked])

class OpenAILLM(LLM):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def generate(self, query: str, context: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        system = (
            "You are a careful assistant that answers ONLY using the provided SOURCES. "
            "If the sources do not contain the answer, say you cannot find it. "
            "Write in Turkish. Prefer short paragraphs and bullet points."
        )
        user = f"SORU:\n{query}\n\nSOURCES:\n{context}\n\nCEVAP:"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"].strip()

class OllamaLLM(LLM):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, query: str, context: str) -> str:
        url = self.base_url + "/api/generate"
        prompt = (
            "Kaynaklara dayalı cevap ver. Sadece aşağıdaki SOURCES içeriğini kullan. "
            "Eğer kaynaklarda cevap yoksa açıkça 'bulamadım' de. Türkçe yaz.\n\n"
            f"SORU:\n{query}\n\nSOURCES:\n{context}\n\nCEVAP:"
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json={"model": self.model, "prompt": prompt, "stream": False})
            r.raise_for_status()
            data = r.json()
        return (data.get("response") or "").strip()

def make_llm() -> LLM:
    p = (settings.llm_provider or "").strip().lower()
    if p == "openai" and settings.openai_api_key.strip():
        return OpenAILLM(settings.openai_api_key.strip(), settings.openai_model.strip() or "gpt-4o-mini")
    if p == "ollama":
        return OllamaLLM(settings.ollama_base_url, settings.ollama_model)
    return ExtractiveLLM()
