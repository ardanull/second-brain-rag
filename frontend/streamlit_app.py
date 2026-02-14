import os
import json
import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Second Brain RAG", layout="wide")

st.title("Second Brain RAG")

left, right = st.columns([1, 2], gap="large")

def api_get(path: str):
    with httpx.Client(timeout=60.0) as client:
        r = client.get(API_BASE + path)
        r.raise_for_status()
        return r.json()

def api_post(path: str, payload):
    with httpx.Client(timeout=60.0) as client:
        r = client.post(API_BASE + path, json=payload)
        r.raise_for_status()
        return r.json()

def api_upload(file):
    with httpx.Client(timeout=120.0) as client:
        files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
        r = client.post(API_BASE + "/documents/upload", files=files)
        r.raise_for_status()
        return r.json()

with left:
    st.subheader("Kütüphane")
    up = st.file_uploader("PDF / Markdown / TXT yükle", type=["pdf", "md", "txt"])
    if up is not None:
        if st.button("Yükle ve İndeksle", use_container_width=True):
            try:
                doc = api_upload(up)
                st.success(f"İndekslendi: {doc.get('original_name')}")
            except Exception as e:
                st.error(str(e))

    if st.button("Yenile", use_container_width=True):
        st.rerun()

    try:
        docs = api_get("/documents")
    except Exception as e:
        docs = []
        st.error(f"Backend erişilemiyor: {e}")

    if docs:
        for d in docs:
            st.markdown(f"**{d.get('original_name')}**")
            st.caption(f"id: {d.get('id')} · chunks: {d.get('chunks')} · sha: {d.get('sha256')[:10]}…")
    else:
        st.info("Henüz doküman yok.")

with right:
    st.subheader("Soru Sor")
    q = st.text_input("Soru", placeholder="Örn: Bu dokümanlarda RAG için en iyi chunk boyutu ne?")
    top_k = st.slider("Top K", min_value=3, max_value=20, value=8, step=1)
    mode = st.selectbox("Mod", ["Cevap üret", "Sadece ara"], index=0)
    if st.button("Çalıştır", use_container_width=True) and q.strip():
        if mode == "Sadece ara":
            try:
                res = api_post("/search", {"query": q, "top_k": top_k})
                st.markdown("### Sonuçlar")
                for i, s in enumerate(res.get("sources", []), start=1):
                    with st.expander(f"[{i}] {s.get('original_name')} · chunk {s.get('chunk_index')} · score {s.get('score'):.3f}", expanded=(i<=2)):
                        st.write(s.get("text", ""))
                        meta = {k: s.get(k) for k in ["doc_id","chunk_id","page_start","page_end","section","vec_score","bm25_score"]}
                        st.json(meta)
            except Exception as e:
                st.error(str(e))
        else:
            try:
                res = api_post("/chat", {"query": q, "top_k": top_k, "style": "concise", "include_sources": True})
                st.markdown("### Cevap")
                st.write(res.get("answer", ""))
                st.markdown("### Kaynaklar")
                for i, s in enumerate(res.get("sources", []), start=1):
                    with st.expander(f"[{i}] {s.get('original_name')} · chunk {s.get('chunk_index')} · score {s.get('score'):.3f}", expanded=(i<=3)):
                        st.write(s.get("text", ""))
                        meta = {k: s.get(k) for k in ["doc_id","chunk_id","page_start","page_end","section"]}
                        st.json(meta)
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Eval")
    st.caption("JSON formatında soru seti yükle: { items: [ { question, expected_doc_ids, expected_chunk_ids } ], top_k }")
    sample = {
        "items": [
            {"question": "Bu dokümanlar ne anlatıyor?", "expected_doc_ids": [], "expected_chunk_ids": []}
        ],
        "top_k": 8
    }
    raw = st.text_area("Eval JSON", value=json.dumps(sample, ensure_ascii=False, indent=2), height=220)
    if st.button("Eval Çalıştır", use_container_width=True):
        try:
            payload = json.loads(raw)
            res = api_post("/eval/run", payload)
            st.json(res)
        except Exception as e:
            st.error(str(e))
