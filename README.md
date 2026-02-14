# Second Brain RAG (Kaynak Gösteren Not Asistanı)

Bu repo kişisel doküman kütüphaneni (PDF / Markdown / TXT) indeksleyip sorulara yalnızca kaynaklardan beslenerek cevap veren, kaynak alıntıları ve doğrulama odaklı bir RAG sistemidir.

## Özellikler
- PDF / Markdown / TXT yükleme ve indeksleme
- Parçalama (chunking) + metadata (sayfa, bölüm, dosya) saklama
- Vektör arama (FAISS) + keyword arama (BM25) ile **hybrid retrieval**
- Opsiyonel reranking (CrossEncoder veya provider tabanlı)
- Kaynak gösterimli cevap formatı (alıntı kartları + referans numaraları)
- “Kaynak yoksa cevap üretme” kuralı (hallucination guard)
- De-dup (yakın benzer chunk’lar) + indeks tutarlılık kontrolleri
- Basit değerlendirme: soru seti ile recall@k / precision@k metrikleri

## Hızlı Başlangıç

### 1) Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

### 2) UI (Streamlit)
```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

UI varsayılan olarak `http://localhost:8000` backend’ine bağlanır.

## Konfigürasyon

Backend `.env` değişkenleri:
- `DATA_DIR`: çalışma dizini (default: `./data`)
- `DB_PATH`: sqlite veritabanı yolu
- `EMBED_MODEL`: embedding modeli (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `TOP_K`: retrieval için temel k
- `HYBRID_ALPHA`: hybrid skorlama karışımı (0..1)
- `MAX_CONTEXT_CHARS`: modele verilecek context sınırı
- `LLM_PROVIDER`: `openai` veya `ollama` (boş bırakılırsa extractive fallback)
- OpenAI için:
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (default: `gpt-4o-mini`)
- Ollama için:
  - `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
  - `OLLAMA_MODEL` (örn: `llama3.1`)

## API
- `POST /documents/upload` dosya yükler ve indeksler
- `GET /documents` dokümanları listeler
- `POST /chat` soru sorar, kaynakları döndürür
- `POST /search` sadece retrieval (cevap üretmeden)
- `POST /eval/run` eval setiyle metrik üretir

## Notlar
- Embedding ve FAISS index `DATA_DIR/index` altında tutulur.
- Metin çıkarımı için PDF’de `pymupdf` kullanılır.
- Büyük PDF’lerde ilk indeksleme sürebilir.

