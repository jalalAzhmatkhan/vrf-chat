# VRF/VRV Technical Chatbot

Chatbot teknis berbasis RAG (Retrieval-Augmented Generation) untuk sistem
VRF/VRV (Variable Refrigerant Flow/Volume), dibangun di atas 7 dokumen
service manual PDF (±400 halaman masing-masing, ≈2.581 halaman total).
Chatbot menjawab pertanyaan troubleshooting, wiring, kode error, dan
prosedur servis dengan referensi yang bisa dilacak balik ke halaman &
dokumen sumber aslinya — termasuk elemen non-teks (skema elektrikal, tabel,
gambar display alat, ikon tombol inline, flow diagram).

> Dokumentasi arsitektur & keputusan desain lengkap ada di
> `../Documentation/system-design/`. README ini adalah ringkasan orientasi
> untuk siapapun yang baru masuk ke repo `vrf-chat/`.

## Status Proyek

| Fase | Cakupan | Status |
|---|---|---|
| **Fase 0 — Foundation** | Scaffold FE/BE, abstraksi provider (LLM/object storage/DB/vector store) via `.env`, skema data inti, Docker/WSL, Authentication & RBAC | ✅ Selesai |
| **Fase 1 — Ingestion Pipeline** | PDF → canonical representation (Docling + PaddleOCR-VL cascade) → chunking → embedding (Qdrant) | ✅ Selesai |
| **Fase 2 — Chat Core** | Hybrid retrieval, Pydantic AI agent + tools, chat streaming (SSE), citation viewer | ⏳ Belum dimulai |
| **Fase 3 — Evaluation & Admin** | Dashboard evaluasi retrieval/generation, Manual Review Queue, halaman RBAC | ⏳ Belum dimulai |

Bukti nyata pipeline ingestion sudah berjalan end-to-end: 1 dokumen penuh
(*Zeggo VRV IV Service Manual REYQ*, 286 halaman) berhasil diproses —
2.916 elements, 1.417 chunk terstruktur, seluruhnya ter-embed (dense+sparse)
di Qdrant.

## Arsitektur Sistem

```mermaid
flowchart TB
    subgraph Client
        FE["React + Vite<br/>(frontend/)"]
    end

    subgraph Backend["vrf-chat/backend/"]
        API["FastAPI<br/>backend-api"]
        Worker["Celery Worker<br/>backend-worker"]
        WorkerGPU["Celery Worker (GPU)<br/>backend-worker-gpu<br/>— Docling parsing"]
    end

    subgraph PaddleService["backend/paddleocr-vl-service/"]
        Paddle["PaddleOCR-VL Service<br/>(venv & Dockerfile terisolasi)"]
    end

    subgraph Data["Data Layer"]
        PG[("PostgreSQL<br/>metadata, elements, chunks")]
        Qdrant[("Qdrant<br/>dense + sparse vectors")]
        Redis[("Redis<br/>Celery broker + rate limit")]
        Storage[("Object Storage<br/>MinIO / S3 / local<br/>PDF asli, image crops")]
        Neo4j[("Neo4j<br/>(opsional, Fase Later — Knowledge Graph)")]
    end

    FE -->|"REST + SSE (Fase 2)"| API
    API --> PG
    API --> Redis
    API -->|"enqueue ingestion job"| Worker
    Worker --> WorkerGPU
    WorkerGPU -->|"remote_api, VRAM terisolasi"| Paddle
    WorkerGPU --> Storage
    WorkerGPU --> PG
    Worker --> Qdrant
    API -.->|Fase Later| Neo4j

    style Paddle fill:#f9d5a7,stroke:#c47f1a
    style Neo4j fill:#e0e0e0,stroke:#999,stroke-dasharray: 5 5
```

**Prinsip desain kunci:**
- **Plug-n-play provider** — LLM (Anthropic/Gemini/OpenAI/Local), object
  storage (MinIO/S3/local), database engine (Postgres/MySQL), dan vector
  store (Qdrant default, Chroma/Milvus opsional) semuanya dikonfigurasi
  murni lewat `.env`, tanpa ubah kode.
- **PaddleOCR-VL sebagai service terpisah** — awalnya direncanakan satu
  proses dengan Docling, tapi PyTorch (Docling) dan PaddlePaddle-GPU
  (PaddleOCR-VL) punya konflik ABI CUDA/NCCL yang tidak bisa hidup di venv
  yang sama. `backend-worker-gpu` (Docling) dan `paddleocr-vl-service`
  berjalan sebagai container terpisah, saling komunikasi lewat HTTP
  (`PADDLE_OCR_VL_BACKEND=remote_api`, bahkan untuk dev lokal).
- **`DOCLING_DEVICE` default `cpu`** — GPU RTX 3060 di lingkungan dev
  hanya 6GB VRAM; `paddleocr-vl-service` sendiri sudah memakai ~5.9-6GB
  saat inferensi, sehingga tidak ada ruang aman untuk Docling ikut memakai
  GPU secara bersamaan. Docling GPU hanya memberi speedup ~1.4x (bukan
  3-10x seperti estimasi awal), jadi trade-off ini diterima demi keamanan
  VRAM.

## Pipeline Ingestion (Fase 1)

```mermaid
flowchart LR
    PDF["PDF sumber<br/>(source-documents/, read-only)"]
    Probe["Stage 1<br/>Native Probe<br/>(PyMuPDF)"]
    Docling["Stage 2<br/>Docling Parser<br/>(struktur, 100% halaman)"]
    Trigger{"Stage 3<br/>Cascade Trigger<br/>table_score < 0.90?<br/>text_confidence < 0.6?<br/>region vector/raster?"}
    Paddle["Stage 4<br/>PaddleOCR-VL<br/>(table reparse, visual description,<br/>OCR halaman scan)"]
    Store["Canonical Store<br/>Postgres: pages/elements<br/>Object Storage: image crops"]
    KG["KG Candidate Extractor<br/>(entities/relations + provenance,<br/>disimpan, belum dimuat ke Neo4j)"]
    Chunk["Hierarchical Chunker<br/>5 tipe: text/table/figure/procedure/entity"]
    Embed["Embedder<br/>dense + sparse → Qdrant<br/>collection vrf_chunks"]

    PDF --> Probe --> Docling --> Trigger
    Trigger -->|ya| Paddle --> Store
    Trigger -->|tidak, confidence cukup| Store
    Store --> KG
    Store --> Chunk --> Embed

    style Trigger fill:#fff3cd,stroke:#997404
    style Paddle fill:#f9d5a7,stroke:#c47f1a
```

**Idempotency**: setiap tahap dikunci `document_hash`/`page_hash`/
`element_hash` — re-ingest PDF yang identik akan di-skip total di level
trigger (`created=False`), re-run manual untuk 1 dokumen yang sudah ada
akan skip penulisan DB per-halaman yang hash-nya cocok (Docling/PaddleOCR-VL
tetap dijalankan ulang, hanya insert/update DB yang dihindari).

**Requirement kritis yang dijaga pipeline ini** (lihat `CLAUDE.md` §4 di
root proyek): ikon/tombol inline tetap terasosiasi ke kalimat induknya
(termasuk kasus lintas halaman), struktur tabel tetap ter-query (bukan
flatten jadi teks acak, termasuk hasil PaddleOCR-VL yang formatnya HTML
bukan markdown), dan traceability penuh `chunk → element → page → document`.

## Sequence Flow: Autentikasi & Otorisasi

```mermaid
sequenceDiagram
    actor User
    participant FE as Frontend (React)
    participant API as Backend API
    participant Redis
    participant DB as PostgreSQL

    User->>FE: input username + password
    FE->>API: POST /api/v1/auth/login
    API->>Redis: cek rate limit percobaan login
    alt rate limit terlampaui
        API-->>FE: 429 + retry_after_seconds + header Retry-After
        FE-->>User: tampilkan countdown
    else kredensial salah
        API-->>FE: 401 Invalid username or password
        FE-->>User: tampilkan pesan error, clear password field
    else sukses
        API->>DB: verifikasi bcrypt hash (cost factor 14, thread-pool offload)
        API->>DB: buat refresh token (rotasi + reuse detection, token_family_id)
        API-->>FE: 200 + access_token (JWT, di memori) + Set-Cookie httpOnly refresh_token
        FE->>API: GET /api/v1/auth/me
        API-->>FE: scopes user
        FE->>FE: filterNavByScopes(scopes) — menu admin hide-total jika scope kurang
        FE-->>User: redirect ke app shell
    end

    Note over FE,API: Request berikutnya ke endpoint scope-gated
    FE->>API: request + Authorization: Bearer <access_token>
    alt token expired/invalid
        API-->>FE: 401
        FE->>API: POST /api/v1/auth/refresh (credentials: include, cookie httpOnly)
        API->>DB: validasi refresh token, deteksi reuse (revoke seluruh family jika reuse)
        API-->>FE: 200 + access_token baru + Set-Cookie refresh_token baru
        FE->>API: retry request asli
    else token valid tapi scope kurang
        API-->>FE: 403 Forbidden + WWW-Authenticate error="insufficient_scope"
        FE-->>User: tampilkan halaman "tidak punya akses" (tanpa retry-refresh)
    end
```

## Sequence Flow: Trigger Ingestion Dokumen

```mermaid
sequenceDiagram
    actor Admin
    participant FE as Frontend
    participant API as Backend API
    participant DB as PostgreSQL
    participant Queue as Celery (Redis broker)
    participant WorkerGPU as backend-worker-gpu
    participant Paddle as paddleocr-vl-service
    participant Storage as Object Storage
    participant Qdrant

    Admin->>FE: upload PDF (scope documents:write)
    FE->>API: POST /api/v1/documents (multipart)
    API->>API: hitung document_hash dari byte PDF
    API->>DB: cek documents.source_hash sudah ada?
    alt dokumen identik sudah pernah diingest
        API-->>FE: 202 {status: "already_ingested", document_id}
    else dokumen baru
        API->>DB: insert documents row (status="pending")
        API->>Queue: enqueue run_ingestion_task
        API-->>FE: 202 {status: "queued", document_id, ingestion_job_id}

        Queue->>WorkerGPU: run_ingestion_task(document_id)
        WorkerGPU->>WorkerGPU: Stage 1 (native probe) + Stage 2 (Docling, CPU default)
        WorkerGPU->>WorkerGPU: Stage 3 (cascade trigger rules)
        opt elemen di-queue Stage 3
            WorkerGPU->>Paddle: POST /describe_figure / /reparse_table / /ocr_page
            Paddle-->>WorkerGPU: visual_description / structured table / OCR text
        end
        WorkerGPU->>Storage: simpan image crop
        WorkerGPU->>DB: simpan pages/elements (idempotent via page_hash/element_hash)
        WorkerGPU->>DB: simpan kg_candidate_entities/relations (provenance)
        WorkerGPU->>WorkerGPU: hierarchical chunking (5 tipe chunk)
        WorkerGPU->>Qdrant: upsert dense+sparse vectors (collection vrf_chunks)
        WorkerGPU->>DB: update documents.status = "ready"

        FE->>API: GET /api/v1/documents/{id}/ingestion-jobs (polling)
        API-->>FE: status progres job
    end
```

## Struktur Repo

```
vrf-chat/
├── backend/                        # FastAPI + Celery, Python (uv)
│   ├── app/
│   │   ├── agent/                  # Pydantic AI agent (Fase 2)
│   │   ├── api/v1/                 # auth, admin_rbac, documents, health, internal_storage
│   │   ├── auth/                   # JWT, refresh token rotation, RBAC scopes
│   │   ├── core/                   # config (Settings), observability
│   │   ├── db/                     # SQLAlchemy models, session/engine factory
│   │   ├── domain/                 # kamus istilah VRF/VRV
│   │   ├── evaluation/             # metrik retrieval/generation (Fase 3)
│   │   ├── ingestion/              # native_probe, docling_parser, cascade_trigger,
│   │   │                           #   paddleocr_vl_cascade, canonical_store, chunker,
│   │   │                           #   kg_candidate_extractor, embedder, orchestrator
│   │   ├── knowledge_graph/        # stub (Fase Later — Neo4j)
│   │   ├── llm_providers/          # factory Anthropic/Gemini/OpenAI/Local
│   │   ├── retrieval/              # VectorStoreClient (Qdrant)
│   │   ├── storage/                # ObjectStorageClient (MinIO/S3/local)
│   │   └── workers/                # Celery task definitions
│   ├── paddleocr-vl-service/       # service terisolasi (venv & Dockerfile sendiri)
│   ├── alembic/                    # migrations (termasuk seed data roles/scopes)
│   └── tests/{unit,integration}/
├── frontend/                       # React + Vite
│   └── src/{auth,components,hooks,lib,pages,routes,shell,styles}/
└── docker-compose.yml              # backend-api, backend-worker, backend-worker-gpu,
                                     #   paddleocr-vl-service, redis, postgres, qdrant,
                                     #   minio, neo4j (profile "kg", tidak default), frontend
```

## API Endpoint (Terimplementasi)

| Method | Path | Scope | Keterangan |
|---|---|---|---|
| `GET` | `/api/v1/health` | — | Health check |
| `POST` | `/api/v1/auth/login` | — | Login, rate-limited |
| `POST` | `/api/v1/auth/refresh` | — | Refresh access token (cookie httpOnly) |
| `POST` | `/api/v1/auth/logout` | — | Revoke refresh token |
| `GET` | `/api/v1/auth/me` | — | Info user + scopes |
| `GET`/`PUT` | `/api/v1/admin/rbac/roles/{id}/scopes` | `admin:rbac:read`/`write` | Permission Management |
| `GET`/`POST`/`PATCH` | `/api/v1/admin/rbac/users` | `admin:rbac:read`/`write` | User Management |
| `POST` | `/api/v1/documents` | `documents:write` | Trigger ingestion (multipart upload) |
| `GET` | `/api/v1/documents/{id}/ingestion-jobs` | `documents:write` | Status job ingestion |

## Menjalankan Secara Lokal

> **Docker hanya boleh dijalankan dari WSL**, tidak pernah dari Windows
> native (lihat `CLAUDE.md` §5 di root proyek).

```bash
# dari shell WSL, di dalam vrf-chat/
docker compose up -d                        # service inti (tanpa GPU worker/neo4j)
docker compose --profile gpu up -d          # + backend-worker-gpu, paddleocr-vl-service
docker compose --profile kg up -d neo4j     # opsional, Fase Later

# migrasi database (sekali di awal, atau setelah migration baru)
docker compose exec backend-api uv run alembic upgrade head
```

Backend: `cd backend && uv sync && uv run uvicorn app.main:app --reload`
Frontend: `cd frontend && npm ci && npm run dev`

Konfigurasi provider (LLM/object storage/DB/vector store) lewat
`backend/.env` — lihat `backend/.env.example` untuk daftar variabel
lengkap per provider.

## Referensi Lebih Lanjut

- Arsitektur & keputusan desain: `../Documentation/system-design/`
- Roadmap & task breakdown: `../Documentation/project-milestones/`
- Laporan QA: `../Documentation/qa-reports/`
- Desain UI/UX: `../Documentation/ui-ux-design/`
