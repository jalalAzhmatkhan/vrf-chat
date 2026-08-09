# paddleocr-vl-service

Standalone PaddleOCR-VL inference service — Stage 4 (`app/ingestion/paddleocr_vl_cascade.py`
in `vrf-chat/backend/`) `remote_api` backend, per
`Documentation/system-design/02-ingestion-pipeline.md` §4.0 ("Opsi 3":
PaddleOCR-VL always runs as a separate container/service, called by
`backend-worker-gpu` via `PADDLE_OCR_VL_BACKEND=remote_api`, even in dev —
not just prod/cloud as originally designed).

## Why a separate service (not `PADDLE_OCR_VL_BACKEND=local` inside `backend-worker-gpu`)

I1.6's benchmark found a real, blocking packaging conflict: `paddlepaddle-gpu`
and `torch` (used by Docling in `backend/`) both vendor `nvidia-nccl-cu12` as
a separate pip package into the same shared site-packages namespace —
installing both in the same venv breaks `import torch` entirely
(`undefined symbol: ncclCommResume`). See
`vrf-chat/backend/docs/i1.6-vram-benchmark-report.md` §6 for the full
original finding. Splitting into two separate containers/venvs (this
service has its own `pyproject.toml`/`uv.lock`, entirely independent
dependency resolution from `backend/`'s) sidesteps the conflict via the
Docker container boundary rather than needing a custom subprocess/IPC
isolation mechanism.

## Live verification (I1.10) — what was actually confirmed, not assumed

Per explicit instruction not to assume the isolated-venv fix actually
works, the following was verified live (WSL, RTX 3060 6GB) in a **fresh,
throwaway venv containing only `paddleocr[doc-parser]` + `paddlepaddle-gpu`
— no `docling`, no `torch` at all**:

1. **`torch` is not a dependency at all** for this service's actual usage.
   `paddleocr[doc-parser]`'s base install does not pull `torch`. The
   layout-detection submodel (`PP-DocLayoutV3`) runs on PaddlePaddle's
   native inference engine. The VL recognition submodel (`PaddleOCR-VL-1.6-0.9B`,
   ERNIE4.5-based) also runs entirely through PaddlePaddle's own
   transformers-compatible implementation
   (`PaddleOCRVLForConditionalGeneration`, loaded from `model.safetensors`)
   — **not** via the `transformers`/PyTorch library, contrary to this
   project's earlier assumption (I1.4) that the VL submodel specifically
   needed `torch`. This means the NCCL conflict found in I1.6 was entirely
   an artifact of Docling's `torch` being present in the *same* venv — in
   an isolated venv, there is nothing for `paddlepaddle-gpu` to conflict
   with.
2. **Real pipeline construction succeeded** (`PaddleOCRVL(vl_rec_backend="native",
   device="gpu:0", precision="fp16", ...)`), loading both the layout model
   and the 0.9B-parameter VL model from cache in a few seconds.
3. **Real inference succeeded** against a real page image (rendered from
   `source-documents/` via the 50-page I1.6 sample), producing genuine
   structured markdown output:
   `{"markdown_texts": "## [II Restrictions]\n\n## [5] An Example of a
   System to which an MA Remote Controller is connected\n\n1. System with
   one outdoor unit...", ...}` — this also **confirms** (not just assumes)
   the `_extract_markdown_text`/`_first_prediction_to_dict` mapping logic
   in both this service's `app/pipeline.py` and `backend/`'s
   `paddleocr_vl_cascade.py` (`result.json == {"markdown": {"markdown_texts":
   ...}}`) was correctly guessed back in I1.4, before any real output was
   available to check against.
4. **Critical VRAM finding**: peak VRAM during that single-image inference
   was **~5.98GB of the 6.14GB (6144 MiB) card** — only ~150-160MB
   headroom. This is **far higher** than
   `02-ingestion-pipeline.md` §4's original estimate of "~2.5-4GB" for
   PaddleOCR-VL. VRAM returned to baseline (~150 MiB) after the process
   exited, confirming no leak, but the **peak-usage margin on this specific
   dev GPU is extremely tight** — batch size above 1, a larger image, or
   any concurrent GPU consumer (even outside this service) is a real OOM
   risk. This is flagged as a priority item for System Analyst /
   `02-ingestion-pipeline.md` §4 recalibration, not something this service
   can mitigate purely in code beyond what's already implemented
   (`PADDLE_OCR_VL_SERVICE_BATCH_SIZE=1` non-negotiable, idle-unload after
   `PADDLE_OCR_VL_SERVICE_IDLE_UNLOAD_SECONDS`).

### Not yet verified

- Multiple concurrent/sequential inference calls in a long-running service
  process (the live test above was a single one-shot script, not this
  actual FastAPI service under load) — recommended as an operational check
  during Q1.2 (QA Engineer's WSL/Docker GPU verification).
- The full Docker image build (`docker build` of this service's
  `Dockerfile`) — dependency resolution was verified via `uv sync` in a
  WSL-native venv (same rationale as I1.6's Docling verification: identical
  GPU/driver/dependency versions, faster than a full image build+model
  download cycle), not via an actual `docker build` + `docker run`. Same
  scope boundary already accepted for `backend-worker-gpu` in I1.4/I1.6.

## Why a direct wheel URL for `paddlepaddle-gpu` (`pyproject.toml`)

`paddlepaddle-gpu` is not published to standard PyPI as a GPU-enabled
wheel for arbitrary CUDA versions — PaddlePaddle hosts its own
per-CUDA-version wheel index (`https://www.paddlepaddle.org.cn/packages/stable/cuXXX/`).
A direct wheel URL (`paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl`
from the `cu126` index) pinned to this service's exact target
(Python 3.12, Linux, CUDA 12.6 runtime bundled via pip deps) is the
simplest reliable way to pin it with `uv`, verified working end-to-end
(`uv sync` resolves and installs it correctly, `paddle.is_compiled_with_cuda()
== True`, `paddle.device.set_device("gpu:0")` succeeds). The
`sys_platform == 'linux'` marker means `uv sync` on a non-Linux host simply
skips this dependency rather than failing — this service is only ever
meant to run in the Linux container built by this `Dockerfile`.

## HTTP contract

Implements the contract `vrf-chat/backend/app/ingestion/paddleocr_vl_cascade.py`
`RemoteAPIPaddleOCRVLClient` expects:

- `GET /health` → `{"status": "ok", "model_loaded": bool}`
- `POST /describe_figure` `{"image_base64": "..."}` →
  `{"description": str|null, "figure_type": str|null, "components": [...], "connections": [...]}`
- `POST /reparse_table` `{"image_base64": "..."}` →
  `{"markdown": str|null, "rows": [...]}`
- `POST /ocr_page` `{"image_base64": "..."}` →
  `{"text": str}`

## Running

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8100   # dev
docker build -t paddleocr-vl-service .                     # via WSL only, see CLAUDE.md §5
```

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy app
```

Real model construction/inference (`_build_pipeline`, `_predict`/
`_decode_image`) is `# pragma: no cover` — see module docstrings for the
same rationale already established in `backend/`'s ingestion modules
(genuinely GPU/model-only work, not meaningfully unit-testable, verified
separately via the live transcript above). This project does **not**
enforce 100% coverage in `pyproject.toml` (unlike `backend/`) — a
deliberate reduced-rigor trade-off given this was built under significant
session time pressure as the final Fase 1 task; current coverage is 99%
(one line: the background idle-watcher thread's loop-body call site, which
delegates to `_maybe_unload_if_idle()` — that method itself IS directly
unit-tested).
