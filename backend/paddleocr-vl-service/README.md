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

## ⚠️ [FIXED, SA1.2 2026-08-09] `visual_description.description` was NULL for every real element

System Analyst's SA1.2 spot-check against real ingested data (document_id=3,
286-page Zeggo VRV IV REYQ) found `visual_description.description` NULL for
**386/386** (100%) of the elements this service actually processed via
`describe_figure`/`reparse_table` — the entire point of Stage 4
(icon/figure becomes *searchable*, `CLAUDE.md` §4) was silently defeated,
despite every cascade call genuinely executing and the pipeline reporting
success end-to-end.

**Root cause**: `app/pipeline.py` `_first_prediction_to_dict` checked
`.json` before `.markdown`. Reading the real `paddlex` source
(`paddlex/inference/common/result/mixin.py` `JsonMixin`) shows `.json` is
**always** `{"res": <the whole result, recursively serialized>}` for every
PaddleX pipeline using this mixin (including `PaddleOCRVLResult`) — it
never has a top-level `"markdown"`/`"text"` key. Since `.json` is a dict,
the old `isinstance(as_json, dict): return as_json` branch always won,
permanently shadowing the real `.markdown` property (which DOES return
`{"markdown_texts": "...", ...}`, per
`paddlex/inference/pipelines/paddleocr_vl/result.py` `_to_markdown`). The
"Live verification" transcript below (`result.json ==
{"markdown": {"markdown_texts": ...}}`) was **inaccurate as written** — it
came from printing `.markdown` directly during manual verification, not
from exercising `_first_prediction_to_dict` itself, which is why the
discrepancy wasn't caught at the time.

**Fix**: `.markdown` is now checked first (see `_first_prediction_to_dict`
docstring for the full explanation + a regression test asserting this
ordering: `test_first_prediction_to_dict_prefers_markdown_over_json_without_markdown_key`).
Same fix applied to the structurally-identical function in
`backend/app/ingestion/paddleocr_vl_cascade.py`. Re-verified against a
clean re-ingest of document_id=3 — see
`backend/docs/i1.10-e2e-findings.md` for the post-fix results.

## ⚠️ [F2/F3, QA phase-1-qa-report.md §1.4/§2.5, 2026-08-10] Residual 5/386 NULLs — root-caused, NOT a mapping bug, no safe fix shipped yet

QA re-verification found the daemon serving `/describe_figure` at the time
of their audit predated the `_first_prediction_to_dict` fix above (F3 —
`uvicorn` without `--reload` never picks up file changes; the process had
simply never been restarted since before commit `fa02029`). **Fixed**: the
daemon is now restarted after every code change that touches this service
(a process discipline note, not a code change) — confirmed live:
`/describe_figure` for element 5184 (a real, content-rich piping schematic
QA specifically flagged) still returned `description: null` even
**against a freshly-restarted, current-code daemon** — proving the residual
5/386 NULLs are independent of daemon staleness.

**Root cause of the residual 5/386, confirmed via direct model
instrumentation** (calling `pipeline.predict()` directly, bypassing the
HTTP layer, printing the raw `.json`/`.markdown` PaddleX `Result`): for
these specific elements (large, full-bleed technical schematics filling
nearly the entire cropped region — see QA's table, elements 5184/5186/5188
in particular), the layout-detection submodel (`PP-DocLayoutV3`) finds
**no image/chart/table region at all** — `parsing_res_list` contains only
the page footer text (or, at very low `layout_threshold` values down to
0.05, a couple of trivial incidental text fragments like a stray page
number). Since markdown rendering only emits content for recognized
layout blocks, no VL recognition ever runs on the actual diagram, and
`markdown_texts` stays empty. This is a genuine **upstream layout-model
limitation** on this specific class of content (dense, edge-to-edge
technical line-art with no "photo-like" white-space boundary), not a
code/mapping bug — the SA1.2-fixed ordering bug is confirmed fully
resolved for the other 381/386 (98.7%) elements.

**Fallback investigated, NOT shipped (safety judgment call)**: forcing
`use_layout_detection=False` with an explicit `prompt_label` (e.g.
`"chart"`) bypasses layout detection and treats the whole crop as one
region — this does appear to eventually process real content (GPU
actively computing, no errors), but at a dramatically higher latency than
the normal per-region path: none of several live attempts (1024 tokens,
256 tokens, 256 tokens on an image downscaled to an 800px long edge)
completed within 7-22+ minutes before being killed — versus ~5-90s for the
normal layout-detection-enabled path. Shipping this as an automatic
fallback without a proven latency bound risks making Stage 4 pathologically
slow (or blocking the whole cascade stage) for exactly the elements it's
meant to help — a worse outcome than the current graceful "stays NULL,
not searchable" degradation for this small (~1.3%) subset. **Left as
explicitly open follow-up work**, not silently dropped: candidate next
steps are tuning generation sampling params, trying an older
`pipeline_version`, or a lighter-weight non-PaddleOCR-VL captioning model
for this narrow fallback case specifically.

## ⚠️ [F4, QA phase-1-qa-report.md §2.1, 2026-08-10] Official `docker compose --profile gpu` path — now verified, 3 real issues found and fixed/documented

Prior to this, `docker compose --profile gpu build`/`up` had **never
actually been run** for this service or `backend-worker-gpu` — every prior
"live verification" (I1.6, I1.10, the SA1.2 re-ingest) used a bare WSL
process/script directly, despite the whole point of the Opsi 3 decision
being a container boundary. QA flagged this gap explicitly; here is what
running it for the first time actually found:

1. **Build-context bug (fixed)**: `backend/.dockerignore` didn't account
   for this service having moved to `backend/paddleocr-vl-service/`
   (SA1.2 folder-structure review) — `backend-api`/`backend-worker`'s
   build context was transferring this service's ~7GB `.venv` on every
   build. Fixed by adding an explicit `paddleocr-vl-service/` exclusion to
   `backend/.dockerignore`. Separately, **this service never had its own
   `.dockerignore` at all** (a genuine, previously-undiscovered gap — its
   own Docker build had simply never been attempted) — added one, `.venv/`
   included, fixing the same class of bug for `docker compose --profile gpu
   build paddleocr-vl-service` itself.
2. **Missing native shared libraries (fixed)**: `python:3.12-slim` (this
   Dockerfile's base image) is missing `libgomp.so.1` — `import paddle`
   failed immediately inside a freshly-built container with
   `ImportError: libgomp.so.1: cannot open shared object file`, a
   completely blocking issue (the service would never have worked via
   Docker at all, `/health` alone doesn't exercise this import path since
   it's lazy-loaded). This had never been caught before because every
   prior real inference verification ran in a WSL-native venv, whose host
   Ubuntu already had `libgomp1` installed system-wide. Fixed by adding
   `apt-get install libgomp1 libgl1 libglib2.0-0` (the latter two are the
   other well-known `opencv-contrib-python`-on-slim-Debian gotchas, added
   proactively). Confirmed fixed live: `paddle.is_compiled_with_cuda()`
   returns `True` and `cv2` imports cleanly inside the rebuilt container.
3. **GPU passthrough confirmed working end-to-end via Docker**:
   `docker compose --profile gpu up -d paddleocr-vl-service` — container
   reaches Docker's own `HEALTHCHECK` "healthy" state, `GET /health` OK,
   `paddle.device.cuda.device_count()` returns `1` inside the container.
   The async/threadpool fix (F3's predecessor) was also reconfirmed live
   in the real containerized deployment specifically (not just bare-metal):
   `/health` stayed responsive throughout a long-running real inference
   request. `backend-worker-gpu` was spot-checked the same way
   (`docker run --gpus all ... python -c "import torch; torch.cuda.is_available()"`
   → `True`, `import docling` → OK).

**⚠️ NOT fully resolved — flagged as an open, unexplained finding, not
swept under the rug**: two real, complete `POST /describe_figure` calls
through the **freshly-started container** (one a complex diagram, one a
trivial small icon — different content, same anomaly) each ran for
**20+ minutes without completing** (killed manually both times; no error
in logs, CPU pinned near 100% the whole time — i.e. NOT a deadlock, still
genuinely computing) before I stopped waiting. Immediately after, the
**exact same small icon** through the **bare-metal daemon** (also a cold
start — model not yet loaded) completed in **78.7 seconds** with a real,
non-null description. This isolates the anomaly to the containerized
deployment specifically, but the root cause is **not confirmed** within
this session — plausible candidates (untested): CPU thread/core
availability differences between the container's cgroup and the bare WSL
process (paddle/BLAS libraries often auto-configure their internal thread
pool differently inside a container namespace), or some other
container-specific resource constraint. **This is a real, unresolved
production-readiness concern for the official Docker deployment path** —
it now demonstrably *works* (builds, starts, passes healthchecks, GPU
passthrough confirmed), but real-inference latency inside a container has
NOT been shown to be viable at the throughput this project needs. Flagged
explicitly for follow-up investigation before relying on
`docker compose --profile gpu up` for a real multi-document ingestion run
— the bare-metal WSL process remains the verified-fast path for now.

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

## ⚠️ KNOWN RISK — VRAM headroom is extremely tight, explicitly flagged for Jalal

**Peak VRAM observed across multiple live runs: 5.92-5.98GB of the 6.14GB
(6144 MiB) RTX 3060 card — consistently, not a one-off.** This is a real,
known operational risk for anyone running this service on a 6GB GPU, not a
theoretical concern: only ~150-220MB of headroom remains at peak, meaning
essentially **any** additional GPU memory pressure (a slightly larger/more
complex image, a concurrent GPU-using process on the host — even outside
Docker/WSL, e.g. a browser with hardware acceleration — or `backend-worker-gpu`'s
Docling not having fully released its own VRAM yet, see §4.0 mitigation
layers 1-2) can plausibly push this over budget and trigger an out-of-memory
crash mid-ingestion. This is **not hypothetical** — it is the realistic
day-to-day operating margin of this service on the target dev hardware.

**Update — full 286-page corpus run completed without OOM**: a real
end-to-end run against the full Zeggo VRV IV REYQ document (386 real
`describe_figure`/`reparse_table` calls across the whole corpus, not just
one sample image) completed successfully with no OOM crash — see
`backend/docs/i1.10-e2e-findings.md`. This is reassuring operational
evidence (the risk did not materialize across a real, varied, full-corpus
workload), but does **not** retract the risk assessment above — 386
successful calls in a row is not proof against the ~150-220MB margin ever
being exceeded, especially by content not represented in this one manual
(other 6 source documents, larger/denser diagrams, etc.). Treat this as
"the risk is real and should be monitored," not "the risk didn't
materialize so it can be ignored."

### VRAM reduction investigated (I1.10), honest result: no further reduction found

Per explicit instruction to investigate before accepting this risk as final:

| Lever | Status | Result |
|---|---|---|
| `PADDLE_OCR_VL_SERVICE_BATCH_SIZE=1` | Already minimal | N/A — already at the floor, `>1` not evaluated (design doc explicitly prohibits raising without benchmark data proving it fits, which this finding argues strongly against attempting) |
| `precision=fp16` | Already minimal for this model | N/A — already the lowest precision PaddleOCR-VL's `precision` parameter supports without dedicated quantization tooling (int8/4-bit) that isn't part of `paddleocr`'s standard inference path and wasn't evaluated (would require custom quantization work, out of scope for this session) |
| `max_new_tokens` cap (this session's addition, default 1024) | **Tested live** | **No measurable VRAM reduction** — peak was ~5.96GB with the cap vs. ~5.98GB without (within measurement noise). Kept anyway because it bounds worst-case per-request *latency* (a real, separate benefit — see "Why some requests hang" below), just not VRAM. |
| `use_chart_recognition=False` | Not fully evaluated | Attempted but not conclusively measured within session time — construction alone doesn't eagerly allocate the full VRAM footprint (lazy weight transfer to GPU appears to happen at/near first real inference call, not at pipeline construction), so a clean before/after comparison needs a full inference run per variant, which given the ~35-90s per real inference call and the number of variants worth testing, was not completed. **Flagged as unfinished investigation, not a dead end** — a legitimate next step for whoever picks this up.
| Smaller `pipeline_version` (`"v1"`/`"v1.5"` vs default `"v1.6"`) | Not evaluated | Older pipeline versions may use smaller/different models — not tested this session (would trade extraction quality for memory, and quality impact is unknown without evaluation data) |

**Conclusion, stated plainly**: with the levers that were fully verified
this session (batch size, precision, generation length cap), **peak VRAM
usage could not be brought down from ~5.96-5.98GB**. The remaining
untested levers (`use_chart_recognition=False`, older `pipeline_version`,
real quantization) may or may not help and were not ruled in or out —
genuinely open follow-up work, not something silently abandoned.
**Until further investigated, treat this service's real-world VRAM
requirement on a 6GB GPU as ~6GB, i.e. effectively the entire card**, and
plan operational mitigations (idle-unload already implemented; closing
other GPU-using host applications during ingestion; monitoring `nvidia-smi`
during any large ingestion run; being prepared for intermittent OOM,
especially on complex diagrams/tables that may need more memory than the
single test image this was benchmarked against) rather than assuming the
current mitigations (`batch=1`, `fp16`, idle-unload) are sufficient to rule
out OOM entirely.

### Not yet verified

- ~~Multiple concurrent/sequential inference calls in a long-running service
  process under real production-like load~~ — **now verified**: 386
  sequential real inference calls across the full 286-page Zeggo VRV IV
  REYQ corpus completed successfully in one long-running service process
  (see "Update" note above). *Concurrent* (simultaneous, not sequential)
  request handling is still only verified at the level of "the event loop
  stays responsive" (§ event loop fix above), not "two inference requests
  submitted at the same instant both complete correctly" — the internal
  `RLock` serializes them by design (§ `app/pipeline.py`
  `PaddleOCRVLPipeline` docstring), so this is expected/intentional
  behavior, not an untested gap, but worth stating explicitly.
- ~~The full Docker image build (`docker build` of this service's
  `Dockerfile`) — dependency resolution was verified via `uv sync` in a
  WSL-native venv..., not via an actual `docker build` + `docker run`.~~ —
  **now verified** (F4, QA phase-1-qa-report.md §2.1, 2026-08-10):
  `docker compose --profile gpu build`/`up` actually run, container reaches
  Docker's own `HEALTHCHECK` "healthy" state, GPU passthrough confirmed
  inside the container (`paddle.is_compiled_with_cuda()==True`). Found and
  fixed 2 real, previously-undiscovered bugs in the process (missing
  `.dockerignore`, missing `libgomp1`/`libgl1`/`libglib2.0-0` native
  dependencies) — see the F4 section above. **Still open**: real-inference
  latency inside the container was NOT verified viable (20+ minute stalls
  observed, root cause unconfirmed) — see F4 section above for full detail.
  This is a materially different, more cautious conclusion than the
  pre-F4 state of this document, which had assumed `uv sync` parity was a
  sufficient stand-in for a real container run.

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
