# Ingestion Page-Range Batching — OOM Fix (2026-08-11)

Backend Engineer implementation note for `features/ingestion-page-range-batching`.
Not a `Documentation/system-design/` doc (out of Backend Engineer's write scope)
— flagged to System Analyst for optional promotion into
`02-ingestion-pipeline.md` if this decision should be made official design
record. See `### DIRECT MESSAGE -> System Analyst` in the STATUS REPORT for
this task.

## 1. Problem (hard evidence, not a hypothesis)

WSL kernel log, repeated 3x in a row, always within 1-2 minutes of starting:

```
python invoked oom-killer
Out of memory: Killed process 4093 (python)
total-vm:90616976kB, anon-rss:12562324kB
```

~12.5GB resident on a 13GB WSL VM. Trigger document: **Zeggo VRV III, 567
pages, 39.7MB**. Clear page-count correlation, not an environment fluke:

| Outcome | Documents | Pages |
|---|---|---:|
| Succeeded (unbatched, single `converter.convert()` call) | REYQ, PUHY-P, PUCY-P, PURY-P, RXQ-A | 286, 322, 334, 372, 403 |
| **Failed (OOM, 3x)** | Zeggo VRV III | **567** |

The orchestrator's stage sequence (`app/ingestion/orchestrator.py`:
`native_probe` -> `docling` -> `paddle_cascade` -> ...) and the "within 1-2
minutes" timing both point at **Stage 2 (Docling)** as the OOM source, not
Stage 1 (PyMuPDF-only, lightweight, would fail near-instantly if at all) or
any later stage.

## 2. Root cause hypothesis

Docling's `DocumentConverter.convert()` builds the *entire* `DoclingDocument`
object graph (per-page layout/table-model outputs, rendered page images used
internally by the pipeline, etc.) for **all** requested pages before
returning — i.e. memory scales with the number of pages given to a *single*
`convert()` call. A 567-page single call therefore holds ~1.4x the object
graph a 403-page call holds, which was already presumably close to the WSL
VM's ceiling.

This is consistent with Docling's own architecture (confirmed by reading
`docling/backend/docling_parse_backend.py`: `page_range` restricts which
pages are `load_page()`d at all — Docling does NOT eagerly load the whole
PDF regardless of `page_range`), which is exactly what page-range batching
exploits.

## 3. Decision: in-process page-range batching (not subprocess-per-batch)

Two options were evaluated, per the task brief:

| Option | Description | Decision |
|---|---|---|
| A. In-process batching + explicit release | Same `DoclingParser`/converter instance, called repeatedly with `page_range=(start, end)` windows; `del` the per-call `ConversionResult`/`DoclingDocument` + `gc.collect()` (+ `torch.cuda.empty_cache()` if CUDA) after every call | **Chosen** |
| B. Subprocess-per-batch | Spawn a fresh OS process per batch, hard OS-level memory isolation | Rejected (for this pass) |

**Why A, not B:**

1. The evidence (§2) implicates memory scaling with *pages given to one
   `convert()` call*, not *repeated calls leaking/accumulating across many
   calls to the same converter*. These are different failure modes — A
   directly addresses the one we have evidence for.
2. B adds real operational complexity: process spawn overhead per batch,
   `DoclingParseResult` (de)serialization across a process boundary (fine —
   it's plain dataclasses — but non-trivial to wire inside a Celery task),
   and either reloading Docling's models every batch (expensive: model
   init is the single most expensive one-time cost in Stage 2) or keeping a
   long-lived subprocess warm across batches (which reintroduces the same
   "does memory accumulate across repeated calls in one process" question B
   was meant to sidestep).
3. Consistent with this project's stated complexity philosophy
   (`02-ingestion-pipeline.md` §4: "tambah kompleksitas hanya kalau
   terbukti perlu" / evaluation-driven escalation) — A is the simpler fix
   that directly targets the evidenced root cause.

**Escalation path, if A later proves insufficient**: if a real batched run
still OOMs (i.e. per-call/per-batch memory doesn't actually get released
between batches, contradicting the hypothesis in §2), subprocess-per-batch
(Option B) is the documented next step — the `_run_docling_in_batches`
helper (`app/ingestion/orchestrator.py`) is the single call site that would
need to change (swap the in-process `parser.parse(page_range=...)` call for
a subprocess dispatch); nothing downstream would need to change, since it
already only sees the final merged `DoclingParseResult`.

## 4. What changed

- `app/ingestion/docling_parser.py`:
  - `DoclingParser.parse()` accepts optional `page_range: tuple[int, int]`
    (forwarded verbatim to Docling's native `convert(page_range=...)`) and
    `carry_state: ParseCarryState | None`.
  - New `ParseCarryState` dataclass (`next_local_id`, `section_path`,
    `last_text_local_id_doc`) threads the three pieces of whole-document
    state `map_document_to_elements` needs across batch boundaries so N
    batched calls produce output identical to one unbatched call — see the
    module docstring and `ParseCarryState`'s own docstring for the full
    "why" (this is the part that protects CLAUDE.md §4's inline-icon
    association requirement across a batch boundary).
  - `parse()` now unconditionally `del`s the per-call `result`/`doc` +
    `gc.collect()`s (+ `torch.cuda.empty_cache()` if CUDA) before returning
    — the actual "explicit memory release between batches" the task asked
    for. This is unconditional (not gated on batching), so the single
    unbatched call path benefits too, at negligible cost.
- `app/ingestion/orchestrator.py`:
  - New `_run_docling_in_batches()`: slices `[1, total_pages]` (from Stage
    1's probe) into `INGESTION_PAGE_BATCH_SIZE`-page windows, calls
    `parser.parse()` once per window threading `ParseCarryState`, and merges
    all batches' `elements`/`page_confidence` into one `DoclingParseResult`.
    `parser.unload()` is still called exactly once, after all batches
    finish — unchanged contract with Stage 4's VRAM-unload requirement.
  - `run_ingestion_pipeline`'s Stage 2 now calls this instead of a bare
    `parser.parse(pdf_path)`.
- `app/core/config.py`: new `INGESTION_PAGE_BATCH_SIZE: int = 100` (see
  inline comment for the full default-value rationale — short version:
  well below even the smallest known-safe whole-document run of 286 pages,
  for real margin, not a bare "just under 403" cutoff).
- **No changes** to `canonical_store.py`, `cascade_trigger.py`,
  `kg_candidate_extractor.py`, `chunker.py`, `embedder.py` — all of them
  already only ever see the final merged `DoclingParseResult`, exactly as
  before this fix (see reasoning in §5 of the orchestrator/docling_parser
  module docstrings for why the OOM and its fix are isolated to Stage 2).

## 5. Known, accepted limitation: figure/caption split across a batch boundary

Docling assigns `self_ref` identifiers (used internally to resolve
`figure.captions -> caption text item`) **per `convert()` call**. If a
figure lands on the last page of batch N and its caption lands on the first
page of batch N+1, batch N's converted document never saw the caption item
(out of its `page_range`) and batch N+1 never saw the figure item — the link
is not recoverable after the fact from either batch's output alone.

This is **not mitigated** in this pass — captions are virtually always
same-page as their figure in these service manuals (visual convention), so
this is a narrow edge case, and a real fix (page-overlap + dedup across
batches) was judged not worth the added complexity for it. Flagged
explicitly here rather than silently accepted — see STATUS REPORT for the
DIRECT MESSAGE to System Analyst about this trade-off.

## 6. What is verified vs. NOT verified

**Verified (this task):**

- 419 unit tests pass, including new ones proving batched output is
  semantically identical to unbatched output — see
  `tests/unit/test_ingestion_docling_parser.py`
  `test_batched_parse_matches_single_pass_icon_fallback_across_batch_boundary`
  (the exact SA1.2 icon-fallback scenario, deliberately split across a
  batch boundary) and `test_batched_parse_matches_single_pass_section_path_across_batch_boundary`.
  Both assert element-for-element equality between one unbatched
  `map_document_to_elements` call and two batched calls merged.
- `_run_docling_in_batches` unit-tested directly (correct page-window
  slicing, correct `carry_state` threading, correct merge, input
  validation) in `tests/unit/test_ingestion_orchestrator.py`.
- 100% branch/line coverage maintained (`app/ingestion/docling_parser.py`,
  `app/ingestion/orchestrator.py`), `ruff check .` clean, `mypy app/` clean.

**NOT verified (environment constraint, stated honestly per task brief):**

- **No real Docling run against the actual 567-page Zeggo VRV III PDF was
  performed.** WSL was unstable throughout this task (repeated
  restarts/crashes per the task brief), and `backend/.venv-wsl` (the only
  environment with Docling/torch installed for real ingestion) was
  explicitly off-limits for Backend Engineer to use in this task. All
  verification here is via the mocked/fake-object unit test suite, which
  proves the *mapping logic and merge correctness* but cannot measure real
  peak memory.
- **No before/after peak-memory measurement was taken.** The claim "this
  should fix the OOM" rests on: (a) the hard log evidence in §1 pointing at
  Docling's per-`convert()`-call memory scaling with page count, (b)
  confirming from Docling's own source that `page_range` genuinely
  restricts which pages are loaded (not just post-filtered after loading
  everything — see `docling/backend/docling_parse_backend.py`), and (c) the
  explicit memory-release additions in §4. It does NOT rest on a measured
  before/after number, because that measurement was not possible in this
  environment during this task.
- Recommended next step once WSL/`backend/.venv-wsl` is stable again:
  ingest the two remaining documents (Zeggo VRV III, 567 pages; Zeggo VRV
  X, 297 pages) for real and record peak `anon-rss` (e.g. via
  `/usr/bin/time -v` or a background `free -h` poll) before vs. after this
  change, ideally on the 567-page document specifically since that's the
  one with a reproducible 3x failure history.

## 7. Scoped out (not implemented this pass)

- **Full crash-resume** (skip already-completed batches when a Celery task
  retries after a mid-run crash). The task brief called this "ideally"
  (`idealnya`), not mandatory. Real resumability would require moving
  `canonical_store.store_pages_and_elements` to a per-batch/incremental
  commit model (carrying `local_id_to_db_id` across batches the same way
  `ParseCarryState` carries Docling's own state) — a materially larger
  change to already-tested, working code that this task's environment
  instability made too risky to build and validate with confidence in the
  time available. Flagged as a follow-up, not silently dropped — see
  STATUS REPORT.
