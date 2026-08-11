# Ingestion Page-Range Batching — OOM Fix (2026-08-11)

Backend Engineer implementation note for `features/ingestion-page-range-batching`.
Not a `Documentation/system-design/` doc (out of Backend Engineer's write scope)
— flagged to System Analyst for optional promotion into
`02-ingestion-pipeline.md` if this decision should be made official design
record. See `### DIRECT MESSAGE -> System Analyst` in the STATUS REPORT for
this task.

**Three rounds in this doc**: round 1 (Docling-only batching) was verified
against the real 567-page document and **still OOM'd** — see §2. Round 2
(parse -> cascade -> store -> release per batch) was ALSO verified against
the real document, this time via memory sampling — it genuinely fixed the
per-batch loop (flat ~5.3GB across a 36-minute run) but a new, more precise
data point showed the OOM had simply moved to the very next stage
(`chunking`/`embedding`, both still whole-document) — see §8. Round 3 (this
doc's current state) extends the same principle to those stages, with an
explicit, reasoned exception for the chunk-*grouping* algorithm itself
(§8.2) — §9 records what IS/ISN'T verified for round 3.

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
minutes" timing both pointed at **Stage 2 (Docling)** as the OOM source —
this was round 1's working hypothesis.

## 2. Round 1 shipped, verified against real hardware, still OOM'd

Round 1 (`ParseCarryState`, `DoclingParser.parse(page_range=...)`,
`_run_docling_in_batches` merging every batch's `ElementDraft`s into ONE
`DoclingParseResult`, `store_pages_and_elements` still called ONCE for the
whole document) was committed and then **actually tested against the real
567-page document** — unlike everything else in this doc, this is a real
measurement, not a hypothesis:

```
Out of memory: Killed process 7354 (python)
anon-rss: 12,722,548 kB   (~12.7 GB)
```

**Virtually unchanged from the ~12.5GB round-1-absent baseline.** However,
round 1 was NOT wasted work — the run survived ~10 minutes and reached real
OCR processing (78 log lines vs. 6 lines in the pre-round-1 attempts that
died in 1-2 minutes), confirming Docling's own per-`convert()` memory WAS
genuinely bounded by page-range batching. The remaining OOM has a different,
independent root cause.

**Root cause, round 2**: `_run_docling_in_batches` (round 1) still
accumulated every batch's `ElementDraft`s into ONE merged `DoclingParseResult`
and called `store_pages_and_elements` exactly ONCE at the end for the whole
document — so peak memory was still **O(total_pages)**, just with the
specific "Docling's internal per-conversion state" contributor now bounded.
**Honesty caveat up front**: the diagnosis below is from reading source
(`app/db/engine.py`, SQLAlchemy's documented behavior) plus general
knowledge of a well-known SQLAlchemy footgun class ("long loop of
`add()`+`commit()` without periodic `expunge()`/session reset grows
unbounded") — NOT from a memory profiler (`tracemalloc`/`memory_profiler`)
run against this specific code, because WSL was not available to run one.
Treat it as the best available diagnosis, not a confirmed root cause.

The leading candidate: this project's SQLAlchemy `Session` is configured
with **`expire_on_commit=False`** (`app/db/engine.py`) — meaning a
`db.commit()` alone does **not** release previously-loaded objects'
attribute state, and separately, ORM instances (via their `InstanceState`)
are known to commonly form reference cycles that plain CPython refcounting
cannot reclaim without an actual garbage-collector cycle-detection pass —
`store_pages_and_elements` never called `gc.collect()` at all internally
(round 1's per-batch `gc.collect()` lived at the ORCHESTRATOR level, and in
round 1's design storage was still a single whole-document call, so that
per-batch collection point never applied to it either). Combined, thousands
of `Page`/`Element` ORM objects per page/element — each fully loaded with
`text`/`bbox`/`kg_candidate_entities`/`kg_candidate_relations` jsonb
payloads — plausibly stayed reachable (via the session's identity map
and/or uncollected cycles) for the rest of the ingestion run even after
being durably committed to Postgres and never needed again as Python
objects.

This is at least consistent with round 1's own numbers: Docling's
contribution was bounded (10 minutes of real progress vs. 1-2 minutes
before), but total peak memory barely moved — something else large and
document-scoped was still there. The fix in §3.2 (explicit `db.expunge()` +
per-batch `gc.collect()`) directly targets this candidate regardless of
which exact mechanism (weak-ref identity map vs. uncollected cycles) turns
out to be the precise cause — both are addressed by the same fix.

## 3. Round 2: parse -> cascade -> store -> release, PER BATCH

Round 2 restructures `app/ingestion/orchestrator.py` so Stages 2-4 +
canonical store + KG candidates (`_run_batched_pipeline`, replacing round
1's `_run_docling_in_batches`) run **once per page-range batch**, not once
per document:

```
for each batch:
    parse (Docling, page_range=(start,end), carry_state=<from previous batch>)
    -> unload Docling (every batch now, not just once — see below)
    build cascade plan (this batch's elements only)
    -> run PaddleOCR-VL cascade (this batch's tasks only)
    -> extract KG candidates (this batch's elements only)
    -> store_pages_and_elements(page_range=(start,end), local_id_to_db_id=<carried>)
       -> db.expunge() every Page/Element object once no longer needed
    -> del batch-local Python objects, gc.collect()
```

`chunking`/`embedding` remain whole-document, unchanged — see §3.2 for why
they're not implicated.

### 3.1 Two pieces of carried state, both required for correctness

- **`ParseCarryState`** (`docling_parser.py`, round 1, unchanged) — keeps
  `local_id`/`section_path`/`last_text_local_id_doc` correct across batches
  at the **parsing** layer.
- **`local_id_to_db_id`** (`canonical_store.py`, **new in round 2**) — an
  optional dict, mutated in place by `store_pages_and_elements`, carried by
  the caller across batch calls. Needed because a child element's
  `parent_local_id` can point at an element that was stored in an EARLIER
  batch (the exact SA1.2 document-wide icon-parent fallback scenario) —
  without carrying this dict, `store_pages_and_elements` would look up that
  `parent_local_id` in a dict that only knows about the CURRENT batch's
  elements, resolve to nothing, and silently write `parent_id=NULL` instead
  of the real parent. This is the storage-layer counterpart to what
  `ParseCarryState` already protects at the parsing layer — **both are
  required together**; round 1 only had the first.

### 3.2 The other, complementary fix: explicit `db.expunge()`

`store_pages_and_elements` now `db.expunge()`s every `Page`/`Element` ORM
object as soon as it's no longer needed (its `.id` captured into a plain
int; any KG-candidate attribute mutation flushed first — `db.expunge()`
silently drops unflushed pending changes, so the flush must happen first,
not after). This directly targets the `expire_on_commit=False` accumulation
identified in §2, independent of whether storage is called once per
document or once per batch — but calling it once per batch (§3) additionally
bounds the OTHER previously-O(document) contributors (the batch's own
`ElementDraft` list, cascade results holding VLM description strings, etc.)
to O(batch) too, via the `del ...; gc.collect()` at the end of each batch
iteration in `_run_batched_pipeline`.

**Why `chunking`/`embedding` were NOT changed**: `_load_chunkable_elements`
does one `SELECT` over the whole document's `elements`, but immediately
maps every row into a lightweight `ChunkableElement` dataclass (text
strings, not the full ORM `Element` object with jsonb payloads) and does
not hold onto the ORM rows themselves beyond that mapping. The `expire_on_
commit=False` accumulation problem in §2 is specifically about `Element`/
`Page` ORM instances the session keeps TRACKING as persistent (from
`db.add()`, never expunged) — `_load_chunkable_elements`'s query results
are read once and converted, not accumulated as long-lived tracked
instances the way `store_pages_and_elements` used to. Left unchanged to
keep this round's blast radius as small as it could be while still fixing
the evidenced problem.

### 3.3 Docling now unloaded every batch, not just once

Round 1 called `parser.unload()` once, after ALL batches. Round 2 calls it
after EVERY batch's parse — this is required for `require_docling_
unloaded_before_paddle_stage` (`02-ingestion-pipeline.md` §4, WAJIB for the
`local` PaddleOCR-VL backend, still the `Settings` default value even
though `remote_api` is the actually-deployed configuration per §4.0) to
hold at every batch's cascade step, not just once — since Docling's
converter is now re-`_ensure_loaded()`ed at the START of every batch's
parse (not kept warm across batches), this is also itself part of the
O(batch) memory guarantee for Docling's own state, not just a guard-
satisfaction technicality. Trade-off: Docling's model gets reloaded once
per batch instead of once per document (a few tens of seconds' extra
overhead total for ~6 batches on the 567-page document) — accepted, since
this is an async background job with no TTFT-style latency requirement
(`05-streaming-and-api-contract.md`) and correctness/memory took priority
per this task's explicit direction.

**Paddle client lifecycle is UNCHANGED** (built lazily once, `unload()`ed
once, after the whole batch loop) — it's a thin HTTP client wrapper around
a remote/separate service in the supported `remote_api` configuration
(`02-ingestion-pipeline.md` §4.0), not a locally-memory-heavy resource in
`backend-worker-gpu`, so there was no O(document) memory concern to fix
there.

### 3.4 `ingestion_jobs` stage-tracking granularity changed

`docling`/`paddle_cascade`/`kg_candidate` now get **one `ingestion_jobs`
row per batch** (not one per document) — for an N-batch document there are
N rows for each of those three stages, interleaved in batch order (not
grouped by stage), e.g. for 2 batches: `native_probe, docling(1),
paddle_cascade(1), kg_candidate(1), docling(2), paddle_cascade(2),
kg_candidate(2), chunking, embedding`. `native_probe`/`chunking`/
`embedding` stay at exactly one row each. No schema change needed —
`ingestion_jobs.stage` was never uniquely constrained per document
(`app/db/models/ingestion_jobs.py`), so this is compatible with the
existing table. This is also a net observability improvement: operators
can now see exactly which batch/stage a long-running or crashed ingestion
reached, not just "docling: done, paddle_cascade: running".

## 4. Decision (unchanged from round 1): in-process batching, not subprocess-per-batch

Still the right call after round 2's investigation — the remaining OOM
turned out to be the ORM accumulation (§2), a Python-object-lifecycle issue
fixable with `db.expunge()`, not evidence that Docling's own in-process
memory wasn't actually released by round 1's `page_range` batching (the
"~10 minutes of real progress, 78 log lines" evidence in §2 confirms it
WAS). Subprocess isolation remains the documented escalation path if a
*future* real run shows Docling's own per-batch memory still isn't
released even after round 2 — not needed based on current evidence.

## 5. Known, accepted limitation (unchanged from round 1): figure/caption split across a batch boundary

Docling assigns `self_ref` identifiers (used internally to resolve
`figure.captions -> caption text item`) **per `convert()` call**. If a
figure lands on the last page of batch N and its caption lands on the first
page of batch N+1, the link is not recoverable after the fact from either
batch's output alone. Not mitigated — captions are virtually always
same-page as their figure in these service manuals, so this is a narrow
edge case; a real fix (page-overlap + dedup across batches) was judged not
worth the added complexity. See STATUS REPORT DIRECT MESSAGE to System
Analyst.

## 6. What is verified vs. NOT verified (round 2)

**Verified (this task, round 2):**

- 424 unit tests pass (up from 419 after round 1), 100% line/branch
  coverage, `ruff check .` clean, `mypy app/` clean.
- End-to-end equivalence proof through the FULL pipeline (not just the
  parsing layer, which round 1 already proved) — `tests/unit/
  test_ingestion_orchestrator.py
  test_run_ingestion_pipeline_batched_matches_unbatched_icon_parent_across_batches`
  runs the exact SA1.2 icon-fallback scenario through `run_ingestion_
  pipeline` twice (batch size 1 -> 2 batches with the boundary literally
  between the paragraph and the icon page; batch size 100 -> 1 batch),
  queries the REAL persisted `elements` rows from the DB afterward, and
  asserts both runs produce structurally identical rows, with both icons'
  `parent_id` correctly resolving to the page-1 paragraph in BOTH runs (not
  `NULL` in the batched run) — this specifically proves `local_id_to_db_id`
  carries correctly through `store_pages_and_elements` across a real batch
  boundary, not just `ParseCarryState` through parsing (round 1's proof).
- `test_run_ingestion_pipeline_multi_batch_job_counts_and_order` proves the
  `ingestion_jobs` interleaving described in §3.4, and that Docling's
  `unload()` is now called once per batch while the PaddleOCR-VL client's
  stays at once overall.
- `test_store_pages_and_elements_expunges_created_objects_from_session`
  (`tests/unit/test_ingestion_canonical_store.py`) is a direct regression
  guard for the §2/§3.2 fix itself — asserts the SQLAlchemy session's
  identity map contains ONLY the `document` object after a
  `store_pages_and_elements` call, not the `Page`/`Element` objects it just
  created.
- `test_store_pages_and_elements_local_id_to_db_id_carries_parent_across_calls`
  plus its explicit NEGATIVE counterpart (`..._without_shared_dict_orphans_
  cross_call_parent`, asserting `parent_id IS NULL` when the dict is NOT
  carried) together prove the carry is load-bearing, not incidental.

**NOT verified (environment constraint, stated honestly, same limitation as
round 1 — WSL got LESS stable during round 2, not more):**

- **No real Docling/Postgres run against the actual 567-page document for
  round 2's fix specifically.** Round 1 WAS verified against real hardware
  (that's how we know it still OOM'd, §2) — but round 2 could not be, given
  WSL was reported failing within seconds even for small documents by the
  time this round started (`wsl -l -v` showing repeated `Stopped` state).
  All round-2 verification is via the mocked/fake-object unit test suite
  (§6 above), which proves mapping/storage correctness and (via the
  identity-map assertion) proves the SPECIFIC mechanism believed
  responsible (`expire_on_commit=False` + un-expunged objects) is now
  fixed for what it can observe in-process — but cannot reproduce or
  re-measure the actual WSL kernel-level OOM.
- **No before/after peak-memory (`anon-rss`) measurement for round 2.**
  The claim "this fixes the remaining OOM" rests on: (a) round 1's own real
  measurement (§2) isolating the remaining problem to something OTHER than
  Docling's per-conversion memory (already fixed and confirmed working),
  (b) `expire_on_commit=False` being an actual, confirmed (read directly
  from `app/db/engine.py`) session configuration whose documented behavior
  matches the failure mode, and (c) the identity-map regression test in §6
  directly proving objects no longer accumulate in-session. It does NOT
  rest on a new real measurement, because WSL was not available to take one
  during this round.
- Recommended next step, unchanged in spirit from round 1: once WSL is
  stable again, re-attempt the 567-page document and record peak `anon-rss`
  before considering this fully closed. If it STILL OOMs after round 2,
  the next things to check, in order: (1) whether `gc.collect()` calls are
  actually reclaiming what's expected on the real Linux/WSL allocator (vs.
  Python holding freed memory in its own arena without returning it to the
  OS — a different, harder problem `gc.collect()` alone doesn't solve), (2)
  whether `INGESTION_PAGE_BATCH_SIZE=100` needs to be lowered further, (3)
  subprocess-per-batch escalation (§4).

## 7. Resumability — partially delivered (was fully scoped out in round 1)

Round 1 scoped this out entirely. Round 2's restructuring **incidentally
delivers a real, if partial, improvement**, since `store_pages_and_elements`
is now called (and internally `db.commit()`s) once per batch instead of
once per document:

- **Delivered**: if an ingestion run crashes partway through a large
  document, all FULLY-COMPLETED batches up to that point are durably
  committed in Postgres (not lost/rolled back). A subsequent retry that
  re-runs the whole pipeline from scratch will re-parse everything with
  Docling again (wasted compute, not correctness-affecting), but
  `store_pages_and_elements`'s pre-existing `page_hash` idempotency check
  means already-unchanged, already-stored pages are SKIPPED at the storage
  layer (no re-render/re-upload/re-insert) even on a from-scratch retry —
  so a retry after a crash is CHEAPER than the very first attempt, even
  though it isn't a true "resume from where it left off".
- **NOT delivered**: true resume-without-re-parsing (skip Docling entirely
  for already-completed batches on retry). This would require persisting
  `ParseCarryState` itself somewhere durable at each batch boundary (it
  currently only lives in the orchestrator's local Python variables,
  lost on process crash/restart) — a real schema change, judged out of
  scope for this task given: (a) it was explicitly framed as a "bonus, you
  decide" item, not a requirement, (b) WSL/Postgres instability made it
  impossible to validate a schema change with any confidence this round,
  and (c) the partial improvement above already meaningfully reduces
  wasted work on retry without that risk. Flagged as a follow-up
  candidate, not silently dropped.

## 8. Round 3: chunking/embedding — real evidence round 2 wasn't the end of the story

The coordinator ran round 2 against the real 567-page document **with
memory sampling** (a real profiler-equivalent, not a hypothesis):

```
t+2100s  rss=5.351 MB   peak=8.183 MB
t+2130s  rss=5.351 MB
t+2160s  rss=5.351 MB      <- flat for ~36 minutes
t+2190s  rss=8.989 MB      <- sudden spike
t+2220s  rss=10.408 MB
OOM:     anon-rss 10,885,484 kB (~10.9 GB)
```

Compared to before: pre-batching died in 1-2 minutes at ~12.5GB; round 1
survived ~10 minutes at ~12.7GB; round 2 survived **~37 minutes**, **flat
at 5.3GB** for most of the run, peaking at 10.9GB. **This is real,
positive, measured confirmation that round 2's per-batch loop works** —
memory genuinely stayed flat across many batches instead of accumulating,
directly corroborating the `expire_on_commit=False` + `db.expunge()`
diagnosis in §2/§3.2.

The spike (5.3GB -> 9.0GB -> 10.4GB -> OOM) happened in the ~60 seconds
**immediately after** the batch loop finished — i.e. during `chunking`/
`embedding`, both still whole-document operations at the time (explicitly
called out as "not implicated by the evidence" in round 2's own STATUS
REPORT — that was accurate given the evidence available *then*; this new,
more precise timing data changes that).

### 8.1 Structural verification (done before patching, per instruction)

Given the explicit instruction to verify before patching, and given a real
profiler run was not repeatable in this environment (WSL down again — see
§9), verification here is via READING the code, not re-measuring:

- **`app/ingestion/chunker.py` `_load_chunkable_elements`** (before this
  round): `select(Element, Page.page_number)....all()` — one query loading
  every `Element` row for the document (for 567 pages, on the order of
  several thousand rows, each with `bbox`/`kg_candidate_entities`/
  `kg_candidate_relations`/`visual_description` jsonb payloads) into one
  Python list, all at once, with no `db.expunge()`.
- **`app/ingestion/chunker.py` `store_chunks`** (before this round):
  accumulated every newly-created `Chunk` ORM object into a `rows: list
  [Chunk]`, RETURNED that list all the way up through `run_ingestion_
  pipeline`'s `chunk_rows` local variable — kept alive (per this project's
  `expire_on_commit=False`, `app/db/engine.py`) all the way through the
  `embedding` stage too, purely to support `len(chunk_rows)` at the very
  end. Structurally identical to the exact bug already found and fixed in
  `canonical_store.py` in round 2.
- **`app/ingestion/embedder.py` `embed_and_upsert_chunks`** (before this
  round): `select(Chunk).where(status=='pending')....all()` — every
  pending chunk for the document loaded into one list, one `texts` list
  built from all of them, one `dense_model.embed(texts)`/`sparse_model.
  embed(texts)` call over the WHOLE list, one `points` list holding a
  Qdrant point per chunk, one `qdrant_client.upsert()` call — fully
  whole-document, no batching, no expunge anywhere.

**Honest caveat, exactly as asked**: this confirms BOTH sub-stages are
unambiguously whole-document as coded — a real, verifiable structural fact
— but does NOT, and cannot without a profiler, tell us the precise GB
split between them, or rule out a THIRD contributor (e.g. first-time
`fastembed`/ONNX model loading inside `_ensure_loaded()`, which is a
one-time, potentially large fixed-cost allocation event that could
plausibly explain a sudden jump rather than a gradual ramp — the sampling
interval, 30s, is too coarse to distinguish "sudden model-load spike" from
"fast accumulation across ~3000+ chunks" within the same window). Given
both `chunker.py` and `embedder.py` are unambiguously buggy in the SAME
way already fixed elsewhere in this codebase (whole-document ORM
accumulation under `expire_on_commit=False`), and the task's explicit
directive was to extend the SAME principle to every remaining whole-
document stage rather than isolate one exact culprit, both were fixed —
see §8.3.

### 8.2 Deliberately NOT touched: the chunk-*grouping* algorithm itself

`build_chunks`/`build_entity_chunks` (`app/ingestion/chunker.py`) are
**unchanged** this round — no page-range batching was applied to the
chunk-grouping algorithm itself. This was an explicit request from the
coordinator ("kalau menurutmu chunking tidak bisa dipecah tanpa merusak
semantik, katakan begitu dan usulkan alternatif") — here is that answer:

**Why it's genuinely harder than `canonical_store.py`'s round-2 fix**:
`store_pages_and_elements`'s per-batch design worked because each element
is independently INSERTed once, and the only cross-batch dependency
(`parent_id` FK) is resolved via a small carried `int -> int` dict
(`local_id_to_db_id`). Chunking has a STRUCTURALLY different problem:

1. **Running chunk continuation is stateful across page boundaries.**
   `build_chunks` keeps a chunk "open" (`current_text_chunk_index`/
   `current_procedure_chunk_index`) and APPENDS to it across MULTIPLE
   elements/pages as long as `section_path` matches and the char limit
   isn't hit — a chunk that would naturally span, say, pages 99-101 must
   not be artificially split into two chunks just because a page-range
   batch boundary happens to fall at page 100. Splitting it would change
   the actual PERSISTED content (different chunk boundaries, different
   `content_text` groupings) — a real semantic regression, not just an
   internal implementation detail, and one the required equivalence tests
   would (correctly) catch.
2. **Icon/figure_caption "always join parent chunk" can reach back
   arbitrarily far** (module docstring, "most critical requirement in the
   whole project") — a chunk that needs to be joined might already be
   CLOSED and durably persisted in an EARLIER batch. Handling this
   correctly would require being able to re-open and `UPDATE` an
   already-persisted, already-`db.expunge()`d `Chunk` row from a prior
   batch (fetch by id, append, re-save) — a real, buildable mechanism in
   principle, but one that adds a genuinely new code path (UPDATE-in-place
   on a previously-closed chunk) that doesn't exist anywhere else in this
   codebase yet, and that the existing equivalence-test suite doesn't
   exercise.

**Alternatives considered** (per the request to propose them, not just
decline):

- **Streaming query (`yield_per`) but still building ALL chunks in one
  logical, uninterrupted pass** — i.e. don't split the ALGORITHM into
  independent batches at all, just avoid materializing the full ORM
  result set during the LOAD. This is what was actually implemented (see
  §8.3) — it's a real, if partial, improvement (avoids double-holding raw
  `Element` ORM state alongside the mapped `ChunkableElement`s during
  the query), but the mapped `elements: list[ChunkableElement]` list, and
  the `drafts: list[ChunkDraft]` list `build_chunks`/`build_entity_chunks`
  produce, are still held in full, in memory, for the whole document.
  Judged an acceptable, bounded cost: `ChunkableElement`/`ChunkDraft` are
  plain lightweight dataclasses (no SQLAlchemy `InstanceState`/session
  machinery), so even a few thousand of them is a materially smaller
  footprint than the ORM-object accumulation bugs that WERE fixed (§8.3) —
  though this is a reasoned estimate, not a profiled number (see §8.1's
  caveat).
- **Process per-section instead of per-page-range batch**: since chunk
  continuation is scoped by `section_path`, batching by SECTION rather
  than by PAGE COUNT would mean a batch boundary never falls in the middle
  of a would-be-continued chunk. This is a genuinely more chunking-aware
  batching unit than a raw page count — but doesn't solve problem #2
  above (an icon can still reference a parent in an EARLIER section/batch)
  and would need its own carried state (last-open-chunk-per-section,
  `element_id -> chunk_id` map) built and verified with the same rigor as
  `ParseCarryState`/`local_id_to_db_id` were in rounds 1-2. Deferred as a
  candidate future round, not attempted here — building AND verifying it
  correctly needs either real infrastructure (currently unavailable, WSL
  down) or a much larger synthetic-equivalence-test investment than was
  safe to take on together with the other round-3 fixes in the same pass.

### 8.3 What WAS fixed this round

- **`app/ingestion/chunker.py` `_load_chunkable_elements`**: streams via
  `execution_options(yield_per=500)` instead of `.all()`, `db.expunge()`s
  each `Element` row immediately after mapping it to a `ChunkableElement`.
- **`app/ingestion/chunker.py` `store_chunks`**: now returns an `int`
  count instead of `list[Chunk]`; every `Chunk` row is `db.flush()`ed (to
  assign its id) and `db.expunge()`d immediately after insert, instead of
  being accumulated into a list held alive through the rest of the
  ingestion run. This was the more clear-cut, "definitely a bug not a
  trade-off" fix — unlike §8.2's chunk-grouping algorithm, nothing about
  `store_chunks`'s OWN correctness required holding onto the ORM objects
  after insert (nothing downstream ever used the returned `list[Chunk]`
  for anything but `len(...)` — confirmed by grepping every call site).
- **`app/ingestion/embedder.py` `embed_and_upsert_chunks`**: new
  `batch_size` parameter (default 500, `Settings.EMBEDDING_BATCH_SIZE`,
  `app/core/config.py`) — processes chunks in bounded-size passes
  (`WHERE embedding_status='pending' LIMIT batch_size`, repeated until
  none remain — chunks already embedded fall out of the filter naturally,
  no offset-pagination needed), `db.commit()`+`db.expunge()`+`gc.collect()`
  per inner batch. For `batch_size >= total pending chunks` this reduces
  to exactly the pre-round-3 single-call behavior (useful equivalence
  anchor, exercised directly in
  `tests/unit/test_ingestion_embedder.py`).
- **Known, deliberately out-of-scope exception found while auditing**:
  `app/ingestion/kg_candidate_reextractor.py` also does a whole-document
  `select(...).all()` load. NOT touched — its own module docstring
  states an explicit, pre-existing Wave 1 DoD constraint ("seluruh
  perubahan hidup di modul KG candidate... jangan ubah orchestrator.py
  atau canonical_store.py") specifically to avoid a real merge conflict
  with the not-yet-merged Fase 2 branch — and it is a separate, low-
  frequency maintenance tool (KG re-extraction for already-ingested
  documents, "~seconds per document" per its own docstring), not part of
  the `run_ingestion_pipeline` hot path this whole investigation has been
  about. Flagged here for visibility, not fixed.

## 9. What is verified vs. NOT verified (round 3)

**Verified:**

- 428 unit tests pass (up from 424 after round 2), 100% line/branch
  coverage, `ruff check .` clean, `mypy app/` clean.
- New regression tests directly proving the `expire_on_commit=False`
  accumulation is fixed for both changed modules — `tests/unit/
  test_ingestion_chunker.py test_store_chunks_expunges_created_rows_from_
  session` and `tests/unit/test_ingestion_embedder.py
  test_embed_and_upsert_chunks_expunges_processed_chunks_from_session`
  both assert the SQLAlchemy session's identity map is empty of the
  relevant ORM class after the call, same pattern as round 2's analogous
  `canonical_store.py` test.
- `test_embed_and_upsert_chunks_processes_multiple_batches` proves the
  batching loop actually processes chunks in bounded-size groups (not
  just accepting `batch_size` and ignoring it) and that every pending
  chunk still gets embedded/upserted regardless.
- All of round 2's equivalence tests (batched vs. unbatched ingestion
  producing identical persisted `elements` rows) still pass unmodified —
  confirms round 3's changes did not alter `canonical_store.py`'s
  behavior at all (it was not touched this round).

**NOT verified (environment constraint — WSL was down again for this
entire round, before any work started):**

- **No real run against the actual 567-page document for round 3's fix.**
  Unlike round 2 (which the coordinator verified with real memory
  sampling), round 3 has NO real-hardware confirmation at all — the
  `wsl -e bash -lc "..."` smoke check at the start of this round returned
  `Wsl/Service/E_UNEXPECTED` immediately, before any diagnostic command
  could even run. All verification is the unit test suite above.
- **§8.1's honest caveat stands**: even with real hardware, this round's
  fix was not preceded by a profiler run isolating the EXACT chunking-vs-
  embedding-vs-model-loading split — the fix targets everything that was
  structurally confirmed (by reading code) to be whole-document, which is
  a defensible action given the task's explicit "extend the same
  principle everywhere" directive, but is not the same rigor as round 2's
  real measurement.
- Recommended next step, unchanged in spirit: once WSL is stable, retry
  the 567-page document with the SAME memory-sampling approach that
  produced the round-2 data. If the process now completes successfully
  (or at least gets meaningfully further before any new OOM), that
  confirms round 3; if a NEW spike appears at a different point, that
  timing data is exactly what's needed to decide whether §8.2's
  chunk-grouping algorithm restructuring (deferred) is actually necessary,
  or whether the residual `ChunkableElement`/`ChunkDraft` list-holding
  discussed in §8.2's first alternative turns out to matter more than
  estimated.
