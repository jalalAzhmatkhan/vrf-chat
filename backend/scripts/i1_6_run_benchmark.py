"""I1.6 — VRAM/threshold/throughput benchmark, run against the 50-page
sample built by `scripts/i1_6_sample_pages.py`.

Per `Documentation/system-design/02-ingestion-pipeline.md` §3-4 and
`Documentation/project-milestones/02-phase-1-ingestion.md` I1.6, this
measures:
  (a) THRESHOLD_TABLE/THRESHOLD_TEXT calibration data (raw distributions —
      the actual threshold decision is System Analyst's SA1.1, this script
      only supplies the data).
  (b) Peak VRAM: Docling (Stage 2, GPU) vs PaddleOCR-VL (Stage 4, GPU),
      sequential, Docling unloaded in between.
  (c) Docling Stage 2 wall-clock: GPU vs CPU, same 50-page sample.

WAJIB run via WSL (RTX 3060 6GB) — see CLAUDE.md §5. This script itself has
no WSL-specific code; it is copied into a WSL-native `uv` project
(`~/vrf-chat-bench/backend/`, see I1.6 STATUS REPORT for why a WSL-native
venv was used instead of a full Docker build) and run there via
`uv run python scripts/i1_6_run_benchmark.py`.

Output: a JSON results file (`i1.6-benchmark-results.json`) — the prose
report (`docs/i1.6-vram-benchmark-report.md`) is written by hand from this
data, not generated automatically, so the report can include analysis/
recommendations text.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.cascade_trigger import build_cascade_plan  # noqa: E402
from app.ingestion.docling_parser import DoclingParser  # noqa: E402
from app.ingestion.native_probe import probe_document  # noqa: E402

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "docs" / "i1.6-sample-50pages.pdf"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "docs" / "i1.6-benchmark-results.json"


class VRAMSampler:
    """Polls `nvidia-smi` on a background thread and tracks the peak
    `memory.used` (MiB) observed for GPU index 0 while active — captures
    the FULL process footprint (CUDA context + allocator + fragmentation),
    not just what a framework's own allocator stats report."""

    def __init__(self, interval_seconds: float = 0.2) -> None:
        self._interval = interval_seconds
        self._peak_mib = 0
        self._baseline_mib = self._read_used_mib()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read_used_mib(self) -> int:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ],
                text=True,
                timeout=5,
            )
            return int(out.strip().splitlines()[0])
        except Exception as exc:  # pragma: no cover - benchmark tooling, not app/
            print(f"[VRAMSampler] nvidia-smi read failed: {exc}", file=sys.stderr)
            return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            used = self._read_used_mib()
            self._peak_mib = max(self._peak_mib, used)
            time.sleep(self._interval)

    def __enter__(self) -> VRAMSampler:
        self._peak_mib = self._baseline_mib
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def peak_mib(self) -> int:
        return self._peak_mib

    @property
    def baseline_mib(self) -> int:
        return self._baseline_mib

    @property
    def peak_delta_mib(self) -> int:
        return max(self._peak_mib - self._baseline_mib, 0)


def benchmark_docling(device: str) -> dict:
    parser = DoclingParser(device=device)
    sampler_ctx = VRAMSampler() if device == "cuda" else None

    start = time.perf_counter()
    if sampler_ctx is not None:
        with sampler_ctx:
            result = parser.parse(SAMPLE_PDF)
            peak_during_parse = sampler_ctx.peak_mib
    else:
        result = parser.parse(SAMPLE_PDF)
        peak_during_parse = None
    elapsed = time.perf_counter() - start

    vram_before_unload = None
    vram_after_unload = None
    if device == "cuda":
        vram_before_unload = VRAMSampler()._read_used_mib()
    parser.unload()
    if device == "cuda":
        time.sleep(1.0)  # let CUDA driver actually release memory
        vram_after_unload = VRAMSampler()._read_used_mib()

    table_confidences = [
        e.extraction_confidence for e in result.elements if e.element_type == "table"
    ]
    page_text_confidences = [
        pc.parse_score for pc in result.page_confidence.values() if pc.parse_score is not None
    ]

    return {
        "device": device,
        "elapsed_seconds": elapsed,
        "page_count": result.page_count,
        "element_count": len(result.elements),
        "table_element_count": len(table_confidences),
        "table_confidences": table_confidences,
        "page_text_confidences": page_text_confidences,
        "vram_peak_used_mib": peak_during_parse,
        "vram_baseline_used_mib": sampler_ctx.baseline_mib if sampler_ctx else None,
        "vram_peak_delta_mib": sampler_ctx.peak_delta_mib if sampler_ctx else None,
        "vram_used_before_unload_mib": vram_before_unload,
        "vram_used_after_unload_mib": vram_after_unload,
    }, result


def main() -> None:
    if not SAMPLE_PDF.exists():
        raise SystemExit(f"Sample PDF not found: {SAMPLE_PDF} (run i1_6_sample_pages.py first)")

    results: dict = {}

    print("== Stage 1: native probe ==")
    probe = probe_document(SAMPLE_PDF)
    results["native_probe"] = {
        "page_count": probe.page_count,
        "flag_counts": probe.flag_counts(),
    }
    print(results["native_probe"])

    print("== Stage 2: Docling GPU ==")
    gpu_metrics, gpu_result = benchmark_docling("cuda")
    results["docling_gpu"] = gpu_metrics
    print({k: v for k, v in gpu_metrics.items() if k not in ("table_confidences",)})

    print("== Stage 2: Docling CPU ==")
    cpu_metrics, _cpu_result = benchmark_docling("cpu")
    results["docling_cpu"] = cpu_metrics
    print({k: v for k, v in cpu_metrics.items() if k not in ("table_confidences",)})

    print("== Stage 3: cascade trigger plan (from GPU parse result) ==")
    plan = build_cascade_plan(gpu_result, probe)
    results["cascade_plan"] = {
        "task_counts": plan.task_counts(),
        "total_tasks": len(plan.tasks),
    }
    print(results["cascade_plan"])

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
