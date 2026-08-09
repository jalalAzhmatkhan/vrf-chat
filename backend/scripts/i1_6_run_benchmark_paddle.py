"""I1.6 — Stage 4 (PaddleOCR-VL, local backend) VRAM benchmark, run
separately/sequentially from `i1_6_run_benchmark.py` (Docling) per the
6GB VRAM constraint (Docling must be fully unloaded — in this case, not
even in the same process — before this runs).

Runs the cascade plan produced by the Docling benchmark
(`docs/i1.6-benchmark-results.json` must exist — see `i1_6_run_benchmark.py`)
against the real local PaddleOCR-VL pipeline for a bounded subset of tasks
(not necessarily all 44, see `MAX_TASKS_TO_RUN` — this is a VRAM/latency
probe, not a full cascade execution).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings  # noqa: E402
from app.ingestion.cascade_trigger import build_cascade_plan  # noqa: E402
from app.ingestion.docling_parser import DoclingParser  # noqa: E402
from app.ingestion.native_probe import probe_document  # noqa: E402
from app.ingestion.paddleocr_vl_cascade import (  # noqa: E402
    LocalPaddleOCRVLClient,
    render_bbox_crop,
)

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "docs" / "i1.6-sample-50pages.pdf"
RESULTS_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "i1.6-benchmark-paddle-results.json"
)
MAX_TASKS_TO_RUN = 5


def _read_used_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        text=True,
        timeout=5,
    )
    return int(out.strip().splitlines()[0])


class VRAMSampler:
    def __init__(self, interval_seconds: float = 0.2) -> None:
        self._interval = interval_seconds
        self.baseline_mib = _read_used_mib()
        self.peak_mib = self.baseline_mib
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, _read_used_mib())
            time.sleep(self._interval)

    def __enter__(self) -> VRAMSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def main() -> None:
    print(f"VRAM before anything: {_read_used_mib()} MiB")

    print("Re-deriving cascade plan (fresh process, Docling not resident here)...")
    probe = probe_document(SAMPLE_PDF)
    parser = DoclingParser(device="cpu")  # CPU here on purpose: this process
    # must never hold Docling on GPU at the same time as PaddleOCR-VL below.
    parse_result = parser.parse(SAMPLE_PDF)
    parser.unload()

    plan = build_cascade_plan(parse_result, probe)
    tasks = plan.tasks[:MAX_TASKS_TO_RUN]
    print(f"Running {len(tasks)} of {len(plan.tasks)} cascade tasks against local PaddleOCR-VL")

    elements_by_local_id = {e.local_id: e for e in parse_result.elements}

    settings = Settings(_env_file=None, PADDLE_OCR_VL_BACKEND="local", PADDLE_OCR_VL_DEVICE="cuda")
    client = LocalPaddleOCRVLClient(settings)

    print(f"VRAM before model load: {_read_used_mib()} MiB")
    per_task_results = []
    with VRAMSampler() as sampler:
        for task in tasks:
            bbox = None
            if task.element_local_id is not None:
                element = elements_by_local_id.get(task.element_local_id)
                bbox = element.bbox if element else None
            image = render_bbox_crop(SAMPLE_PDF, task.page_number, bbox)

            start = time.perf_counter()
            if task.task_type == "table_reparse":
                result = client.reparse_table(image)
            elif task.task_type == "visual_description":
                result = client.describe_figure(image)
            else:
                result = client.ocr_page(image)
            elapsed = time.perf_counter() - start

            per_task_results.append(
                {
                    "task_type": task.task_type,
                    "page_number": task.page_number,
                    "elapsed_seconds": elapsed,
                    "vram_used_mib_now": _read_used_mib(),
                    "result_repr": repr(result)[:300],
                }
            )
            print(per_task_results[-1])

    peak_mib = sampler.peak_mib
    baseline_mib = sampler.baseline_mib

    print(f"VRAM before unload: {_read_used_mib()} MiB")
    client.unload()
    time.sleep(1.0)
    vram_after_unload = _read_used_mib()
    print(f"VRAM after unload: {vram_after_unload} MiB")

    results = {
        "tasks_run": len(tasks),
        "total_tasks_available": len(plan.tasks),
        "per_task_results": per_task_results,
        "vram_baseline_mib": baseline_mib,
        "vram_peak_mib": peak_mib,
        "vram_peak_delta_mib": max(peak_mib - baseline_mib, 0),
        "vram_after_unload_mib": vram_after_unload,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
