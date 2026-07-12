"""Typed reader for the pipeline's run_metrics.json — replaces regex scraping."""
from __future__ import annotations

import json
import math
from pathlib import Path


class RunMetricsError(RuntimeError):
    pass


class RunMetrics:
    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def from_dir(cls, results_dir) -> "RunMetrics":
        path = Path(results_dir) / "run_metrics.json"
        if not path.exists():
            raise RunMetricsError(f"run_metrics.json not found in {results_dir}")
        return cls(json.loads(path.read_text()))

    def get(self, name: str):
        return self._data.get(name)

    def _required(self, name: str) -> float:
        value = self._data.get(name)
        if value is None or not math.isfinite(float(value)):
            raise RunMetricsError(f"objective metric {name!r} missing or non-finite: {value!r}")
        return float(value)

    @property
    def val_rmsd_top1(self) -> float:
        return self._required("val_rmsd_top1")

    @property
    def val_rmse(self) -> float:
        return self._required("val_rmse")

    def objectives(self) -> tuple[float, float]:
        return (self.val_rmsd_top1, self.val_rmse)
