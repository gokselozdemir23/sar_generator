
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# Built-in reference profiles (aggregated from real Telco SAR data)


BUILTIN_REFERENCE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "compute": {
        "%usr":     (25.0, 12.0),
        "%sys":     (6.0,  3.0),
        "%iowait":  (3.0,  2.5),
        "%steal":   (0.8,  1.2),
        "%idle":    (60.0, 15.0),
        "%memused": (55.0, 10.0),
        "await":    (4.0,  3.0),
        "%util":    (25.0, 12.0),
        "txkB/s":   (1200.0, 800.0),
        "ldavg-1":  (4.0,  3.0),
    },
    "ceph_storage": {
        "%usr":     (12.0, 5.0),
        "%sys":     (8.0,  3.0),
        "%iowait":  (6.0,  4.0),
        "%memused": (60.0, 8.0),
        "await":    (6.0,  5.0),
        "svctm":    (1.2,  0.8),
        "%util":    (45.0, 18.0),
        "tps":      (800.0, 300.0),
        "bread/s":  (80000.0, 30000.0),
        "bwrtn/s":  (120000.0, 50000.0),
    },
    "control_plane": {
        "%usr":     (18.0, 8.0),
        "%sys":     (7.0,  3.0),
        "%iowait":  (4.0,  3.0),
        "ldavg-1":  (3.0,  2.0),
        "%memused": (50.0, 8.0),
    },
    "network": {
        "%usr":     (6.0,  3.0),
        "%soft":    (5.0,  3.0),
        "rxpck/s":  (100000.0, 50000.0),
        "txpck/s":  (110000.0, 55000.0),
        "%ifutil":  (65.0, 15.0),
    },
}


# Result dataclasses

@dataclass
class ComparisonResult:
    """Tek bir metrik için karşılaştırma sonucu."""
    metric:          str
    node_type:       str
    synthetic_mean:  float
    synthetic_std:   float
    reference_mean:  float
    reference_std:   float
    ks_statistic:    float      # Kolmogorov-Smirnov test statistic
    ks_pvalue:       float
    mean_diff_pct:   float      # Yüzde fark
    passed:          bool
    details:         str = ""


@dataclass
class ComparisonReport:
    """Tüm karşılaştırma sonuçları."""
    results:     List[ComparisonResult] = field(default_factory=list)
    passed:      bool = True
    summary:     str  = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def print_summary(self) -> None:
        total = len(self.results)
        ok = sum(1 for r in self.results if r.passed)
        status = "✅ PASS" if self.passed else "❌ FAIL"

        print(f"\n{'='*65}")
        print(f"  SAR Pattern Comparison  {status}")
        print(f"{'='*65}")
        print(f"  Checks: {ok}/{total} passed")
        print()
        for r in self.results:
            icon = "✓" if r.passed else "✗"
            print(
                f"  [{icon}] {r.node_type:<16} {r.metric:<14} "
                f"syn={r.synthetic_mean:>10.2f}  "
                f"ref={r.reference_mean:>10.2f}  "
                f"diff={r.mean_diff_pct:>6.1f}%  "
                f"KS={r.ks_statistic:.3f}"
            )
        print(f"\n  Summary: {self.summary}")
        print(f"{'='*65}\n")


# Comparator

class SARPatternComparator:
    """
    Sentetik SAR verisini referans profillerle karşılaştırır.

    """

    # Mean fark eşiği (% olarak)
    MEAN_DIFF_THRESHOLD = 60.0

    def __init__(self, synthetic_df: pd.DataFrame):
        self._df = synthetic_df.copy()

    @staticmethod
    def _infer_node_type(hostname: str) -> str:
        if hostname.startswith("compute"):
            return "compute"
        if hostname.startswith("ceph"):
            return "ceph_storage"
        if hostname.startswith("ctrl"):
            return "control_plane"
        if hostname.startswith("net"):
            return "network"
        return "unknown"

    def compare_builtin(
        self,
        reference: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    ) -> ComparisonReport:
        """
        Built-in veya sağlanan referans profillere karşı karşılaştırma.
        """
        ref = reference or BUILTIN_REFERENCE
        results: List[ComparisonResult] = []

        if "hostname" not in self._df.columns:
            return ComparisonReport(summary="hostname kolonu bulunamadı.")

        for hostname, gdf in self._df.groupby("hostname"):
            ntype = self._infer_node_type(str(hostname))
            if ntype not in ref:
                continue
            for metric, (ref_mean, ref_std) in ref[ntype].items():
                if metric not in gdf.columns:
                    continue
                series = gdf[metric].dropna()
                if len(series) < 10:
                    continue

                syn_mean = float(series.mean())
                syn_std = float(series.std())

                # KS test: sentetik vs referans normal dağılımı
                ref_samples = np.random.default_rng(42).normal(
                    ref_mean, max(ref_std, 0.01), len(series)
                )
                ks_stat, ks_p = sp_stats.ks_2samp(series.values, ref_samples)

                mean_diff = abs(syn_mean - ref_mean) / max(abs(ref_mean), 0.01) * 100
                passed = mean_diff < self.MEAN_DIFF_THRESHOLD

                results.append(ComparisonResult(
                    metric=metric,
                    node_type=ntype,
                    synthetic_mean=round(syn_mean, 3),
                    synthetic_std=round(syn_std, 3),
                    reference_mean=ref_mean,
                    reference_std=ref_std,
                    ks_statistic=round(float(ks_stat), 4),
                    ks_pvalue=round(float(ks_p), 6),
                    mean_diff_pct=round(mean_diff, 2),
                    passed=passed,
                    details=f"KS p={ks_p:.4f}",
                ))

        failed = [r for r in results if not r.passed]
        return ComparisonReport(
            results=results,
            passed=len(failed) == 0,
            summary=(f"{len(failed)} kontrol başarısız."
                     if failed else "Tüm kontroller başarılı."),
        )

    def compare_with(
        self,
        reference_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
    ) -> ComparisonReport:
        """
        Gerçek bir SAR CSV/DataFrame ile karşılaştırma.
        """
        target_cols = metrics or [
            c for c in self._df.columns
            if pd.api.types.is_numeric_dtype(self._df[c])
            and c not in {"TTY", "blocked"}
        ]
        results: List[ComparisonResult] = []

        for col in target_cols:
            if col not in reference_df.columns:
                continue
            syn = self._df[col].dropna()
            ref = reference_df[col].dropna()
            if len(syn) < 10 or len(ref) < 10:
                continue

            ks_stat, ks_p = sp_stats.ks_2samp(syn.values, ref.values)
            syn_m, ref_m = float(syn.mean()), float(ref.mean())
            diff = abs(syn_m - ref_m) / max(abs(ref_m), 0.01) * 100
            passed = diff < self.MEAN_DIFF_THRESHOLD

            results.append(ComparisonResult(
                metric=col,
                node_type="all",
                synthetic_mean=round(syn_m, 3),
                synthetic_std=round(float(syn.std()), 3),
                reference_mean=round(ref_m, 3),
                reference_std=round(float(ref.std()), 3),
                ks_statistic=round(float(ks_stat), 4),
                ks_pvalue=round(float(ks_p), 6),
                mean_diff_pct=round(diff, 2),
                passed=passed,
                details=f"KS p={ks_p:.4f}",
            ))

        failed = [r for r in results if not r.passed]
        return ComparisonReport(
            results=results,
            passed=len(failed) == 0,
            summary=(f"{len(failed)} kontrol başarısız."
                     if failed else "Tüm kontroller başarılı."),
        )


# CLI entry point

def _cli_main() -> None:
    """python -m validation.comparison <csv> [--reference ref.csv] [--json]"""
    import argparse

    parser = argparse.ArgumentParser(description="SAR Pattern Comparison Tool")
    parser.add_argument("csv_path", help="Sentetik SAR CSV dosyası")
    parser.add_argument("--reference", default=None, help="Gerçek SAR referans CSV")
    parser.add_argument("--use-builtin", action="store_true", help="Built-in profil kullan")
    parser.add_argument("--json", action="store_true", help="JSON çıktı")
    parser.add_argument("--out", default=None, help="Raporu dosyaya kaydet")
    args = parser.parse_args()

    syn_df = pd.read_csv(args.csv_path)
    comparator = SARPatternComparator(syn_df)

    if args.reference:
        ref_df = pd.read_csv(args.reference)
        report = comparator.compare_with(ref_df)
    else:
        report = comparator.compare_builtin()

    if args.json:
        print(report.to_json())
    else:
        report.print_summary()

    if args.out:
        Path(args.out).write_text(report.to_json())
        print(f"Rapor kaydedildi: {args.out}")

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    _cli_main()
