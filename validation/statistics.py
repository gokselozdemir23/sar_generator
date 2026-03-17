"""
Statistical Validation Tools – Üretilen SAR verisinin kalitesini analiz eder.

Mevcut generator'a dokunmaz; üretilen DataFrame'i girdi olarak alır.

CLI:
    python -m validation.statistics output/sar_synthetic_full.csv
    python main.py --validate-data output/sar_synthetic_full.csv
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MetricCheck:
    """Tek bir metrik için doğrulama sonucu."""
    metric:       str
    passed:       bool
    actual_mean:  float
    actual_std:   float
    expected_mean: Optional[float] = None
    expected_std:  Optional[float] = None
    details:      str = ""


@dataclass
class CorrelationCheck:
    """İki metrik arasındaki korelasyon doğrulama sonucu."""
    metric_a:    str
    metric_b:    str
    correlation: float
    passed:      bool
    threshold:   float
    direction:   str   # "positive" | "negative" | "any"
    details:     str = ""


@dataclass
class AnomalyCheck:
    """Anomali frekansı doğrulama sonucu."""
    hostname:          str
    metric:            str
    anomaly_fraction:  float
    expected_min:      float
    expected_max:      float
    passed:            bool
    details:           str = ""


@dataclass
class ValidationReport:
    """Tüm doğrulama kontrolleri için özet rapor."""
    total_rows:         int
    total_columns:      int
    hostnames:          List[str]
    metric_checks:      List[MetricCheck]      = field(default_factory=list)
    correlation_checks: List[CorrelationCheck] = field(default_factory=list)
    anomaly_checks:     List[AnomalyCheck]     = field(default_factory=list)
    passed:             bool                   = True
    summary:            str                    = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def print_summary(self) -> None:
        """İnsan okunabilir özet yazdır."""
        total = (
            len(self.metric_checks)
            + len(self.correlation_checks)
            + len(self.anomaly_checks)
        )
        passed = sum([
            sum(1 for c in self.metric_checks      if c.passed),
            sum(1 for c in self.correlation_checks if c.passed),
            sum(1 for c in self.anomaly_checks      if c.passed),
        ])
        status = "✅ PASS" if self.passed else "❌ FAIL"

        print(f"\n{'='*60}")
        print(f"  SAR Data Validation Report  {status}")
        print(f"{'='*60}")
        print(f"  Rows     : {self.total_rows:,}")
        print(f"  Columns  : {self.total_columns}")
        print(f"  Hostnames: {len(self.hostnames)}")
        print(f"  Checks   : {passed}/{total} passed")
        print()

        if self.metric_checks:
            print("── Metric Checks ──────────────────────────────────────")
            for c in self.metric_checks:
                icon = "✓" if c.passed else "✗"
                print(
                    f"  [{icon}] {c.metric:<20} mean={c.actual_mean:>10.2f}  "
                    f"std={c.actual_std:>9.2f}  {c.details}"
                )

        if self.correlation_checks:
            print("\n── Correlation Checks ─────────────────────────────────")
            for c in self.correlation_checks:
                icon = "✓" if c.passed else "✗"
                print(
                    f"  [{icon}] {c.metric_a} ↔ {c.metric_b}  "
                    f"r={c.correlation:.3f}  {c.details}"
                )

        if self.anomaly_checks:
            print("\n── Anomaly Frequency Checks ───────────────────────────")
            for c in self.anomaly_checks:
                icon = "✓" if c.passed else "✗"
                print(
                    f"  [{icon}] {c.hostname:<15} {c.metric:<15} "
                    f"frac={c.anomaly_fraction:.4f}  {c.details}"
                )

        print(f"\n  Summary: {self.summary}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class StatisticalValidator:
    """
    Üretilen SAR verisinin istatistiksel kalitesini doğrular.

    Kontroller:
    1. Metrik dağılım kontrolü (mean/std aralık testi)
    2. Çapraz metrik korelasyon kontrolü
    3. Anomali frekansı kontrolü
    4. Veri bütünlüğü (null, negatif, sınır aşımı)

    Örnek::

        validator = StatisticalValidator(df)
        report = validator.run_all()
        report.print_summary()
    """

    # Beklenen değer aralıkları: (min_mean, max_mean, max_std)
    EXPECTED_RANGES: Dict[str, Tuple[float, float, float]] = {
        "%usr":    (0.0,  80.0, 30.0),
        "%sys":    (0.0,  40.0, 15.0),
        "%iowait": (0.0,  60.0, 15.0),
        "%steal":  (0.0,  30.0, 10.0),
        "%idle":   (5.0,  99.9, 30.0),
        "%memused":(1.0,  99.9, 20.0),
        "%util":   (0.0,  100.0,30.0),
        "await":   (0.0,  2000.0,500.0),
        "ldavg-1": (0.0,  256.0, 20.0),
        "%ifutil": (0.0,  100.0, 20.0),
    }

    # Beklenen pozitif korelasyonlar: (metrik_a, metrik_b, min_r)
    EXPECTED_CORRELATIONS: List[Tuple[str, str, float, str]] = [
        ("%usr",    "ldavg-1",  0.1, "positive"),
        ("%iowait", "await",    0.1, "positive"),
        ("%util",   "await",    0.1, "positive"),
        ("bwrtn/s", "txkB/s",   0.0, "positive"),
        ("kbmemused", "fault/s",0.0, "positive"),
    ]

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Doğrulanacak SAR DataFrame.
        """
        self._df = df.copy()
        self._numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in {"TTY", "blocked"}
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> ValidationReport:
        """
        Tüm kontrolleri çalıştır ve birleşik ValidationReport döndür.

        Returns:
            ValidationReport – tüm kontrollerinin sonucu.
        """
        report = ValidationReport(
            total_rows=len(self._df),
            total_columns=len(self._df.columns),
            hostnames=sorted(self._df["hostname"].unique().tolist())
            if "hostname" in self._df.columns else [],
        )

        report.metric_checks      = self.check_distributions()
        report.correlation_checks = self.check_correlations()
        report.anomaly_checks     = self.check_anomaly_frequencies()

        failed = [
            c for checks in [
                report.metric_checks, report.correlation_checks, report.anomaly_checks
            ]
            for c in checks if not c.passed
        ]
        report.passed  = len(failed) == 0
        report.summary = (
            f"{len(failed)} kontrol başarısız."
            if failed
            else "Tüm kontroller başarılı."
        )
        return report

    def check_distributions(
        self,
        columns: Optional[List[str]] = None,
    ) -> List[MetricCheck]:
        """
        Belirtilen kolonların mean/std değerlerini beklenen aralıklarla karşılaştırır.

        Args:
            columns: Kontrol edilecek kolon listesi. None ise EXPECTED_RANGES kullanılır.

        Returns:
            Her metrik için MetricCheck listesi.
        """
        targets = columns or list(self.EXPECTED_RANGES.keys())
        results: List[MetricCheck] = []

        for col in targets:
            if col not in self._df.columns:
                continue
            series = self._df[col].dropna()
            if series.empty:
                continue

            mean = float(series.mean())
            std  = float(series.std())
            passed = True
            details = ""

            if col in self.EXPECTED_RANGES:
                min_m, max_m, max_s = self.EXPECTED_RANGES[col]
                if not (min_m <= mean <= max_m):
                    passed = False
                    details = f"mean {mean:.2f} ∉ [{min_m}, {max_m}]"
                elif std > max_s:
                    passed = False
                    details = f"std {std:.2f} > max {max_s}"
                else:
                    details = "OK"

            results.append(MetricCheck(
                metric=col, passed=passed,
                actual_mean=round(mean, 4), actual_std=round(std, 4),
                expected_mean=(self.EXPECTED_RANGES[col][0] + self.EXPECTED_RANGES[col][1]) / 2
                if col in self.EXPECTED_RANGES else None,
                details=details,
            ))

        return results

    def check_correlations(
        self,
        pairs: Optional[List[Tuple[str, str, float, str]]] = None,
    ) -> List[CorrelationCheck]:
        """
        Metrik çiftleri arasındaki Pearson korelasyonunu doğrular.

        Args:
            pairs: [(metrik_a, metrik_b, min_r, yön)] listesi.
                   None ise EXPECTED_CORRELATIONS kullanılır.

        Returns:
            CorrelationCheck listesi.
        """
        targets = pairs or self.EXPECTED_CORRELATIONS
        results: List[CorrelationCheck] = []

        for metric_a, metric_b, threshold, direction in targets:
            if metric_a not in self._df.columns or metric_b not in self._df.columns:
                continue

            a = self._df[metric_a].dropna()
            b = self._df[metric_b].dropna()
            common = a.index.intersection(b.index)
            if len(common) < 10:
                continue

            r, p_value = stats.pearsonr(a.loc[common], b.loc[common])
            r = float(r)

            if direction == "positive":
                passed = r >= threshold
            elif direction == "negative":
                passed = r <= -threshold
            else:
                passed = abs(r) >= threshold

            results.append(CorrelationCheck(
                metric_a=metric_a, metric_b=metric_b,
                correlation=round(r, 4), passed=passed,
                threshold=threshold, direction=direction,
                details=f"p={p_value:.4f}",
            ))

        return results

    def check_anomaly_frequencies(
        self,
        metric: str = "await",
        threshold_multiplier: float = 3.0,
    ) -> List[AnomalyCheck]:
        """
        Her hostname için anomali (aşırı değer) frekansını kontrol eder.

        Anomali tanımı: mean + threshold_multiplier * std üzerindeki değerler.

        Args:
            metric:               Kontrol edilecek metrik.
            threshold_multiplier: Kaç sigma üzeri anomali sayılır.

        Returns:
            AnomalyCheck listesi.
        """
        if metric not in self._df.columns or "hostname" not in self._df.columns:
            return []

        results: List[AnomalyCheck] = []
        for hostname, group in self._df.groupby("hostname"):
            series = group[metric].dropna()
            if series.empty:
                continue
            mean = series.mean()
            std  = series.std() or 1.0
            threshold = mean + threshold_multiplier * std
            frac = float((series > threshold).mean())

            # Beklenti: %0 – %5 anomali normal kabul edilir
            passed = 0.0 <= frac <= 0.05
            results.append(AnomalyCheck(
                hostname=str(hostname), metric=metric,
                anomaly_fraction=round(frac, 6),
                expected_min=0.0, expected_max=0.05,
                passed=passed,
                details=f"threshold={threshold:.2f}",
            ))

        return results

    def null_check(self) -> Dict[str, int]:
        """
        Tüm kolonlardaki null değer sayısını döndür.

        Returns:
            {kolon: null_sayısı}
        """
        nulls = self._df[self._numeric_cols].isnull().sum()
        return {col: int(cnt) for col, cnt in nulls.items() if cnt > 0}

    def range_check(self) -> Dict[str, Dict[str, float]]:
        """
        Sayısal kolonların min/max değerlerini döndür.

        Returns:
            {kolon: {min: x, max: y}}
        """
        result: Dict[str, Dict[str, float]] = {}
        for col in self._numeric_cols:
            if col not in self._df.columns:
                continue
            result[col] = {
                "min": float(self._df[col].min()),
                "max": float(self._df[col].max()),
            }
        return result

    def distribution_summary(self) -> pd.DataFrame:
        """
        Tüm sayısal kolonlar için describe() çıktısını döndür.

        Returns:
            pd.DataFrame – istatistik özeti.
        """
        return self._df[self._numeric_cols].describe().T.round(3)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """python -m validation.statistics <csv_path> [--json] [--out report.json]"""
    import argparse

    parser = argparse.ArgumentParser(description="SAR veri doğrulama aracı")
    parser.add_argument("csv_path", help="Doğrulanacak SAR CSV dosyası")
    parser.add_argument("--json", action="store_true", help="JSON çıktı ver")
    parser.add_argument("--out", default=None, help="Raporu dosyaya kaydet")
    args = parser.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        print(f"Dosya bulunamadı: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Yükleniyor: {path} …")
    df = pd.read_csv(path)
    print(f"  {len(df):,} satır, {len(df.columns)} sütun")

    validator = StatisticalValidator(df)
    report = validator.run_all()

    if args.json:
        output = report.to_json()
        print(output)
    else:
        report.print_summary()

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(report.to_json())
        print(f"Rapor kaydedildi: {out_path}")

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    _cli_main()
