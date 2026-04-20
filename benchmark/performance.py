from __future__ import annotations

import gc
import logging
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Result dataclass

@dataclass
class BenchmarkResult:
    """Benchmark çalışması sonuçları."""

    # Konfigürasyon
    total_nodes:      int
    days:             int
    interval_seconds: int
    warmup_runs:      int

    # Sonuçlar
    total_rows:       int
    total_seconds:    float
    rows_per_second:  float
    peak_memory_mb:   float
    avg_memory_mb:    float

    # Chunk bazlı istatistikler
    chunk_times_s:    List[float] = field(default_factory=list)
    chunk_rows:       List[int]   = field(default_factory=list)

    @property
    def p50_chunk_s(self) -> float:
        """Medyan chunk süresi (saniye)."""
        return statistics.median(self.chunk_times_s) if self.chunk_times_s else 0.0

    @property
    def p95_chunk_s(self) -> float:
        """95. yüzdelik chunk süresi (saniye)."""
        if not self.chunk_times_s:
            return 0.0
        sorted_times = sorted(self.chunk_times_s)
        idx = max(0, int(len(sorted_times) * 0.95) - 1)
        return sorted_times[idx]

    def print_report(self) -> None:
        """Benchmark raporunu konsola yazdır."""
        print(f"\n{'='*60}")
        print("  SAR Generator – Performance Benchmark")
        print(f"{'='*60}")
        print(f"  Config         : {self.total_nodes} node × {self.days} gün"
              f" @ {self.interval_seconds}s interval")
        print(f"  Warmup runs    : {self.warmup_runs}")
        print()
        print(f"  Total rows     : {self.total_rows:>12,}")
        print(f"  Total time     : {self.total_seconds:>11.2f}s")
        print(f"  Rows/second    : {self.rows_per_second:>12,.0f}")
        print()
        print(f"  Peak memory    : {self.peak_memory_mb:>11.1f} MB")
        print(f"  Avg memory     : {self.avg_memory_mb:>11.1f} MB")
        print()
        if self.chunk_times_s:
            print(f"  Chunks         : {len(self.chunk_times_s):>12,}")
            print(f"  Chunk p50      : {self.p50_chunk_s*1000:>10.1f} ms")
            print(f"  Chunk p95      : {self.p95_chunk_s*1000:>10.1f} ms")
        print(f"{'='*60}")

        # Performans yargısı
        rps = self.rows_per_second
        if rps >= 1_000_000:
            grade = "🚀 Mükemmel (>1M rows/s)"
        elif rps >= 500_000:
            grade = "✅ Çok İyi (>500K rows/s)"
        elif rps >= 100_000:
            grade = "👍 İyi (>100K rows/s)"
        elif rps >= 10_000:
            grade = "⚠️  Orta (>10K rows/s)"
        else:
            grade = "❌ Düşük (<10K rows/s)"
        print(f"  Performans     : {grade}")
        print(f"{'='*60}\n")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["p50_chunk_s"] = self.p50_chunk_s
        d["p95_chunk_s"] = self.p95_chunk_s
        return d


# Benchmark engine

class PerformanceBenchmark:
    """
    SAR Data Generator performans benchmark aracı.

    Örnek::

        bm = PerformanceBenchmark(total_nodes=100, days=7)
        result = bm.run()
        result.print_report()
    """

    DEFAULT_CHUNK_SIZE = 50_000

    def __init__(
        self,
        total_nodes:      int   = 18,
        days:             int   = 7,
        interval_seconds: int   = 300,
        warmup_runs:      int   = 1,
        chunk_size:       int   = DEFAULT_CHUNK_SIZE,
        node_distribution: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            total_nodes:       Toplam node sayısı.
            days:              Simülasyon süresi (gün).
            interval_seconds:  Örnekleme aralığı.
            warmup_runs:       Ölçüm öncesi ısınma çalışması sayısı.
            chunk_size:        Chunk başına maksimum satır.
            node_distribution: {node_type: count} dict. None ise eşit dağıtır.
        """
        self.total_nodes       = total_nodes
        self.days              = days
        self.interval_seconds  = interval_seconds
        self.warmup_runs       = warmup_runs
        self.chunk_size        = chunk_size
        self.node_distribution = node_distribution

    def _build_config(self):
        """Benchmark için SimulationConfig oluştur."""
        from config import (
            SimulationConfig, NodeConfig, NodeType,
            AnomalyFrequency, OutputConfig, OutputFormat,
        )

        start = datetime(2024, 1, 1)
        end   = start + timedelta(days=self.days)

        # Node dağılımı
        dist = self.node_distribution or {
            "compute":       max(1, int(self.total_nodes * 0.55)),
            "ceph_storage":  max(1, int(self.total_nodes * 0.20)),
            "control_plane": max(1, int(self.total_nodes * 0.15)),
            "network":       max(1, int(self.total_nodes * 0.10)),
        }

        type_map = {
            "compute":       NodeType.COMPUTE,
            "ceph_storage":  NodeType.CEPH_STORAGE,
            "control_plane": NodeType.CONTROL_PLANE,
            "network":       NodeType.NETWORK,
        }

        nodes = [
            NodeConfig(type=type_map[t], count=cnt, base_load=0.4,
                       anomaly_frequency=AnomalyFrequency.NONE)
            for t, cnt in dist.items() if cnt > 0
        ]

        return SimulationConfig(
            start_time=start,
            end_time=end,
            interval_seconds=self.interval_seconds,
            nodes=nodes,
            scenarios=[],
            output=OutputConfig(format=OutputFormat.CSV, output_dir="/tmp"),
            random_seed=42,
            diurnal_pattern=True,
            weekly_pattern=True,
        )

    def _run_once(self, config) -> Dict[str, Any]:
        """Tek bir benchmark turunu çalıştır; chunk bazlı ölçüm yapar."""
        from engine.generator import SARDataGenerator

        gen = SARDataGenerator(config)
        chunk_times: List[float] = []
        chunk_rows:  List[int]   = []
        total_rows = 0

        mem_samples: List[float] = []
        tracemalloc.start()
        t_total_start = time.perf_counter()

        for chunk_df in gen.generate_chunks(self.chunk_size):
            t0 = time.perf_counter()
            n = len(chunk_df)
            # Force evaluation (chunks are already computed; just measure)
            total_rows += n
            elapsed = time.perf_counter() - t0
            chunk_times.append(elapsed)
            chunk_rows.append(n)

            current_mb, _ = tracemalloc.get_traced_memory()
            mem_samples.append(current_mb / 1024 / 1024)

            del chunk_df
            gc.collect()

        total_elapsed = time.perf_counter() - t_total_start
        _, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "total_rows":    total_rows,
            "total_elapsed": total_elapsed,
            "chunk_times":   chunk_times,
            "chunk_rows":    chunk_rows,
            "peak_mem_mb":   peak_traced / 1024 / 1024,
            "avg_mem_mb":    statistics.mean(mem_samples) if mem_samples else 0.0,
        }

    def run(self) -> BenchmarkResult:
        """
        Benchmark'ı çalıştır; warmup sonrası ölçüm yapar.

        Returns:
            BenchmarkResult – ölçüm sonuçları.
        """
        config = self._build_config()
        expected_rows = config.num_intervals * self.total_nodes
        logger.info(
            "Benchmark başlıyor: %d node × %d gün = ~%,d satır bekleniyor",
            self.total_nodes, self.days, expected_rows,
        )

        # Warmup
        for i in range(self.warmup_runs):
            logger.info("Warmup %d/%d …", i + 1, self.warmup_runs)
            self._run_once(config)
            gc.collect()

        # Ölçüm
        logger.info("Ölçüm başlıyor …")
        m = self._run_once(config)

        rows_per_sec = m["total_rows"] / max(m["total_elapsed"], 1e-9)

        return BenchmarkResult(
            total_nodes=self.total_nodes,
            days=self.days,
            interval_seconds=self.interval_seconds,
            warmup_runs=self.warmup_runs,
            total_rows=m["total_rows"],
            total_seconds=round(m["total_elapsed"], 4),
            rows_per_second=round(rows_per_sec, 1),
            peak_memory_mb=round(m["peak_mem_mb"], 2),
            avg_memory_mb=round(m["avg_mem_mb"], 2),
            chunk_times_s=m["chunk_times"],
            chunk_rows=m["chunk_rows"],
        )


# CLI entry point  (python benchmark.py …)

def _cli_main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="SAR Generator Performans Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nodes",    type=int, default=18,  help="Toplam node sayısı")
    parser.add_argument("--days",     type=int, default=7,   help="Simülasyon günü")
    parser.add_argument("--interval", type=int, default=300, help="Örnekleme aralığı (saniye)")
    parser.add_argument("--warmup",   type=int, default=1,   help="Warmup run sayısı")
    parser.add_argument("--chunk",    type=int, default=50_000, help="Chunk boyutu")
    parser.add_argument("--json",     action="store_true",   help="JSON çıktı ver")
    parser.add_argument("--out",      default=None,          help="Sonucu dosyaya kaydet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    bm = PerformanceBenchmark(
        total_nodes=args.nodes,
        days=args.days,
        interval_seconds=args.interval,
        warmup_runs=args.warmup,
        chunk_size=args.chunk,
    )
    result = bm.run()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        result.print_report()

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"Sonuç kaydedildi: {out_path}")

    sys.exit(0)


if __name__ == "__main__":
    _cli_main()
