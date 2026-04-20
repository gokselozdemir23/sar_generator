"""
Yeni modüller için birim testler.

Mevcut test_generator.py'ye dokunmaz; tamamen bağımsız çalışır.
pytest veya doğrudan `python tests/test_extensions.py` ile çalıştırılabilir.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_ok = _fail = 0

def T(name: str, fn):
    global _ok, _fail
    try:
        fn()
        print(f"  ✓ {name}")
        _ok += 1
    except Exception as exc:
        print(f"  ✗ {name}: {exc}")
        traceback.print_exc()
        _fail += 1


def _make_sample_df(n_rows: int = 200) -> pd.DataFrame:
    """Test için küçük bir SAR DataFrame üret."""
    from config import (
        SimulationConfig, NodeConfig, NodeType, OutputConfig,
        OutputFormat, AnomalyFrequency,
    )
    from engine.generator import SARDataGenerator

    cfg = SimulationConfig(
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 1, 2, 0, 0),
        interval_seconds=300,
        random_seed=42,
        nodes=[
            NodeConfig(type=NodeType.COMPUTE,      count=2, base_load=0.5,
                       total_memory_kb=65_536_000,
                       anomaly_frequency=AnomalyFrequency.NONE),
            NodeConfig(type=NodeType.CEPH_STORAGE,  count=1, base_load=0.6,
                       total_memory_kb=131_072_000,
                       anomaly_frequency=AnomalyFrequency.NONE),
        ],
        output=OutputConfig(format=OutputFormat.CSV, output_dir="/tmp"),
    )
    return SARDataGenerator(cfg).generate_all()


# 1. Database adapter config tests

def _suite_database_config():
    print("\n── Database Config ─────────────────────────────────────")

    def _influx_defaults():
        from adapters.database import InfluxDBConfig
        c = InfluxDBConfig()
        assert c.enabled is False
        assert c.port == 8086
        assert c.url == "http://localhost:8086"

    T("InfluxDBConfig defaults", _influx_defaults)

    def _pg_defaults():
        from adapters.database import PostgreSQLConfig
        c = PostgreSQLConfig(enabled=True, password="secret")
        assert c.enabled is True
        assert c.pool_size == 5
        assert c.table == "sar_data"

    T("PostgreSQLConfig defaults", _pg_defaults)

    def _prometheus_defaults():
        from adapters.database import PrometheusConfig
        c = PrometheusConfig(enabled=True)
        assert c.port == 9100
        assert c.prefix == "sar"
        assert c.metrics == []

    T("PrometheusConfig defaults", _prometheus_defaults)

    def _db_pipeline_from_config():
        from adapters.database import DatabaseConfig, DatabasePipeline
        cfg = DatabaseConfig()  # all disabled
        pipeline = DatabasePipeline.from_config(cfg)
        assert len(pipeline.active_writers) == 0

    T("DatabasePipeline.from_config (all disabled) → empty", _db_pipeline_from_config)

    def _db_pipeline_influx_enabled():
        from adapters.database import DatabaseConfig, InfluxDBConfig, DatabasePipeline, InfluxDBWriter
        cfg = DatabaseConfig(influxdb=InfluxDBConfig(enabled=True))
        pipeline = DatabasePipeline.from_config(cfg)
        assert len(pipeline.active_writers) == 1
        assert isinstance(pipeline.active_writers[0], InfluxDBWriter)

    T("DatabasePipeline builds InfluxDBWriter when enabled", _db_pipeline_influx_enabled)

    def _base_writer_retry_raises():
        from adapters.database import BaseWriter
        import pandas as pd

        class AlwaysFail(BaseWriter):
            def connect(self): pass
            def write(self, df): raise ConnectionError("mock failure")
            def close(self): pass

        w = AlwaysFail(type("C", (), {"retries": 2})())
        try:
            w.write_with_retry(pd.DataFrame())
            assert False, "should raise"
        except RuntimeError as e:
            assert "failed after" in str(e)

    T("BaseWriter.write_with_retry raises after max retries", _base_writer_retry_raises)


# 2. Streaming config tests

def _suite_streaming():
    print("\n── Streaming ───────────────────────────────────────────")

    def _config_defaults():
        from streaming.streamer import StreamingConfig, StreamMode
        c = StreamingConfig()
        assert c.enabled is False
        assert c.mode == StreamMode.STDOUT
        assert c.port == 9000

    T("StreamingConfig defaults", _config_defaults)

    def _stdout_streamer_sends(capsys=None):
        import io
        from streaming.streamer import StreamingConfig, StreamMode, StdoutStreamer
        cfg = StreamingConfig(enabled=True, mode=StreamMode.STDOUT, record_format="json")
        streamer = StdoutStreamer(cfg)

        df = _make_sample_df()
        # Capture stdout
        old_stdout = sys.stdout
        buf = io.StringIO()
        sys.stdout = buf
        try:
            streamer.send_chunk(df.head(5))
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()
        lines = [l for l in output.strip().split("\n") if l]
        assert len(lines) == 5, f"Expected 5 lines, got {len(lines)}"
        # Her satır geçerli JSON olmalı
        for line in lines:
            obj = json.loads(line)
            assert "DateTime" in obj

    T("StdoutStreamer: 5 row → 5 JSON lines", _stdout_streamer_sends)

    def _csv_format():
        import io
        from streaming.streamer import StreamingConfig, StreamMode, StdoutStreamer
        cfg = StreamingConfig(enabled=True, mode=StreamMode.STDOUT, record_format="csv")
        streamer = StdoutStreamer(cfg)
        df = _make_sample_df().head(3)

        old_stdout = sys.stdout
        buf = io.StringIO()
        sys.stdout = buf
        try:
            streamer.send_chunk(df)
        finally:
            sys.stdout = old_stdout

        lines = [l for l in buf.getvalue().strip().split("\n") if l]
        # header + 3 data rows
        assert len(lines) == 4, f"Expected 4 lines (header+3), got {len(lines)}"

    T("StdoutStreamer: CSV format includes header", _csv_format)

    def _sar_streamer_disabled():
        from streaming.streamer import StreamingConfig, SARStreamer
        cfg = StreamingConfig(enabled=False)
        s = SARStreamer(cfg)
        s.start()
        s.send_chunk(_make_sample_df().head(2))  # no-op, should not crash
        s.stop()

    T("SARStreamer: disabled → no-op", _sar_streamer_disabled)

    def _sar_streamer_context_manager():
        from streaming.streamer import StreamingConfig, StreamMode, SARStreamer
        cfg = StreamingConfig(enabled=True, mode=StreamMode.STDOUT, record_format="json")
        df = _make_sample_df().head(2)
        import io
        old = sys.stdout; sys.stdout = io.StringIO()
        try:
            with SARStreamer(cfg) as s:
                s.send_chunk(df)
        finally:
            sys.stdout = old

    T("SARStreamer: context manager works", _sar_streamer_context_manager)


# 3. File rotation tests

def _suite_rotation():
    print("\n── File Rotation ───────────────────────────────────────")

    def _size_rotation():
        from adapters.rotation import RotatingFileHandle, RotationConfig, RotationStrategy
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "test_rot"
            cfg = RotationConfig(strategy=RotationStrategy.SIZE, max_mb=0.0001)  # 100 bytes
            handle = RotatingFileHandle(base, cfg, ".csv")
            # Yeterince veri yaz → rotasyonu tetikle
            for _ in range(50):
                handle.write("a" * 10 + "\n")
            handle.close()
            # En az 1 rotasyon dosyası olmalı
            assert len(handle.rotated_files) >= 1

    T("RotatingFileHandle: size-based rotation triggers", _size_rotation)

    def _no_rotation_small_data():
        from adapters.rotation import RotatingFileHandle, RotationConfig, RotationStrategy
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "no_rot"
            cfg = RotationConfig(strategy=RotationStrategy.SIZE, max_mb=100)
            handle = RotatingFileHandle(base, cfg, ".csv")
            handle.write("small data\n")
            handle.close()
            # Sadece aktif dosya, hiç rotasyon yok
            assert len(handle.rotated_files) == 1  # close ekler aktifi

    T("RotatingFileHandle: no rotation for small data", _no_rotation_small_data)

    def _rotating_csv_adapter():
        from adapters.rotation import RotatingCSVAdapter, RotationConfig, RotationStrategy
        df = _make_sample_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = RotationConfig(strategy=RotationStrategy.SIZE, max_mb=100)
            with RotatingCSVAdapter(tmpdir, "test", cfg) as adapter:
                for i in range(0, len(df), 10):
                    adapter.write(df.iloc[i:i+10])
            # Dizinde CSV dosyası var mı?
            csv_files = list(Path(tmpdir).glob("*.csv"))
            assert len(csv_files) >= 1

    T("RotatingCSVAdapter: writes CSV files", _rotating_csv_adapter)

    def _compression():
        from adapters.rotation import RotatingFileHandle, RotationConfig, RotationStrategy
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "comp"
            cfg = RotationConfig(strategy=RotationStrategy.SIZE, max_mb=0.0001, compress=True)
            handle = RotatingFileHandle(base, cfg, ".csv")
            for _ in range(100):
                handle.write("compress me " * 5 + "\n")
            handle.close()
            gz_files = list(Path(tmpdir).glob("*.csv.gz"))
            assert len(gz_files) >= 1, f"No .gz files found in {list(Path(tmpdir).iterdir())}"

    T("RotatingFileHandle: compression produces .gz files", _compression)

    def _rotating_json_adapter():
        from adapters.rotation import RotatingJSONAdapter, RotationConfig, RotationStrategy
        df = _make_sample_df().head(20)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = RotationConfig(strategy=RotationStrategy.SIZE, max_mb=100)
            with RotatingJSONAdapter(tmpdir, "test_json", cfg) as adapter:
                adapter.write(df)
            ndjson_files = list(Path(tmpdir).glob("*.ndjson"))
            assert len(ndjson_files) >= 1

    T("RotatingJSONAdapter: writes .ndjson files", _rotating_json_adapter)


# 4. Statistical validation tests

def _suite_validation():
    print("\n── Statistical Validation ──────────────────────────────")

    def _report_structure():
        from validation.statistics import StatisticalValidator
        df = _make_sample_df()
        v = StatisticalValidator(df)
        report = v.run_all()
        assert report.total_rows == len(df)
        assert report.total_columns == len(df.columns)
        assert len(report.hostnames) >= 1

    T("StatisticalValidator: report structure OK", _report_structure)

    def _distribution_checks():
        from validation.statistics import StatisticalValidator
        df = _make_sample_df()
        v = StatisticalValidator(df)
        checks = v.check_distributions()
        assert len(checks) > 0
        # Her check'in zorunlu alanları var
        for c in checks:
            assert hasattr(c, "metric")
            assert hasattr(c, "passed")
            assert isinstance(c.actual_mean, float)

    T("check_distributions: returns MetricCheck list", _distribution_checks)

    def _correlation_checks():
        from validation.statistics import StatisticalValidator
        df = _make_sample_df()
        v = StatisticalValidator(df)
        checks = v.check_correlations()
        # Her check'in zorunlu alanları var
        for c in checks:
            assert -1.0 <= c.correlation <= 1.0

    T("check_correlations: Pearson r ∈ [-1, 1]", _correlation_checks)

    def _anomaly_checks():
        from validation.statistics import StatisticalValidator
        df = _make_sample_df()
        v = StatisticalValidator(df)
        checks = v.check_anomaly_frequencies(metric="await")
        assert len(checks) >= 1
        for c in checks:
            assert 0.0 <= c.anomaly_fraction <= 1.0

    T("check_anomaly_frequencies: fractions ∈ [0,1]", _anomaly_checks)

    def _null_check():
        from validation.statistics import StatisticalValidator
        df = _make_sample_df()
        v = StatisticalValidator(df)
        nulls = v.null_check()
        assert isinstance(nulls, dict)

    T("null_check: returns dict", _null_check)

    def _to_json():
        from validation.statistics import StatisticalValidator
        df = _make_sample_df()
        report = StatisticalValidator(df).run_all()
        j = report.to_json()
        parsed = json.loads(j)
        assert "total_rows" in parsed
        assert "metric_checks" in parsed

    T("ValidationReport.to_json: valid JSON", _to_json)

    def _cpu_range_valid():
        """Üretilen CPU değerleri ≥ 0 ve ≤ 100 olmalı."""
        from validation.statistics import StatisticalValidator
        df = _make_sample_df()
        v = StatisticalValidator(df)
        ranges = v.range_check()
        for col in ["%usr", "%sys", "%iowait", "%idle"]:
            if col in ranges:
                assert ranges[col]["min"] >= 0.0, f"{col} min < 0"
                assert ranges[col]["max"] <= 100.0, f"{col} max > 100"

    T("range_check: CPU %% columns within [0, 100]", _cpu_range_valid)


# ---------------------------------------------------------------------------
# 5. Visualization tests (headless)
# ---------------------------------------------------------------------------

def _suite_visualization():
    print("\n── Visualization ───────────────────────────────────────")

    def _plot_cpu():
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("    ⚠ matplotlib yok, atlanıyor.")
            return
        from validation.plots import plot_cpu_timeseries
        df = _make_sample_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = plot_cpu_timeseries(df, output_dir=tmpdir)
            assert Path(p).exists()

    T("plot_cpu_timeseries: creates PNG file", _plot_cpu)

    def _plot_memory():
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            return
        from validation.plots import plot_memory_trends
        df = _make_sample_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = plot_memory_trends(df, output_dir=tmpdir)
            assert Path(p).exists()

    T("plot_memory_trends: creates PNG file", _plot_memory)

    def _plot_network():
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            return
        from validation.plots import plot_network_throughput
        df = _make_sample_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = plot_network_throughput(df, output_dir=tmpdir)
            assert Path(p).exists()

    T("plot_network_throughput: creates PNG file", _plot_network)

    def _plot_anomaly():
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            return
        from validation.plots import plot_anomaly_distribution
        df = _make_sample_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = plot_anomaly_distribution(df, output_dir=tmpdir)
            assert Path(p).exists()

    T("plot_anomaly_distribution: creates PNG file", _plot_anomaly)

    def _generate_all():
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            return
        from validation.plots import generate_all_plots
        df = _make_sample_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_all_plots(df, tmpdir)
            assert len(paths) >= 4

    T("generate_all_plots: creates ≥ 4 files", _generate_all)


# ---------------------------------------------------------------------------
# 6. Benchmark tests
# ---------------------------------------------------------------------------

def _suite_benchmark():
    print("\n── Benchmark ───────────────────────────────────────────")

    def _result_fields():
        from benchmark.performance import BenchmarkResult
        r = BenchmarkResult(
            total_nodes=5, days=1, interval_seconds=300, warmup_runs=0,
            total_rows=100, total_seconds=0.5, rows_per_second=200.0,
            peak_memory_mb=10.0, avg_memory_mb=8.0,
            chunk_times_s=[0.1, 0.2, 0.15],
            chunk_rows=[50, 50, 0],
        )
        assert r.p50_chunk_s == 0.15
        assert 0.0 < r.p95_chunk_s <= 0.2
        d = r.to_dict()
        assert "rows_per_second" in d

    T("BenchmarkResult: fields and p50/p95 OK", _result_fields)

    def _small_benchmark():
        from benchmark.performance import PerformanceBenchmark
        bm = PerformanceBenchmark(
            total_nodes=3, days=1, interval_seconds=300,
            warmup_runs=0, chunk_size=10_000,
        )
        result = bm.run()
        assert result.total_rows > 0
        assert result.rows_per_second > 0
        assert result.peak_memory_mb >= 0

    T("PerformanceBenchmark: 3 nodes × 1 day → valid result", _small_benchmark)

    def _benchmark_node_dist():
        from benchmark.performance import PerformanceBenchmark
        bm = PerformanceBenchmark(
            total_nodes=4, days=1, interval_seconds=600,
            warmup_runs=0,
            node_distribution={"compute": 2, "ceph_storage": 1, "network": 1},
        )
        r = bm.run()
        assert r.total_rows > 0

    T("PerformanceBenchmark: custom node_distribution", _benchmark_node_dist)


# ---------------------------------------------------------------------------
# 7. Backward compatibility check
# ---------------------------------------------------------------------------

def _suite_backward_compat():
    print("\n── Backward Compatibility ──────────────────────────────")

    def _existing_generator_unchanged():
        from config import get_default_config
        from engine.generator import SARDataGenerator, SAR_COLUMNS
        cfg = get_default_config()
        # Sadece 1 saatlik test için konfigürasyonu kısalt
        from datetime import datetime
        cfg.start_time = datetime(2024, 1, 1, 0, 0, 0)
        cfg.end_time   = datetime(2024, 1, 1, 1, 0, 0)
        cfg.scenarios  = []
        df = SARDataGenerator(cfg).generate_all()
        assert len(df.columns) == len(SAR_COLUMNS)

    T("Existing SARDataGenerator still produces all SAR columns", _existing_generator_unchanged)

    def _existing_output_adapter():
        from config import SimulationConfig, NodeConfig, NodeType, OutputConfig, OutputFormat
        from engine.generator import SARDataGenerator
        from adapters.output import CSVOutputAdapter, JSONOutputAdapter
        from datetime import datetime

        cfg = SimulationConfig(
            start_time=datetime(2024, 1, 1), end_time=datetime(2024, 1, 1, 0, 30),
            interval_seconds=300,
            nodes=[NodeConfig(type=NodeType.COMPUTE, count=1, total_memory_kb=32_768_000)],
            output=OutputConfig(format=OutputFormat.CSV, output_dir="/tmp"),
        )
        df = SARDataGenerator(cfg).generate_all()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = OutputConfig(format=OutputFormat.CSV, output_dir=tmpdir)
            p = CSVOutputAdapter(out).write(df)
            assert Path(p).exists()
            out2 = OutputConfig(format=OutputFormat.JSON, output_dir=tmpdir)
            p2 = JSONOutputAdapter(out2).write(df)
            assert Path(p2).exists()

    T("Existing CSV/JSON adapters unchanged", _existing_output_adapter)

    def _config_new_optional_fields():
        from config import SimulationConfig, NodeConfig, NodeType
        from datetime import datetime
        # database ve streaming alanları opsiyonel ve geriye uyumlu
        cfg = SimulationConfig(
            start_time=datetime(2024, 1, 1), end_time=datetime(2024, 1, 2),
            nodes=[NodeConfig(type=NodeType.COMPUTE, count=1)],
            database=None, streaming=None,
        )
        assert cfg.database is None
        assert cfg.streaming is None

    T("SimulationConfig: new optional fields don't break existing", _config_new_optional_fields)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("=" * 60)
    print("  SAR Generator – Extension Test Suite")
    print("=" * 60)

    _suite_database_config()
    _suite_streaming()
    _suite_rotation()
    _suite_validation()
    _suite_visualization()
    _suite_benchmark()
    _suite_backward_compat()

    print(f"\n{'='*60}")
    status = "✅ ALL PASSED" if _fail == 0 else f"❌ {_fail} FAILED"
    print(f"  {_ok + _fail} tests: {_ok} passed, {_fail} failed  {status}")
    print(f"{'='*60}\n")
    return _fail == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
