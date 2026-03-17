"""
Unit tests for the SAR Log Data Generator.
Covers configuration, generation engine, anomaly injection, and output adapters.
"""
from __future__ import annotations

import os
import sys
import tempfile
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    AnomalyFrequency, AnomalySeverity, NodeConfig, NodeType,
    OutputConfig, OutputFormat, ScenarioConfig, ScenarioType,
    SimulationConfig, get_default_config,
)
from engine.generator import SARDataGenerator, SAR_COLUMNS, TimeSeriesGenerator
from engine.anomaly import AnomalyEngine
from adapters.output import CSVOutputAdapter, JSONOutputAdapter, OutputManager
from models.node_profiles import get_profile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def short_config() -> SimulationConfig:
    """A minimal 1-hour, 2-node config for fast tests."""
    return SimulationConfig(
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 1, 0, 0),
        interval_seconds=300,
        random_seed=42,
        nodes=[
            NodeConfig(type=NodeType.COMPUTE, count=2, base_load=0.5,
                       total_memory_kb=65536000),
            NodeConfig(type=NodeType.CEPH_STORAGE, count=1, base_load=0.6,
                       total_memory_kb=131072000),
        ],
        output=OutputConfig(format=OutputFormat.CSV, output_dir="/tmp/sar_test"),
    )


@pytest.fixture
def scenario_config() -> SimulationConfig:
    """Config with a storage_contention scenario."""
    return SimulationConfig(
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 6, 0, 0),
        interval_seconds=300,
        random_seed=7,
        nodes=[
            NodeConfig(type=NodeType.CEPH_STORAGE, count=1, base_load=0.5,
                       total_memory_kb=131072000),
        ],
        scenarios=[
            ScenarioConfig(
                type=ScenarioType.STORAGE_CONTENTION,
                start_time=datetime(2024, 1, 1, 2, 0, 0),
                duration_hours=2,
                severity=AnomalySeverity.HIGH,
            ),
        ],
        output=OutputConfig(format=OutputFormat.CSV, output_dir="/tmp/sar_test"),
    )


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_default_config_valid(self):
        cfg = get_default_config()
        assert cfg.num_intervals > 0
        assert len(cfg.nodes) > 0

    def test_config_validation_end_before_start(self):
        with pytest.raises(Exception):
            SimulationConfig(
                start_time=datetime(2024, 1, 2),
                end_time=datetime(2024, 1, 1),
                nodes=[NodeConfig(type=NodeType.COMPUTE, count=1)],
            )

    def test_num_intervals_calculation(self, short_config):
        # 1 hour / 300s = 12 intervals
        assert short_config.num_intervals == 12

    def test_scenario_end_time_auto_computed(self):
        sc = ScenarioConfig(
            type=ScenarioType.MEMORY_PRESSURE,
            start_time=datetime(2024, 1, 1, 2, 0, 0),
            duration_hours=3.0,
        )
        assert sc.end_time == datetime(2024, 1, 1, 5, 0, 0)


# ---------------------------------------------------------------------------
# Node profile tests
# ---------------------------------------------------------------------------

class TestNodeProfiles:
    @pytest.mark.parametrize("node_type", [
        "compute", "ceph_storage", "control_plane", "network"
    ])
    def test_profiles_loadable(self, node_type):
        profile = get_profile(node_type)
        assert profile is not None
        assert len(profile.metrics) > 0

    def test_unknown_node_type_raises(self):
        with pytest.raises(ValueError):
            get_profile("nonexistent_type")

    def test_compute_has_key_metrics(self):
        p = get_profile("compute")
        for key in ["%steal", "%iowait", "kbmemused", "txkB/s"]:
            assert key in p.metrics, f"Missing key metric: {key}"

    def test_ceph_has_key_metrics(self):
        p = get_profile("ceph_storage")
        for key in ["await", "svctm", "%util", "tps", "bread/s", "bwrtn/s"]:
            assert key in p.metrics, f"Missing key metric: {key}"

    def test_network_has_key_metrics(self):
        p = get_profile("network")
        for key in ["rxpck/s", "txpck/s", "%soft", "%ifutil"]:
            assert key in p.metrics, f"Missing key metric: {key}"

    def test_control_has_key_metrics(self):
        p = get_profile("control_plane")
        for key in ["ldavg-1", "%iowait"]:
            assert key in p.metrics, f"Missing key metric: {key}"


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

class TestTimeSeriesGenerator:
    def test_generates_correct_row_count(self, short_config):
        node_cfg = short_config.nodes[0]
        rng = np.random.default_rng(42)
        gen = TimeSeriesGenerator(short_config, node_cfg, "compute-01", rng)
        df = gen.generate()
        assert len(df) == short_config.num_intervals

    def test_all_sar_columns_present(self, short_config):
        node_cfg = short_config.nodes[0]
        rng = np.random.default_rng(42)
        gen = TimeSeriesGenerator(short_config, node_cfg, "compute-01", rng)
        df = gen.generate()
        for col in SAR_COLUMNS:
            assert col in df.columns, f"Missing SAR column: {col}"

    def test_cpu_percentages_sum_to_100(self, short_config):
        node_cfg = short_config.nodes[0]
        rng = np.random.default_rng(42)
        gen = TimeSeriesGenerator(short_config, node_cfg, "compute-01", rng)
        df = gen.generate()
        cpu_cols = ['%usr', '%nice', '%sys', '%iowait', '%steal',
                    '%irq', '%soft', '%guest', '%gnice', '%idle']
        cpu_sum = df[cpu_cols].sum(axis=1)
        # Should be close to 100 (within floating point tolerance)
        assert (np.abs(cpu_sum - 100.0) < 1.0).all(), \
            f"CPU columns don't sum to 100: min={cpu_sum.min():.2f} max={cpu_sum.max():.2f}"

    def test_memory_values_non_negative(self, short_config):
        node_cfg = short_config.nodes[0]
        rng = np.random.default_rng(42)
        gen = TimeSeriesGenerator(short_config, node_cfg, "compute-01", rng)
        df = gen.generate()
        for col in ['kbmemfree', 'kbmemused', 'kbbuffers', 'kbcached']:
            assert (df[col] >= 0).all(), f"Negative values in {col}"

    def test_memory_used_plus_free_equals_total(self, short_config):
        node_cfg = short_config.nodes[0]
        rng = np.random.default_rng(42)
        gen = TimeSeriesGenerator(short_config, node_cfg, "compute-01", rng)
        df = gen.generate()
        total = node_cfg.total_memory_kb
        # Within 1% tolerance due to rounding
        delta = np.abs((df['kbmemused'] + df['kbmemfree']) - total)
        assert (delta / total < 0.01).all()

    def test_datetime_column_format(self, short_config):
        node_cfg = short_config.nodes[0]
        rng = np.random.default_rng(42)
        gen = TimeSeriesGenerator(short_config, node_cfg, "compute-01", rng)
        df = gen.generate()
        # Verify DateTime is parseable
        pd.to_datetime(df['DateTime'])

    def test_hostname_set(self, short_config):
        node_cfg = short_config.nodes[0]
        rng = np.random.default_rng(42)
        gen = TimeSeriesGenerator(short_config, node_cfg, "compute-99", rng)
        df = gen.generate()
        assert (df['hostname'] == 'compute-99').all()


# ---------------------------------------------------------------------------
# Full generator tests
# ---------------------------------------------------------------------------

class TestSARDataGenerator:
    def test_generates_all_nodes(self, short_config):
        gen = SARDataGenerator(short_config)
        df = gen.generate_all()
        expected_hostnames = {
            'compute-01', 'compute-02', 'ceph-01'
        }
        actual = set(df['hostname'].unique())
        assert actual == expected_hostnames

    def test_total_row_count(self, short_config):
        gen = SARDataGenerator(short_config)
        df = gen.generate_all()
        # 3 nodes * 12 intervals
        assert len(df) == 3 * 12

    def test_chunk_generation(self, short_config):
        gen = SARDataGenerator(short_config)
        chunks = list(gen.generate_chunks(chunk_size=10))
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 3 * 12


# ---------------------------------------------------------------------------
# Anomaly engine tests
# ---------------------------------------------------------------------------

class TestAnomalyEngine:
    def test_storage_contention_raises_await(self, scenario_config):
        gen = SARDataGenerator(scenario_config)
        df = gen.generate_all()

        # Compare await during scenario vs baseline
        dt = pd.to_datetime(df['DateTime'])
        baseline = df[dt < pd.Timestamp('2024-01-01 02:00:00')]['await'].mean()
        during = df[
            (dt >= pd.Timestamp('2024-01-01 02:00:00')) &
            (dt < pd.Timestamp('2024-01-01 04:00:00'))
        ]['await'].mean()

        assert during > baseline * 1.5, \
            f"Await should increase during storage_contention: baseline={baseline:.2f} during={during:.2f}"

    def test_storage_contention_raises_util(self, scenario_config):
        gen = SARDataGenerator(scenario_config)
        df = gen.generate_all()
        dt = pd.to_datetime(df['DateTime'])
        baseline = df[dt < pd.Timestamp('2024-01-01 02:00:00')]['%util'].mean()
        during = df[
            (dt >= pd.Timestamp('2024-01-01 02:00:00')) &
            (dt < pd.Timestamp('2024-01-01 04:00:00'))
        ]['%util'].mean()
        assert during > baseline, \
            f"%util should increase: baseline={baseline:.2f} during={during:.2f}"

    def test_no_negative_percentages_after_anomaly(self, scenario_config):
        gen = SARDataGenerator(scenario_config)
        df = gen.generate_all()
        pct_cols = [c for c in df.columns if c.startswith('%')]
        for col in pct_cols:
            assert (df[col] >= 0).all(), f"Negative values in {col} after anomaly"


# ---------------------------------------------------------------------------
# Output adapter tests
# ---------------------------------------------------------------------------

class TestOutputAdapters:
    def test_csv_output_has_all_columns(self, short_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            short_config.output.output_dir = tmpdir
            short_config.output.format = OutputFormat.CSV

            gen = SARDataGenerator(short_config)
            df = gen.generate_all()

            adapter = CSVOutputAdapter(short_config.output)
            path = adapter.write(df)

            result = pd.read_csv(path)
            for col in SAR_COLUMNS:
                assert col in result.columns, f"Missing SAR column in CSV: {col}"

    def test_json_output_structure(self, short_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            short_config.output.output_dir = tmpdir
            short_config.output.format = OutputFormat.JSON

            gen = SARDataGenerator(short_config)
            df = gen.generate_all()

            adapter = JSONOutputAdapter(short_config.output)
            path = adapter.write(df)

            with open(path) as f:
                payload = json.load(f)

            assert "metadata" in payload
            assert "nodes" in payload
            assert payload["metadata"]["total_records"] == len(df)

    def test_output_manager_both_formats(self, short_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            short_config.output.output_dir = tmpdir
            short_config.output.format = OutputFormat.BOTH

            gen = SARDataGenerator(short_config)
            df = gen.generate_all()

            mgr = OutputManager(short_config.output)
            results = mgr.write(df)

            assert 'csv' in results
            assert 'json' in results
            assert Path(results['csv'][0]).exists()
            assert Path(results['json'][0]).exists()

    def test_csv_row_count_preserved(self, short_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            short_config.output.output_dir = tmpdir

            gen = SARDataGenerator(short_config)
            df = gen.generate_all()

            adapter = CSVOutputAdapter(short_config.output)
            path = adapter.write(df)

            result = pd.read_csv(path)
            assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Performance test (optional, skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Performance test - run manually")
def test_throughput_1m_rows():
    """Test that generator can produce >1M rows in under 60 seconds."""
    import time

    cfg = SimulationConfig(
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 8),   # 1 week
        interval_seconds=60,             # 1-minute interval
        random_seed=42,
        nodes=[
            NodeConfig(type=NodeType.COMPUTE, count=15, base_load=0.4),
            NodeConfig(type=NodeType.CEPH_STORAGE, count=5, base_load=0.6),
        ],
        output=OutputConfig(format=OutputFormat.CSV, output_dir="/tmp"),
    )

    t0 = time.perf_counter()
    gen = SARDataGenerator(cfg)
    df = gen.generate_all()
    elapsed = time.perf_counter() - t0

    rows_per_sec = len(df) / elapsed
    print(f"\nThroughput: {len(df):,} rows in {elapsed:.2f}s = {rows_per_sec:,.0f} rows/s")
    assert len(df) >= 1_000_000
    assert rows_per_sec > 100_000   # Conservative threshold
