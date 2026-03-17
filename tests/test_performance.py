"""
Performance criteria tests – Doküman başarı kriterlerini doğrular.

Çalıştırma:
    pytest tests/test_performance.py -v -m slow
    pytest tests/test_performance.py -v -m slow --timeout=600
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SimulationConfig, NodeConfig, NodeType,
    OutputConfig, OutputFormat, AnomalyFrequency,
)
from engine.generator import SARDataGenerator


@pytest.mark.slow
def test_100_nodes_7_days_under_5_minutes():
    """
    Başarı Kriteri: 1 week × 100 nodes @ 5-min interval < 5 dakika.
    """
    cfg = SimulationConfig(
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 8),
        interval_seconds=300,
        random_seed=42,
        nodes=[
            NodeConfig(type=NodeType.COMPUTE, count=55,
                       base_load=0.4,
                       anomaly_frequency=AnomalyFrequency.NONE),
            NodeConfig(type=NodeType.CEPH_STORAGE, count=20,
                       base_load=0.6,
                       anomaly_frequency=AnomalyFrequency.NONE),
            NodeConfig(type=NodeType.CONTROL_PLANE, count=15,
                       base_load=0.3,
                       anomaly_frequency=AnomalyFrequency.NONE),
            NodeConfig(type=NodeType.NETWORK, count=10,
                       base_load=0.35,
                       anomaly_frequency=AnomalyFrequency.NONE),
        ],
        output=OutputConfig(format=OutputFormat.CSV, output_dir="/tmp"),
    )

    t0 = time.perf_counter()
    gen = SARDataGenerator(cfg)
    df = gen.generate_all()
    elapsed = time.perf_counter() - t0

    total_nodes = sum(n.count for n in cfg.nodes)
    expected_rows = cfg.num_intervals * total_nodes

    print(f"\nGenerated {len(df):,} rows in {elapsed:.2f}s")
    print(f"Throughput: {len(df) / elapsed:,.0f} rows/s")

    assert len(df) == expected_rows
    assert elapsed < 300, f"Exceeded 5 min limit: {elapsed:.1f}s"


@pytest.mark.slow
def test_1m_rows_per_minute_throughput():
    """
    Başarı Kriteri: >1M data points per minute.
    """
    cfg = SimulationConfig(
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 8),
        interval_seconds=60,
        random_seed=42,
        nodes=[
            NodeConfig(type=NodeType.COMPUTE, count=15,
                       base_load=0.4,
                       anomaly_frequency=AnomalyFrequency.NONE),
            NodeConfig(type=NodeType.CEPH_STORAGE, count=5,
                       base_load=0.6,
                       anomaly_frequency=AnomalyFrequency.NONE),
        ],
        output=OutputConfig(format=OutputFormat.CSV, output_dir="/tmp"),
    )

    t0 = time.perf_counter()
    gen = SARDataGenerator(cfg)
    df = gen.generate_all()
    elapsed = time.perf_counter() - t0

    rows_per_min = len(df) / (elapsed / 60)
    print(f"\n{len(df):,} rows in {elapsed:.2f}s = {rows_per_min:,.0f} rows/minute")

    assert rows_per_min > 1_000_000, f"Below 1M/min: {rows_per_min:,.0f}"
