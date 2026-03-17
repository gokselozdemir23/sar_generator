"""
Property-based tests using hypothesis.
Mevcut test dosyalarına dokunmaz; tamamen bağımsız çalışır.

Çalıştırma:
    pytest tests/test_properties.py -v
    pytest tests/test_properties.py -v --hypothesis-show-statistics
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SimulationConfig, NodeConfig, NodeType,
    OutputConfig, OutputFormat, AnomalyFrequency,
)
from engine.generator import SARDataGenerator, SAR_COLUMNS


# ---------------------------------------------------------------------------
# Skip entire module if hypothesis not installed
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not HAS_HYPOTHESIS,
    reason="hypothesis not installed: pip install hypothesis",
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

if HAS_HYPOTHESIS:
    node_type_st = st.sampled_from([
        NodeType.COMPUTE, NodeType.CEPH_STORAGE,
        NodeType.CONTROL_PLANE, NodeType.NETWORK,
    ])

    @st.composite
    def sim_config_st(draw):
        """Rastgele geçerli SimulationConfig üretir."""
        days = draw(st.integers(min_value=1, max_value=2))
        interval = draw(st.sampled_from([300, 600]))
        ntype = draw(node_type_st)
        count = draw(st.integers(min_value=1, max_value=2))
        base_load = draw(st.floats(min_value=0.2, max_value=0.8))
        noise = draw(st.floats(min_value=0.01, max_value=0.15))
        seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

        start = datetime(2024, 1, 1)
        end = start + timedelta(days=days)

        return SimulationConfig(
            start_time=start,
            end_time=end,
            interval_seconds=interval,
            nodes=[NodeConfig(
                type=ntype,
                count=count,
                base_load=round(base_load, 2),
                anomaly_frequency=AnomalyFrequency.NONE,
            )],
            output=OutputConfig(
                format=OutputFormat.CSV,
                output_dir="/tmp/sar_prop_test",
            ),
            random_seed=seed,
            noise_level=round(noise, 3),
        )


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@given(cfg=sim_config_st())
@settings(
    max_examples=10,
    deadline=90_000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_all_sar_columns_present(cfg):
    """Her rastgele konfigürasyonda tüm SAR kolonları bulunmalı."""
    gen = SARDataGenerator(cfg)
    df = gen.generate_all()
    missing = [col for col in SAR_COLUMNS if col not in df.columns]
    assert missing == [], f"Missing SAR columns: {missing}"


@given(cfg=sim_config_st())
@settings(
    max_examples=10,
    deadline=90_000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_cpu_percentages_sum_near_100(cfg):
    """CPU yüzdeleri toplamda ~100 olmalı (±2 tolerans)."""
    gen = SARDataGenerator(cfg)
    df = gen.generate_all()
    cpu_cols = [
        '%usr', '%nice', '%sys', '%iowait', '%steal',
        '%irq', '%soft', '%guest', '%gnice', '%idle',
    ]
    cpu_sum = df[cpu_cols].sum(axis=1)
    deviation = np.abs(cpu_sum - 100.0)
    assert (deviation < 2.0).all(), (
        f"CPU sum deviation: min={cpu_sum.min():.2f} max={cpu_sum.max():.2f}"
    )


@given(cfg=sim_config_st())
@settings(
    max_examples=10,
    deadline=90_000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_no_negative_percentages(cfg):
    """Yüzde kolonları asla negatif olmamalı."""
    gen = SARDataGenerator(cfg)
    df = gen.generate_all()
    pct_cols = [c for c in df.columns if c.startswith('%')]
    for col in pct_cols:
        vals = pd.to_numeric(df[col], errors='coerce')
        assert (vals >= 0).all(), f"Negative values in {col}: min={vals.min()}"


@given(cfg=sim_config_st())
@settings(
    max_examples=10,
    deadline=90_000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_row_count_matches_config(cfg):
    """Üretilen satır sayısı = num_intervals × toplam node sayısı."""
    gen = SARDataGenerator(cfg)
    df = gen.generate_all()
    total_nodes = sum(n.count for n in cfg.nodes)
    expected = cfg.num_intervals * total_nodes
    assert len(df) == expected, f"Expected {expected} rows, got {len(df)}"


@given(cfg=sim_config_st())
@settings(
    max_examples=10,
    deadline=90_000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_memory_used_plus_free_equals_total(cfg):
    """kbmemused + kbmemfree ≈ total_memory_kb (±%1)."""
    gen = SARDataGenerator(cfg)
    df = gen.generate_all()
    total_mem = cfg.nodes[0].total_memory_kb
    computed_total = df['kbmemused'] + df['kbmemfree']
    delta = np.abs(computed_total - total_mem)
    assert (delta / total_mem < 0.01).all(), (
        f"Memory mismatch: max delta = {delta.max()}"
    )


@given(cfg=sim_config_st())
@settings(
    max_examples=10,
    deadline=90_000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_hostnames_match_node_config(cfg):
    """Üretilen hostname sayısı konfigürasyondaki toplam node sayısına eşit olmalı."""
    gen = SARDataGenerator(cfg)
    df = gen.generate_all()
    total_nodes = sum(n.count for n in cfg.nodes)
    actual_hosts = df['hostname'].nunique()
    assert actual_hosts == total_nodes, (
        f"Expected {total_nodes} hostnames, got {actual_hosts}"
    )


@given(cfg=sim_config_st())
@settings(
    max_examples=8,
    deadline=90_000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_datetime_column_parseable(cfg):
    """DateTime kolonları geçerli tarih olarak parse edilebilmeli."""
    gen = SARDataGenerator(cfg)
    df = gen.generate_all()
    parsed = pd.to_datetime(df['DateTime'])
    assert parsed.notna().all(), "Some DateTime values could not be parsed"


@given(cfg=sim_config_st())
@settings(
    max_examples=8,
    deadline=90_000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_no_nan_in_numeric_columns(cfg):
    """Sayısal kolonlarda NaN bulunmamalı."""
    gen = SARDataGenerator(cfg)
    df = gen.generate_all()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        nan_count = df[col].isna().sum()
        assert nan_count == 0, f"NaN found in {col}: {nan_count} values"
