"""
Tests for validation/comparison.py module.
Mevcut testlere dokunmaz; tamamen bağımsız çalışır.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SimulationConfig, NodeConfig, NodeType,
    OutputConfig, OutputFormat, AnomalyFrequency,
)
from engine.generator import SARDataGenerator
from validation.comparison import (
    SARPatternComparator,
    ComparisonReport,
    BUILTIN_REFERENCE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Kısa bir sentetik DataFrame üret."""
    cfg = SimulationConfig(
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 1, 6, 0, 0),
        interval_seconds=300,
        random_seed=42,
        nodes=[
            NodeConfig(type=NodeType.COMPUTE, count=2, base_load=0.5,
                       total_memory_kb=65_536_000,
                       anomaly_frequency=AnomalyFrequency.NONE),
            NodeConfig(type=NodeType.CEPH_STORAGE, count=1, base_load=0.6,
                       total_memory_kb=131_072_000,
                       anomaly_frequency=AnomalyFrequency.NONE),
        ],
        output=OutputConfig(format=OutputFormat.CSV, output_dir="/tmp"),
    )
    return SARDataGenerator(cfg).generate_all()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSARPatternComparator:

    def test_compare_builtin_returns_report(self, sample_df):
        comparator = SARPatternComparator(sample_df)
        report = comparator.compare_builtin()
        assert isinstance(report, ComparisonReport)
        assert len(report.results) > 0

    def test_all_results_have_required_fields(self, sample_df):
        comparator = SARPatternComparator(sample_df)
        report = comparator.compare_builtin()
        for r in report.results:
            assert hasattr(r, 'metric')
            assert hasattr(r, 'node_type')
            assert hasattr(r, 'ks_statistic')
            assert 0.0 <= r.ks_statistic <= 1.0
            assert r.mean_diff_pct >= 0.0

    def test_compute_and_ceph_both_tested(self, sample_df):
        comparator = SARPatternComparator(sample_df)
        report = comparator.compare_builtin()
        tested_types = {r.node_type for r in report.results}
        assert "compute" in tested_types
        assert "ceph_storage" in tested_types

    def test_report_to_json(self, sample_df):
        import json
        comparator = SARPatternComparator(sample_df)
        report = comparator.compare_builtin()
        j = report.to_json()
        parsed = json.loads(j)
        assert "results" in parsed
        assert "passed" in parsed

    def test_compare_with_itself(self, sample_df):
        """Veri kendisiyle karşılaştırıldığında tümü geçmeli."""
        comparator = SARPatternComparator(sample_df)
        report = comparator.compare_with(sample_df)
        assert isinstance(report, ComparisonReport)
        # Kendisiyle karşılaştırma: diff ≈ 0
        for r in report.results:
            assert r.mean_diff_pct < 1.0, (
                f"Self-comparison {r.metric}: diff={r.mean_diff_pct}%"
            )

    def test_no_hostname_column(self):
        """hostname olmadan builtin karşılaştırma boş rapor döner."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        comparator = SARPatternComparator(df)
        report = comparator.compare_builtin()
        assert len(report.results) == 0

    def test_builtin_reference_has_all_node_types(self):
        """Built-in referans 4 node tipi içermeli."""
        assert "compute" in BUILTIN_REFERENCE
        assert "ceph_storage" in BUILTIN_REFERENCE
        assert "control_plane" in BUILTIN_REFERENCE
        assert "network" in BUILTIN_REFERENCE

    def test_custom_reference_profiles(self, sample_df):
        """Özel referans profili ile karşılaştırma."""
        custom_ref = {
            "compute": {
                "%usr": (30.0, 10.0),
                "%idle": (55.0, 12.0),
            }
        }
        comparator = SARPatternComparator(sample_df)
        report = comparator.compare_builtin(reference=custom_ref)
        tested_metrics = {r.metric for r in report.results}
        assert "%usr" in tested_metrics or "%idle" in tested_metrics
