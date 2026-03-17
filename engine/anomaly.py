"""
Pattern & Anomaly Engine - Injects realistic failure scenarios into generated time-series data.
Supports: storage contention, memory pressure, network saturation, CPU steal spikes,
cascading failures, gradual degradation, and backup storm scenarios.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    AnomalyFrequency,
    AnomalySeverity,
    NodeConfig,
    ScenarioConfig,
    ScenarioType,
    SimulationConfig,
)

logger = logging.getLogger(__name__)

# Severity -> multiplier mapping
SEVERITY_MULTIPLIERS: Dict[str, float] = {
    "low":      1.5,
    "medium":   3.0,
    "high":     6.0,
    "critical": 12.0,
}

# Anomaly frequency -> probability per interval
ANOMALY_PROBS: Dict[str, float] = {
    "none":   0.0,
    "low":    0.001,
    "medium": 0.005,
    "high":   0.02,
}


def _ramp_envelope(n: int, ramp_up: int, ramp_down: int) -> np.ndarray:
    """Create a smooth ramp-up / plateau / ramp-down envelope [0..1]."""
    env = np.ones(n)
    if ramp_up > 0 and ramp_up < n:
        env[:ramp_up] = np.linspace(0.0, 1.0, ramp_up)
    if ramp_down > 0 and ramp_down < n:
        env[-ramp_down:] = np.linspace(1.0, 0.0, ramp_down)
    return env


class AnomalyEngine:
    """
    Applies all configured scenario anomalies to per-node DataFrames.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.scenarios = config.scenarios

    def apply(
        self,
        df: pd.DataFrame,
        node_cfg: NodeConfig,
        hostname: str,
    ) -> pd.DataFrame:
        """Apply all relevant anomaly scenarios to a node's DataFrame."""
        df = df.copy()

        # Apply global scenarios
        for scenario in self.scenarios:
            if self._node_targeted(scenario, node_cfg):
                df = self._apply_scenario(df, scenario, node_cfg)

        # Apply node-level random anomalies
        df = self._inject_random_anomalies(df, node_cfg)

        return df

    # ------------------------------------------------------------------
    # Scenario dispatch
    # ------------------------------------------------------------------

    def _node_targeted(self, scenario: ScenarioConfig, node_cfg: NodeConfig) -> bool:
        """Check if this scenario targets this node type."""
        if scenario.target_node_types is None:
            return True
        return node_cfg.type in scenario.target_node_types

    def _apply_scenario(
        self,
        df: pd.DataFrame,
        scenario: ScenarioConfig,
        node_cfg: NodeConfig,
    ) -> pd.DataFrame:
        """Route to the appropriate scenario handler."""
        mask = self._time_mask(df, scenario.start_time, scenario.end_time)
        if not mask.any():
            return df

        n_affected = mask.sum()
        severity_mult = SEVERITY_MULTIPLIERS.get(scenario.severity, 3.0)

        # Compute ramp envelope over scenario window
        interval = self.config.interval_seconds
        ramp_up_steps = max(0, scenario.ramp_up_minutes * 60 // interval)
        ramp_down_steps = max(0, scenario.ramp_down_minutes * 60 // interval)
        envelope = _ramp_envelope(n_affected, ramp_up_steps, ramp_down_steps)

        dispatch = {
            ScenarioType.STORAGE_CONTENTION:  self._storage_contention,
            ScenarioType.MEMORY_PRESSURE:     self._memory_pressure,
            ScenarioType.NETWORK_SATURATION:  self._network_saturation,
            ScenarioType.CPU_STEAL_SPIKE:     self._cpu_steal_spike,
            ScenarioType.CASCADING_FAILURE:   self._cascading_failure,
            ScenarioType.GRADUAL_DEGRADATION: self._gradual_degradation,
            ScenarioType.BACKUP_STORM:        self._backup_storm,
        }

        handler = dispatch.get(scenario.type)
        if handler:
            df = handler(df, mask, envelope, severity_mult, node_cfg)

        return df

    # ------------------------------------------------------------------
    # Scenario handlers
    # ------------------------------------------------------------------

    def _storage_contention(
        self,
        df: pd.DataFrame,
        mask: np.ndarray,
        envelope: np.ndarray,
        severity: float,
        node_cfg: NodeConfig,
    ) -> pd.DataFrame:
        """
        Storage contention: high await, %util, tps, bwrtn/s.
        Secondary: elevated %iowait on CPU, network txkB/s spike (replication).
        """
        logger.debug(f"Injecting storage_contention (severity={severity})")

        # Disk metrics
        _inject(df, mask, envelope, '%util',   add_pct=min(40.0 * severity / 6, 80.0))
        _inject(df, mask, envelope, 'await',   multiplier=severity * 2.0, cap=2000.0)
        _inject(df, mask, envelope, 'svctm',   multiplier=severity * 1.5, cap=500.0)
        _inject(df, mask, envelope, 'tps',     multiplier=severity * 1.8, cap=20000.0)
        _inject(df, mask, envelope, 'wtps',    multiplier=severity * 2.0, cap=10000.0)
        _inject(df, mask, envelope, 'bwrtn/s', multiplier=severity * 3.0, cap=5000000.0)
        _inject(df, mask, envelope, 'wr_sec/s',multiplier=severity * 3.0, cap=5000000.0)
        _inject(df, mask, envelope, 'avgqu-sz',multiplier=severity * 4.0, cap=128.0)

        # CPU secondary effect
        _inject(df, mask, envelope, '%iowait', add_pct=min(15.0 * severity / 6, 40.0))
        _inject(df, mask, envelope, 'ldavg-1', multiplier=severity * 1.5, cap=256.0)

        # Network secondary: replication spike
        _inject(df, mask, envelope, 'txkB/s',  multiplier=severity * 1.4, cap=1000000.0)
        _inject(df, mask, envelope, 'txpck/s', multiplier=severity * 1.3, cap=10000000.0)

        # Adjust %idle
        df = _rebalance_cpu(df, mask)
        return df

    def _memory_pressure(
        self,
        df: pd.DataFrame,
        mask: np.ndarray,
        envelope: np.ndarray,
        severity: float,
        node_cfg: NodeConfig,
    ) -> pd.DataFrame:
        """
        Memory pressure: kbmemused increases, page scanning starts,
        swap usage grows, fault/s spikes, %sys increases.
        """
        logger.debug(f"Injecting memory_pressure (severity={severity})")
        total_mem = node_cfg.total_memory_kb
        pressure_kb = int(total_mem * 0.08 * (severity / 6.0))

        # Memory increase
        _inject(df, mask, envelope, 'kbmemused', add_abs=pressure_kb, cap=total_mem * 0.99)
        _inject(df, mask, envelope, '%memused',  add_pct=min(20.0 * severity / 6, 40.0))
        _inject(df, mask, envelope, 'kbdirty',   multiplier=severity * 2.0,
                cap=int(total_mem * 0.2))

        # Page scanning
        _inject(df, mask, envelope, 'pgscand/s', multiplier=severity * 5.0, cap=50000.0)
        _inject(df, mask, envelope, 'pgscank/s', multiplier=severity * 3.0, cap=200000.0)
        _inject(df, mask, envelope, 'pgsteal/s', multiplier=severity * 4.0, cap=200000.0)
        _inject(df, mask, envelope, 'fault/s',   multiplier=severity * 2.0, cap=500000.0)
        _inject(df, mask, envelope, 'majflt/s',  multiplier=severity * 8.0, cap=5000.0)

        # Swap
        if severity >= 4.0:
            swap_total = total_mem // 2
            swap_add = int(swap_total * 0.05 * (severity / 6.0))
            _inject(df, mask, envelope, 'kbswpused', add_abs=swap_add, cap=swap_total)
            _inject(df, mask, envelope, '%swpused',  add_pct=min(15.0 * severity / 6, 80.0))
            _inject(df, mask, envelope, 'pswpout/s', add_abs=10.0 * severity, cap=500.0)

        # CPU secondary: high %sys due to page faults
        _inject(df, mask, envelope, '%sys',    add_pct=min(8.0 * severity / 6, 25.0))
        _inject(df, mask, envelope, 'cswch/s', multiplier=severity * 1.5, cap=500000.0)

        # vmeff drops
        if '%vmeff' in df.columns:
            df.loc[mask, '%vmeff'] = np.clip(
                df.loc[mask, '%vmeff'].values - envelope * (severity * 5.0), 0.0, 100.0
            )

        df = _rebalance_cpu(df, mask)
        return df

    def _network_saturation(
        self,
        df: pd.DataFrame,
        mask: np.ndarray,
        envelope: np.ndarray,
        severity: float,
        node_cfg: NodeConfig,
    ) -> pd.DataFrame:
        """
        Network saturation: rxpck/s, txpck/s spike, %ifutil near 100,
        packet drops appear, %soft increases.
        """
        logger.debug(f"Injecting network_saturation (severity={severity})")

        _inject(df, mask, envelope, 'rxpck/s', multiplier=severity * 2.0, cap=10000000.0)
        _inject(df, mask, envelope, 'txpck/s', multiplier=severity * 2.0, cap=10000000.0)
        _inject(df, mask, envelope, 'rxkB/s',  multiplier=severity * 1.8, cap=1000000.0)
        _inject(df, mask, envelope, 'txkB/s',  multiplier=severity * 1.8, cap=1000000.0)
        _inject(df, mask, envelope, '%ifutil', add_pct=min(25.0 * severity / 6, 95.0))

        # Drops appear as saturation increases
        if severity >= 3.0:
            _inject(df, mask, envelope, 'rxdrop/s', add_abs=50.0 * (severity - 2.0), cap=50000.0)
            _inject(df, mask, envelope, 'txdrop/s', add_abs=20.0 * (severity - 2.0), cap=20000.0)
            _inject(df, mask, envelope, 'dropd/s',  add_abs=50.0 * (severity - 2.0), cap=50000.0)

        # CPU softirq increase
        _inject(df, mask, envelope, '%soft', add_pct=min(10.0 * severity / 6, 30.0))
        _inject(df, mask, envelope, 'total/s', multiplier=severity * 2.0, cap=10000000.0)

        df = _rebalance_cpu(df, mask)
        return df

    def _cpu_steal_spike(
        self,
        df: pd.DataFrame,
        mask: np.ndarray,
        envelope: np.ndarray,
        severity: float,
        node_cfg: NodeConfig,
    ) -> pd.DataFrame:
        """
        CPU steal spike: %steal increases (hypervisor contention),
        secondary: %iowait increases, application retries -> storage await up.
        """
        logger.debug(f"Injecting cpu_steal_spike (severity={severity})")

        _inject(df, mask, envelope, '%steal',  add_pct=min(20.0 * severity / 6, 50.0))
        _inject(df, mask, envelope, '%iowait', add_pct=min(8.0  * severity / 6, 25.0))
        _inject(df, mask, envelope, 'ldavg-1', multiplier=severity * 1.3, cap=256.0)

        # Secondary: app retries -> storage
        _inject(df, mask, envelope, 'await', multiplier=severity * 1.5, cap=2000.0)

        df = _rebalance_cpu(df, mask)
        return df

    def _cascading_failure(
        self,
        df: pd.DataFrame,
        mask: np.ndarray,
        envelope: np.ndarray,
        severity: float,
        node_cfg: NodeConfig,
    ) -> pd.DataFrame:
        """
        Cascading failure: multiple subsystems degrade progressively.
        Escalating pattern: starts with CPU steal, then memory, then storage.
        """
        logger.debug(f"Injecting cascading_failure (severity={severity})")
        n = mask.sum()

        # Phase 1: CPU steal (first third)
        phase1_end = n // 3
        ph1 = np.zeros(n, dtype=bool)
        ph1[:phase1_end] = True

        _inject_submask(df, mask, ph1, '%steal',  add_pct=min(15.0 * severity / 6, 40.0))
        _inject_submask(df, mask, ph1, '%iowait', add_pct=min(5.0  * severity / 6, 20.0))

        # Phase 2: Memory pressure (second third)
        ph2 = np.zeros(n, dtype=bool)
        ph2[phase1_end: 2 * n // 3] = True
        total_mem = node_cfg.total_memory_kb
        _inject_submask(df, mask, ph2, 'fault/s', multiplier=severity * 3.0, cap=500000.0)
        _inject_submask(df, mask, ph2, 'pgscand/s', multiplier=severity * 4.0, cap=50000.0)
        _inject_submask(df, mask, ph2, '%sys', add_pct=min(10.0 * severity / 6, 30.0))

        # Phase 3: Storage breakdown (final third)
        ph3 = np.zeros(n, dtype=bool)
        ph3[2 * n // 3:] = True
        _inject_submask(df, mask, ph3, 'await',   multiplier=severity * 5.0, cap=2000.0)
        _inject_submask(df, mask, ph3, '%util',   add_pct=min(50.0 * severity / 6, 99.0))
        _inject_submask(df, mask, ph3, 'ldavg-1', multiplier=severity * 3.0, cap=256.0)

        df = _rebalance_cpu(df, mask)
        return df

    def _gradual_degradation(
        self,
        df: pd.DataFrame,
        mask: np.ndarray,
        envelope: np.ndarray,
        severity: float,
        node_cfg: NodeConfig,
    ) -> pd.DataFrame:
        """
        Gradual degradation: metrics slowly worsen over the scenario window.
        Models slow leak / memory/disk accumulation over time.
        """
        logger.debug(f"Injecting gradual_degradation (severity={severity})")
        n = mask.sum()
        # Linear ramp from 1x to severity*x
        ramp = np.linspace(1.0, severity, n)

        for col, add in [('%iowait', 3.0), ('%sys', 2.0), ('await', 5.0),
                         ('ldavg-1', 1.0), ('fault/s', 200.0)]:
            if col in df.columns:
                idx = np.where(mask)[0]
                df.loc[idx, col] = np.clip(
                    df.loc[idx, col].values + ramp * add,
                    0.0,
                    df[col].max() * 10 if df[col].max() > 0 else 1000.0,
                )

        df = _rebalance_cpu(df, mask)
        return df

    def _backup_storm(
        self,
        df: pd.DataFrame,
        mask: np.ndarray,
        envelope: np.ndarray,
        severity: float,
        node_cfg: NodeConfig,
    ) -> pd.DataFrame:
        """
        Backup storm: massive read I/O, high network traffic, CPU %iowait spike.
        Models nightly backup window behaviour.
        """
        logger.debug(f"Injecting backup_storm (severity={severity})")

        _inject(df, mask, envelope, 'rtps',    multiplier=severity * 4.0, cap=20000.0)
        _inject(df, mask, envelope, 'bread/s', multiplier=severity * 5.0, cap=5000000.0)
        _inject(df, mask, envelope, 'rd_sec/s',multiplier=severity * 5.0, cap=5000000.0)
        _inject(df, mask, envelope, 'rkB/s',   multiplier=severity * 5.0, cap=2500000.0)
        _inject(df, mask, envelope, '%iowait', add_pct=min(20.0 * severity / 6, 50.0))
        _inject(df, mask, envelope, '%util',   add_pct=min(30.0 * severity / 6, 80.0))
        _inject(df, mask, envelope, 'await',   multiplier=severity * 2.0, cap=2000.0)

        # Network spike (backup target)
        _inject(df, mask, envelope, 'txkB/s',  multiplier=severity * 3.0, cap=1000000.0)
        _inject(df, mask, envelope, 'txpck/s', multiplier=severity * 2.5, cap=10000000.0)
        _inject(df, mask, envelope, '%ifutil', add_pct=min(30.0 * severity / 6, 90.0))

        df = _rebalance_cpu(df, mask)
        return df

    # ------------------------------------------------------------------
    # Random micro-anomalies
    # ------------------------------------------------------------------

    def _inject_random_anomalies(
        self,
        df: pd.DataFrame,
        node_cfg: NodeConfig,
    ) -> pd.DataFrame:
        """
        Inject short-duration random spikes based on anomaly_frequency config.
        """
        prob = ANOMALY_PROBS.get(node_cfg.anomaly_frequency, 0.0)
        if prob == 0.0:
            return df

        n = len(df)
        rng = np.random.default_rng()
        spike_mask = rng.random(n) < prob

        if not spike_mask.any():
            return df

        # Random mini spike: boost %iowait and await briefly
        spike_idx = np.where(spike_mask)[0]
        for idx in spike_idx:
            # Spike window: 1-3 intervals
            end = min(idx + rng.integers(1, 4), n)
            window = slice(idx, end)
            mult = rng.uniform(2.0, 5.0)

            for col in ['%iowait', 'await', 'tps']:
                if col in df.columns:
                    df.loc[df.index[window], col] = np.clip(
                        df.loc[df.index[window], col].values * mult,
                        0.0, df[col].max() * 10 + 1,
                    )

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _time_mask(
        self,
        df: pd.DataFrame,
        start: datetime,
        end: Optional[datetime],
    ) -> np.ndarray:
        """Return boolean mask for rows within [start, end)."""
        datetimes = pd.to_datetime(df['DateTime'])
        mask = (datetimes >= pd.Timestamp(start))
        if end is not None:
            mask = mask & (datetimes < pd.Timestamp(end))
        return mask.values


# ---------------------------------------------------------------------------
# Injection helpers (module-level for performance)
# ---------------------------------------------------------------------------

def _inject(
    df: pd.DataFrame,
    mask: np.ndarray,
    envelope: np.ndarray,
    col: str,
    multiplier: float = 1.0,
    add_pct: float = 0.0,
    add_abs: float = 0.0,
    cap: Optional[float] = None,
) -> None:
    """In-place metric injection with envelope-shaped intensity."""
    if col not in df.columns:
        return

    idx = np.where(mask)[0]
    vals = df.loc[df.index[idx], col].values.astype(float)

    if multiplier != 1.0:
        vals = vals * (1.0 + (multiplier - 1.0) * envelope)
    if add_pct != 0.0:
        vals = vals + add_pct * envelope
    if add_abs != 0.0:
        vals = vals + add_abs * envelope

    if cap is not None:
        vals = np.clip(vals, 0.0, cap)

    df.loc[df.index[idx], col] = vals


def _inject_submask(
    df: pd.DataFrame,
    outer_mask: np.ndarray,
    sub_mask: np.ndarray,  # boolean array over the outer masked subset
    col: str,
    multiplier: float = 1.0,
    add_pct: float = 0.0,
    add_abs: float = 0.0,
    cap: Optional[float] = None,
) -> None:
    """Inject into a sub-window of a mask."""
    if col not in df.columns:
        return

    outer_idx = np.where(outer_mask)[0]
    inner_idx = outer_idx[sub_mask]
    if len(inner_idx) == 0:
        return

    vals = df.loc[df.index[inner_idx], col].values.astype(float)

    if multiplier != 1.0:
        vals = vals * multiplier
    if add_pct != 0.0:
        vals = vals + add_pct
    if add_abs != 0.0:
        vals = vals + add_abs

    if cap is not None:
        vals = np.clip(vals, 0.0, cap)

    df.loc[df.index[inner_idx], col] = vals


def _rebalance_cpu(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """
    After injecting CPU anomalies, recalculate %idle so columns sum to ~100.
    """
    cpu_active_cols = ['%usr', '%nice', '%sys', '%iowait', '%steal',
                       '%irq', '%soft', '%guest', '%gnice']

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return df

    active_sum = sum(
        df.loc[df.index[idx], c].values
        for c in cpu_active_cols
        if c in df.columns
    )

    # Scale down if over budget
    over = active_sum > 99.0
    if np.any(over):
        scale = np.where(over, 99.0 / np.maximum(active_sum, 1e-9), 1.0)
        for c in cpu_active_cols:
            if c in df.columns:
                df.loc[df.index[idx], c] = (
                    df.loc[df.index[idx], c].values * scale
                )
        active_sum = np.minimum(active_sum, 99.0)

    df.loc[df.index[idx], '%idle'] = np.clip(100.0 - active_sum, 0.1, 99.9)
    return df
