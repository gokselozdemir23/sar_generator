"""
Data Generation Engine - Core time-series generator with cross-metric correlations.
Supports diurnal/weekly patterns, noise injection, and multi-node batch generation.
Designed for >1M data points/minute throughput using vectorized NumPy operations.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import NodeConfig, SimulationConfig
from models.node_profiles import (
    HOSTNAME_PREFIXES,
    MetricProfile,
    NodeProfile,
    get_profile,
)

logger = logging.getLogger(__name__)

# Full list of SAR output columns (exactly as specified)
SAR_COLUMNS = [
    'DateTime', 'hostname', 'CPU',
    '%usr', '%nice', '%sys', '%iowait', '%steal', '%irq', '%soft',
    '%guest', '%gnice', '%idle',
    'proc/s', 'cswch/s', 'pswpin/s', 'pswpout/s',
    'pgpgin/s', 'pgpgout/s', 'fault/s', 'majflt/s', 'pgfree/s',
    'pgscank/s', 'pgscand/s', 'pgsteal/s', '%vmeff',
    'tps', 'rtps', 'wtps', 'bread/s', 'bwrtn/s',
    'frmpg/s', 'bufpg/s', 'campg/s',
    'kbmemfree', 'kbmemused', '%memused', 'kbbuffers', 'kbcached',
    'kbcommit', '%commit', 'kbactive', 'kbinact', 'kbdirty',
    'kbswpfree', 'kbswpused', '%swpused', 'kbswpcad', '%swpcad',
    'kbhugfree', 'kbhugused', '%hugused',
    'dentunusd', 'file-nr', 'inode-nr', 'pty-nr',
    'runq-sz', 'plist-sz', 'ldavg-1', 'ldavg-5', 'ldavg-15', 'blocked',
    'TTY', 'rcvin/s', 'xmtin/s', 'framerr/s', 'prtyerr/s', 'brk/s', 'ovrun/s',
    'rd_sec/s', 'wr_sec/s', 'avgrq-sz', 'avgqu-sz', 'await', 'svctm', '%util',
    'rxpck/s', 'txpck/s', 'rxkB/s', 'txkB/s', 'rxcmp/s', 'txcmp/s',
    'rxmcst/s', 'rxerr/s', 'txerr/s', 'coll/s', 'rxdrop/s', 'txdrop/s',
    'txcarr/s', 'rxfram/s', 'rxfifo/s', 'txfifo/s',
    'call/s', 'retrans/s', 'read/s', 'write/s', 'access/s', 'getatt/s',
    'scall/s', 'badcall/s', 'packet/s', 'udp/s', 'tcp/s',
    'hit/s', 'miss/s', 'sread/s', 'swrite/s', 'saccess/s', 'sgetatt/s',
    'kbavail', 'kbanonpg', 'kbslab', 'kbkstack', 'kbpgtbl', 'kbvmused',
    'totsck', 'tcpsck', 'udpsck', 'rawsck', 'ip-frag', 'tcp-tw',
    'total/s', 'dropd/s', 'squeezd/s', 'rx_rps/s', 'flw_lim/s', '%ifutil',
    'txmtin/s', 'rkB/s', 'wkB/s', 'areq-sz', 'aqu-sz',
]

# Columns that are integer-typed
INTEGER_COLUMNS = {
    'dentunusd', 'file-nr', 'inode-nr', 'pty-nr', 'runq-sz', 'plist-sz',
    'blocked', 'TTY', 'totsck', 'tcpsck', 'udpsck', 'rawsck', 'ip-frag', 'tcp-tw',
    'kbmemfree', 'kbmemused', 'kbbuffers', 'kbcached', 'kbcommit', 'kbactive',
    'kbinact', 'kbdirty', 'kbswpfree', 'kbswpused', 'kbswpcad',
    'kbhugfree', 'kbhugused', 'kbavail', 'kbanonpg', 'kbslab',
    'kbkstack', 'kbpgtbl', 'kbvmused',
}


class TimeSeriesGenerator:
    """
    Generates vectorized time-series data for a single node.
    Uses NumPy broadcasting for high throughput.
    """

    def __init__(
        self,
        config: SimulationConfig,
        node_config: NodeConfig,
        hostname: str,
        rng: np.random.Generator,
    ):
        self.config = config
        self.node_config = node_config
        self.hostname = hostname
        self.rng = rng
        self.profile: NodeProfile = get_profile(node_config.type)
        self.n = config.num_intervals
        self.total_mem_kb = node_config.total_memory_kb
        self._timestamps: Optional[np.ndarray] = None

        # Data quality preset → noise / AR(1) overrides
        _quality_map = {
            "clean":    {"noise": 0.01, "ar_coeff": 0.9},
            "normal":   {"noise": None, "ar_coeff": 0.7},
            "noisy":    {"noise": 0.15, "ar_coeff": 0.4},
            "degraded": {"noise": 0.10, "ar_coeff": 0.5},
        }
        _q = getattr(config, 'data_quality', 'normal')
        _qval = _q.value if hasattr(_q, 'value') else str(_q)
        _preset = _quality_map.get(_qval, {})
        self._quality_noise = _preset.get("noise")      # None ⇒ use config.noise_level
        self._quality_ar    = _preset.get("ar_coeff", 0.7)

    # Public API

    def generate(self) -> pd.DataFrame:
        """Generate full time-series DataFrame for this node."""
        logger.debug(f"Generating {self.n} intervals for {self.hostname}")

        timestamps = self._build_timestamps()
        t_frac = self._time_fractions(timestamps)   # 0..1 over total duration
        hour_of_day = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
        day_of_week = np.array([ts.weekday() for ts in timestamps])

        # Compute pattern multipliers
        diurnal_mult = self._diurnal_pattern(hour_of_day)
        weekly_mult  = self._weekly_pattern(day_of_week)
        combined_mult = diurnal_mult * weekly_mult

        # Build data dict
        data: Dict[str, np.ndarray] = {}

        #  Generate metric base values
        metric_arrays = self._generate_base_metrics(combined_mult)
        data.update(metric_arrays)

        #  Apply cross-metric correlations
        data = self._apply_correlations(data)

        #  Ensure CPU columns sum to ~100% (idle adjustment)
        data = self._normalize_cpu(data)

        #  Convert memory fractions to absolute kB values
        data = self._resolve_memory(data)

        #  Fill remaining SAR columns with defaults
        data = self._fill_defaults(data)

        # -- Build DataFrame --
        df = pd.DataFrame(data, index=range(self.n))
        df.insert(0, 'DateTime', [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps])
        df.insert(1, 'hostname', self.hostname)
        df.insert(2, 'CPU', 'all')

        # -- Ensure all SAR columns present --
        for col in SAR_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        df = df[SAR_COLUMNS]
        df = self._cast_types(df)
        return df

    # Timestamp helpers

    def _build_timestamps(self) -> List[datetime]:
        start = self.config.start_time
        interval = self.config.interval_seconds
        return [start + timedelta(seconds=i * interval) for i in range(self.n)]

    def _time_fractions(self, timestamps: List[datetime]) -> np.ndarray:
        total = self.config.total_seconds
        if total == 0:
            return np.zeros(self.n)
        return np.array([
            (ts - self.config.start_time).total_seconds() / total
            for ts in timestamps
        ])

    # Pattern generators

    def _diurnal_pattern(self, hour_of_day: np.ndarray) -> np.ndarray:
        """
        Business-hours diurnal pattern.
        Peak at ~10:00, trough at ~03:00.
        """
        if not self.config.diurnal_pattern:
            return np.ones(self.n)

        amp = self.profile.diurnal_amplitude * self.node_config.base_load
        # Sinusoidal with peak at hour 10
        phase = 2 * np.pi * (hour_of_day - 10.0) / 24.0
        pattern = 1.0 + amp * np.cos(phase)
        return np.clip(pattern, 0.3, 2.5)

    def _weekly_pattern(self, day_of_week: np.ndarray) -> np.ndarray:
        """
        Weekly pattern: weekdays higher load, weekends lower.
        Monday=0 .. Sunday=6
        """
        if not self.config.weekly_pattern:
            return np.ones(self.n)

        amp = self.profile.weekly_amplitude
        # Weekend multiplier
        is_weekend = (day_of_week >= 5).astype(float)
        pattern = 1.0 - amp * is_weekend
        return np.clip(pattern, 0.4, 1.5)

    # Base metric generation

    def _generate_base_metrics(
        self, combined_mult: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Vectorized generation of base metric values using profile distributions.
        Uses a scaled normal distribution with AR(1) autocorrelation for realism.
        """
        result: Dict[str, np.ndarray] = {}
        base_load = self.node_config.base_load
        noise = self._quality_noise if self._quality_noise is not None else self.config.noise_level

        for metric_name, mp in self.profile.metrics.items():
            scaled_mean = mp.mean * base_load * combined_mult
            scaled_std  = mp.std * (base_load + noise)

            # AR(1) process for temporal autocorrelation
            ar_coeff = self._quality_ar
            innovations = self.rng.normal(0, scaled_std, self.n)
            series = np.zeros(self.n)
            series[0] = scaled_mean[0] + innovations[0]
            for i in range(1, self.n):
                series[i] = (
                    ar_coeff * series[i - 1]
                    + (1 - ar_coeff) * scaled_mean[i]
                    + innovations[i]
                )

            # Clip to valid range
            series = np.clip(series, mp.min_val, mp.max_val)
            result[metric_name] = series

        return result

    # Correlation engine

    def _apply_correlations(
        self, data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply cross-metric dependency rules from node profile."""
        for rule in self.profile.correlations:
            src = rule.source_metric
            tgt = rule.target_metric
            if src not in data or tgt not in data:
                continue

            source_vals = data[src]
            delta = rule.coefficient * source_vals

            if rule.lag_seconds > 0:
                lag_steps = max(1, rule.lag_seconds // self.config.interval_seconds)
                delta = np.roll(delta, lag_steps)
                delta[:lag_steps] = 0.0

            data[tgt] = data[tgt] + delta

            # Clip to profile limits if available
            if tgt in self.profile.metrics:
                mp = self.profile.metrics[tgt]
                data[tgt] = np.clip(data[tgt], mp.min_val, mp.max_val)

        return data

    # CPU normalization

    def _normalize_cpu(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Ensure CPU percentages sum to ~100."""
        cpu_active_cols = ['%usr', '%nice', '%sys', '%iowait', '%steal',
                           '%irq', '%soft', '%guest', '%gnice']

        active_sum = sum(
            data.get(c, np.zeros(self.n)) for c in cpu_active_cols
        )
        # Compute idle as remainder
        idle = np.clip(100.0 - active_sum, 0.1, 99.9)
        data['%idle'] = idle

        # Scale active components if they exceed 99%
        over = active_sum > 99.0
        if np.any(over):
            scale = np.where(over, 99.0 / np.maximum(active_sum, 1e-9), 1.0)
            for c in cpu_active_cols:
                if c in data:
                    data[c] = data[c] * scale

        return data

    # Memory resolution

    def _resolve_memory(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert fraction-based memory metrics to absolute kB values.
        Adds derived memory metrics (kbavail, kbanonpg, kbslab, etc.).
        """
        total = self.total_mem_kb

        mem_used_frac = data.get('kbmemused', np.full(self.n, 0.55))
        mem_used_frac = np.clip(mem_used_frac, 0.01, 0.99)

        data['kbmemused'] = np.round(mem_used_frac * total).astype(np.int64)
        # Derive free as remainder so used + free == total exactly
        data['kbmemfree'] = (total - data['kbmemused']).astype(np.int64)
        data['%memused']  = mem_used_frac * 100.0

        buf_frac   = data.get('kbbuffers', np.full(self.n, 0.02))
        cache_frac = data.get('kbcached',  np.full(self.n, 0.15))
        dirty_frac = data.get('kbdirty',   np.full(self.n, 0.005))

        data['kbbuffers'] = np.round(buf_frac   * total).astype(np.int64)
        data['kbcached']  = np.round(cache_frac * total).astype(np.int64)
        data['kbdirty']   = np.round(dirty_frac * total).astype(np.int64)

        commit_frac = data.get('kbcommit', np.full(self.n, 0.70))
        data['kbcommit']  = np.round(commit_frac * total).astype(np.int64)
        data['%commit']   = commit_frac * 100.0

        active_frac = data.get('kbactive', np.full(self.n, 0.35))
        inact_frac  = data.get('kbinact',  np.full(self.n, 0.15))
        data['kbactive']  = np.round(active_frac * total).astype(np.int64)
        data['kbinact']   = np.round(inact_frac  * total).astype(np.int64)

        # Swap
        swap_total_kb = total // 2
        swap_used_frac = data.get('kbswpused', np.full(self.n, 0.01))
        data['kbswpused'] = np.round(swap_used_frac * swap_total_kb).astype(np.int64)
        data['kbswpfree'] = swap_total_kb - data['kbswpused']
        data['%swpused']  = swap_used_frac * 100.0
        data['kbswpcad']  = np.zeros(self.n, dtype=np.int64)
        data['%swpcad']   = np.zeros(self.n)

        # Huge pages (disabled by default in telco)
        data['kbhugfree'] = np.zeros(self.n, dtype=np.int64)
        data['kbhugused'] = np.zeros(self.n, dtype=np.int64)
        data['%hugused']  = np.zeros(self.n)

        # Derived
        data['kbavail']  = (data['kbmemfree'] + data['kbcached']).astype(np.int64)
        data['kbanonpg'] = np.round(active_frac * total * 0.6).astype(np.int64)
        data['kbslab']   = np.round(np.full(self.n, total * 0.03)).astype(np.int64)
        data['kbkstack'] = np.round(np.full(self.n, total * 0.001)).astype(np.int64)
        data['kbpgtbl']  = np.round(np.full(self.n, total * 0.002)).astype(np.int64)
        data['kbvmused'] = (data['kbmemused'] + data['kbswpused']).astype(np.int64)

        return data

    # Default filler

    def _fill_defaults(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fill SAR columns not generated by profile with realistic defaults."""
        n = self.n

        # Process/kernel counters
        defaults = {
            'pswpin/s':   (0.0, 0.5,  0.0,  500.0),
            'pswpout/s':  (0.0, 0.5,  0.0,  500.0),
            'majflt/s':   (0.5, 1.0,  0.0,  200.0),
            'pgscank/s':  (0.0, 50.0, 0.0, 50000.0),
            'pgscand/s':  (0.0, 5.0,  0.0, 10000.0),
            'pgsteal/s':  (0.0, 50.0, 0.0, 50000.0),
            '%vmeff':     (90.0, 5.0, 0.0, 100.0),
            'frmpg/s':    (0.0, 5.0, -100.0, 100.0),
            'bufpg/s':    (0.0, 2.0, -100.0, 100.0),
            'campg/s':    (0.0, 3.0, -100.0, 100.0),
        }

        for col, (mean, std, lo, hi) in defaults.items():
            if col not in data:
                vals = self.rng.normal(mean, std, n)
                data[col] = np.clip(vals, lo, hi)

        # Kernel counters (low-noise, mostly constant)
        kernel_counters = {
            'dentunusd': (50000.0, 5000.0, 0.0, 1e7),
            'file-nr':   (5000.0,  500.0,  0.0, 1e6),
            'inode-nr':  (60000.0, 3000.0, 0.0, 1e7),
            'pty-nr':    (20.0,    5.0,    0.0, 1000.0),
            'runq-sz':   (3.0,     2.0,    0.0, 256.0),
            'plist-sz':  (700.0,   50.0,   0.0, 10000.0),
            'blocked':   (0.5,     0.5,    0.0, 100.0),
        }
        for col, (mean, std, lo, hi) in kernel_counters.items():
            if col not in data:
                vals = self.rng.normal(mean, std, n)
                data[col] = np.clip(vals, lo, hi)

        # TTY (usually 0)
        if 'TTY' not in data:
            data['TTY'] = np.zeros(n)
        for col in ['rcvin/s', 'xmtin/s', 'framerr/s', 'prtyerr/s', 'brk/s', 'ovrun/s']:
            if col not in data:
                data[col] = np.zeros(n)

        # Network extras
        net_extras = {
            'rxcmp/s': (0.0, 0.0, 0.0, 1000.0),
            'txcmp/s': (0.0, 0.0, 0.0, 1000.0),
            'rxmcst/s': (5.0, 3.0, 0.0, 10000.0),
            'rxerr/s': (0.0, 0.01, 0.0, 100.0),
            'txerr/s': (0.0, 0.01, 0.0, 100.0),
            'coll/s':  (0.0, 0.01, 0.0, 100.0),
            'rxdrop/s':(0.0, 0.02, 0.0, 1000.0),
            'txdrop/s':(0.0, 0.01, 0.0, 500.0),
            'txcarr/s':(0.0, 0.0,  0.0, 100.0),
            'rxfram/s':(0.0, 0.0,  0.0, 100.0),
            'rxfifo/s':(0.0, 0.01, 0.0, 1000.0),
            'txfifo/s':(0.0, 0.01, 0.0, 1000.0),
            'txmtin/s':(0.0, 0.0,  0.0, 1000.0),
        }
        for col, (mean, std, lo, hi) in net_extras.items():
            if col not in data:
                vals = self.rng.normal(mean, std, n)
                data[col] = np.clip(vals, lo, hi)

        # NFS/RPC (typically zero unless NFS traffic)
        nfs_cols = ['call/s', 'retrans/s', 'read/s', 'write/s', 'access/s', 'getatt/s',
                    'scall/s', 'badcall/s', 'packet/s', 'udp/s', 'tcp/s',
                    'hit/s', 'miss/s', 'sread/s', 'swrite/s', 'saccess/s', 'sgetatt/s']
        for col in nfs_cols:
            if col not in data:
                data[col] = np.zeros(n)

        # Socket stats
        sock_defaults = {
            'totsck':  (500.0,  50.0,  0.0, 100000.0),
            'tcpsck':  (400.0,  40.0,  0.0,  50000.0),
            'udpsck':  (20.0,   5.0,   0.0,   1000.0),
            'rawsck':  (0.0,    0.0,   0.0,    100.0),
            'ip-frag': (0.0,    0.0,   0.0,   1000.0),
            'tcp-tw':  (50.0,  20.0,   0.0,  10000.0),
        }
        for col, (mean, std, lo, hi) in sock_defaults.items():
            if col not in data:
                vals = self.rng.normal(mean, std, n)
                data[col] = np.clip(vals, lo, hi)

        # Softnet
        for col in ['total/s', 'dropd/s', 'squeezd/s', 'rx_rps/s', 'flw_lim/s']:
            if col not in data:
                data[col] = np.zeros(n)

        # Disk aliases
        if 'rd_sec/s' not in data:
            data['rd_sec/s'] = data.get('bread/s', np.zeros(n))
        if 'wr_sec/s' not in data:
            data['wr_sec/s'] = data.get('bwrtn/s', np.zeros(n))
        if 'avgrq-sz' not in data:
            data['avgrq-sz'] = np.full(n, 64.0)
        if 'avgqu-sz' not in data:
            data['avgqu-sz'] = np.abs(self.rng.normal(0.5, 0.3, n))

        # Extended disk aliases (newer sar format)
        if 'rkB/s' not in data:
            data['rkB/s'] = data['rd_sec/s'] / 2.0    # 512-byte sectors -> kB
        if 'wkB/s' not in data:
            data['wkB/s'] = data['wr_sec/s'] / 2.0
        if 'areq-sz' not in data:
            data['areq-sz'] = data['avgrq-sz'] / 2.0   # sectors -> kB
        if 'aqu-sz' not in data:
            data['aqu-sz'] = data['avgqu-sz']

        return data

    # Type casting

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in INTEGER_COLUMNS:
            if col in df.columns:
                df[col] = df[col].round(0).astype(np.int64)

        # Round floats to 2 decimals for CSV readability
        float_cols = df.select_dtypes(include=[np.floating]).columns
        df[float_cols] = df[float_cols].round(2)
        return df


# Multi-node orchestrator

class SARDataGenerator:

    # Orchestrates generation across all configured nodes.
    # Yields DataFrames by chunk for memory efficiency.


    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def generate_all(self) -> pd.DataFrame:
        """Generate all nodes and return concatenated DataFrame."""
        from engine.anomaly import AnomalyEngine
        anomaly_engine = AnomalyEngine(self.config)

        all_frames: List[pd.DataFrame] = []

        for node_cfg in self.config.nodes:
            node_type = node_cfg.type
            prefix = HOSTNAME_PREFIXES.get(node_type, node_type)
            prefix = node_cfg.hostname_prefix or prefix

            for idx in range(node_cfg.count):
                hostname = f"{prefix}-{idx + 1:02d}"
                logger.info(f"Generating data for {hostname} ({node_type})")

                node_rng = np.random.default_rng(
                    self.rng.integers(0, 2**31)
                )
                gen = TimeSeriesGenerator(
                    config=self.config,
                    node_config=node_cfg,
                    hostname=hostname,
                    rng=node_rng,
                )
                df = gen.generate()

                # Apply anomaly scenarios
                df = anomaly_engine.apply(df, node_cfg, hostname)

                all_frames.append(df)

        if not all_frames:
            return pd.DataFrame(columns=SAR_COLUMNS)

        logger.info(f"Concatenating {len(all_frames)} node DataFrames...")
        result = pd.concat(all_frames, ignore_index=True)
        result.sort_values(['DateTime', 'hostname'], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    def generate_chunks(self, chunk_size: int = 100_000):
        """Generator that yields DataFrames in chunks."""
        from engine.anomaly import AnomalyEngine
        anomaly_engine = AnomalyEngine(self.config)
        buffer: List[pd.DataFrame] = []
        buffer_rows = 0

        for node_cfg in self.config.nodes:
            node_type = node_cfg.type
            prefix = HOSTNAME_PREFIXES.get(node_type, node_type)
            prefix = node_cfg.hostname_prefix or prefix

            for idx in range(node_cfg.count):
                hostname = f"{prefix}-{idx + 1:02d}"
                node_rng = np.random.default_rng(self.rng.integers(0, 2**31))
                gen = TimeSeriesGenerator(
                    config=self.config,
                    node_config=node_cfg,
                    hostname=hostname,
                    rng=node_rng,
                )
                df = gen.generate()
                df = anomaly_engine.apply(df, node_cfg, hostname)

                buffer.append(df)
                buffer_rows += len(df)

                if buffer_rows >= chunk_size:
                    yield pd.concat(buffer, ignore_index=True)
                    buffer = []
                    buffer_rows = 0

        if buffer:
            yield pd.concat(buffer, ignore_index=True)
