"""
Node Profiles - Base load characteristics and metric coefficients for each node type.
Defines statistical distributions and cross-metric correlation weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class MetricProfile:
    """Statistical profile for a single metric: (base_mean, base_std, min_val, max_val)."""
    mean: float
    std: float
    min_val: float
    max_val: float
    unit: str = ""


@dataclass
class CorrelationRule:
    """Cross-metric dependency rule."""
    source_metric: str
    target_metric: str
    coefficient: float       # How much 1-unit change in source affects target
    lag_seconds: int = 0     # Propagation delay
    description: str = ""


@dataclass
class NodeProfile:
    name: str
    node_type: str
    # Metric base profiles keyed by column name
    metrics: Dict[str, MetricProfile] = field(default_factory=dict)
    # Correlation rules between metrics
    correlations: list = field(default_factory=list)
    # Diurnal pattern amplitude (0=flat, 1=full swing)
    diurnal_amplitude: float = 0.3
    # Weekly pattern amplitude
    weekly_amplitude: float = 0.15
    # Typical CPU core utilization at base load
    cpu_base_util: float = 0.35
    # Memory baseline fraction
    mem_base_fraction: float = 0.55


# ---------------------------------------------------------------------------
# Profile Definitions
# ---------------------------------------------------------------------------

def build_compute_profile() -> NodeProfile:
    """
    Compute Node Profile.
    Critical: %steal, %iowait, kbmemused, txkB/s
    Correlation: High memory -> high page fault -> high %sys
    """
    p = NodeProfile(
        name="Compute Node",
        node_type="compute",
        diurnal_amplitude=0.35,
        weekly_amplitude=0.20,
        cpu_base_util=0.35,
        mem_base_fraction=0.55,
    )

    p.metrics = {
        # CPU metrics
        "%usr":       MetricProfile(30.0, 6.0,   0.0,  99.0, "%"),
        "%nice":      MetricProfile(0.1,  0.05,  0.0,   5.0, "%"),
        "%sys":       MetricProfile(5.0,  1.5,   0.0,  30.0, "%"),
        "%iowait":    MetricProfile(2.5,  1.5,   0.0,  40.0, "%"),
        "%steal":     MetricProfile(0.5,  0.8,   0.0,  25.0, "%"),   # KEY metric
        "%irq":       MetricProfile(0.1,  0.05,  0.0,   5.0, "%"),
        "%soft":      MetricProfile(0.8,  0.3,   0.0,  10.0, "%"),
        "%guest":     MetricProfile(5.0,  2.0,   0.0,  50.0, "%"),   # VMs running
        "%gnice":     MetricProfile(0.0,  0.01,  0.0,   5.0, "%"),
        "%idle":      MetricProfile(55.0, 8.0,   0.0,  99.9, "%"),
        # Process/context
        "proc/s":     MetricProfile(120.0, 30.0, 0.0, 2000.0, "/s"),
        "cswch/s":    MetricProfile(5000.0, 1500.0, 0.0, 100000.0, "/s"),
        # Swap
        "pswpin/s":   MetricProfile(0.0,  0.5,   0.0,  500.0, "/s"),
        "pswpout/s":  MetricProfile(0.0,  0.5,   0.0,  500.0, "/s"),
        # Paging
        "pgpgin/s":   MetricProfile(100.0, 80.0,  0.0, 50000.0, "/s"),
        "pgpgout/s":  MetricProfile(200.0, 100.0, 0.0, 100000.0, "/s"),
        "fault/s":    MetricProfile(2000.0, 500.0, 0.0, 100000.0, "/s"),
        "majflt/s":   MetricProfile(0.5,  1.0,   0.0,  500.0, "/s"),
        "pgfree/s":   MetricProfile(3000.0, 800.0, 0.0, 200000.0, "/s"),
        "pgscank/s":  MetricProfile(0.0,  50.0,  0.0,  50000.0, "/s"),
        "pgscand/s":  MetricProfile(0.0,  5.0,   0.0,  10000.0, "/s"),
        "pgsteal/s":  MetricProfile(0.0,  50.0,  0.0,  50000.0, "/s"),
        "%vmeff":     MetricProfile(90.0, 10.0,  0.0,  100.0, "%"),
        # Block I/O
        "tps":        MetricProfile(150.0, 50.0,  0.0, 5000.0, "/s"),
        "rtps":       MetricProfile(80.0,  30.0,  0.0, 3000.0, "/s"),
        "wtps":       MetricProfile(70.0,  30.0,  0.0, 3000.0, "/s"),
        "bread/s":    MetricProfile(5000.0, 2000.0, 0.0, 500000.0, "sectors/s"),
        "bwrtn/s":    MetricProfile(8000.0, 3000.0, 0.0, 500000.0, "sectors/s"),
        # Buffer/cache pages
        "frmpg/s":    MetricProfile(0.0,  5.0,  -100.0, 100.0, "pages/s"),
        "bufpg/s":    MetricProfile(0.0,  2.0,  -100.0, 100.0, "pages/s"),
        "campg/s":    MetricProfile(0.0,  3.0,  -100.0, 100.0, "pages/s"),
        # Memory - absolute values filled by generator based on node config
        "kbmemfree":  MetricProfile(0.45, 0.05, 0.01, 0.99, "fraction"),  # stored as fraction
        "kbmemused":  MetricProfile(0.55, 0.05, 0.01, 0.99, "fraction"),  # KEY metric
        "%memused":   MetricProfile(55.0, 5.0,  1.0,  99.9, "%"),
        "kbbuffers":  MetricProfile(0.02, 0.005, 0.0, 0.1,  "fraction"),
        "kbcached":   MetricProfile(0.15, 0.02,  0.0, 0.5,  "fraction"),
        "kbcommit":   MetricProfile(0.70, 0.10,  0.1, 2.0,  "fraction"),
        "%commit":    MetricProfile(70.0, 10.0,  1.0, 200.0, "%"),
        "kbactive":   MetricProfile(0.35, 0.05,  0.0, 0.99, "fraction"),
        "kbinact":    MetricProfile(0.15, 0.03,  0.0, 0.5,  "fraction"),
        "kbdirty":    MetricProfile(0.005,0.002, 0.0, 0.1,  "fraction"),
        "kbswpfree":  MetricProfile(0.99, 0.01,  0.0, 1.0,  "fraction"),
        "kbswpused":  MetricProfile(0.01, 0.01,  0.0, 1.0,  "fraction"),
        "%swpused":   MetricProfile(1.0,  2.0,   0.0, 100.0, "%"),
        # Network - key metric txkB/s
        "rxpck/s":    MetricProfile(8000.0,  2000.0,  0.0, 1000000.0, "/s"),
        "txpck/s":    MetricProfile(9000.0,  2500.0,  0.0, 1000000.0, "/s"),
        "rxkB/s":     MetricProfile(600.0,   200.0,   0.0, 100000.0,  "kB/s"),
        "txkB/s":     MetricProfile(1200.0,  400.0,   0.0, 100000.0,  "kB/s"),   # KEY metric
        "%ifutil":    MetricProfile(10.0,    5.0,     0.0, 100.0,     "%"),
        # Load average
        "ldavg-1":    MetricProfile(4.0,  2.0,  0.0, 256.0),
        "ldavg-5":    MetricProfile(3.8,  1.8,  0.0, 256.0),
        "ldavg-15":   MetricProfile(3.5,  1.5,  0.0, 256.0),
        # Disk extended
        "await":      MetricProfile(3.0,  1.5,  0.0, 5000.0, "ms"),
        "svctm":      MetricProfile(0.8,  0.3,  0.0,  500.0, "ms"),
        "%util":      MetricProfile(25.0, 10.0, 0.0,  100.0, "%"),
        "rd_sec/s":   MetricProfile(5000.0, 2000.0, 0.0, 500000.0, "sectors/s"),
        "wr_sec/s":   MetricProfile(8000.0, 3000.0, 0.0, 500000.0, "sectors/s"),
        "avgrq-sz":   MetricProfile(64.0,  16.0,   1.0, 512.0, "sectors"),
        "avgqu-sz":   MetricProfile(0.5,   0.3,    0.0,  64.0),
    }

    p.correlations = [
        # High memory usage -> high fault/s
        CorrelationRule("kbmemused", "fault/s", coefficient=500.0,
                        description="High mem -> page faults"),
        # High fault/s -> high %sys
        CorrelationRule("fault/s", "%sys", coefficient=0.0001,
                        description="Page faults -> CPU sys"),
        # High %steal -> high %iowait
        CorrelationRule("%steal", "%iowait", coefficient=0.4,
                        description="Steal -> IO wait"),
        # High bwrtn/s -> txkB/s spike (storage replication)
        CorrelationRule("bwrtn/s", "txkB/s", coefficient=0.05,
                        description="Storage writes -> network TX"),
    ]
    return p


def build_ceph_profile() -> NodeProfile:
    """
    CEPH Storage Node Profile.
    Critical: await, svctm, %util, tps, bread/s, bwrtn/s
    Correlation: Storage replication -> high bwrtn/s -> network txkB/s spike
    """
    p = NodeProfile(
        name="CEPH Storage Node",
        node_type="ceph_storage",
        diurnal_amplitude=0.25,
        weekly_amplitude=0.10,
        cpu_base_util=0.25,
        mem_base_fraction=0.60,
    )

    p.metrics = {
        "%usr":       MetricProfile(15.0, 4.0,   0.0,  60.0, "%"),
        "%nice":      MetricProfile(0.0,  0.01,  0.0,   1.0, "%"),
        "%sys":       MetricProfile(8.0,  2.0,   0.0,  40.0, "%"),
        "%iowait":    MetricProfile(5.0,  3.0,   0.0,  60.0, "%"),
        "%steal":     MetricProfile(0.1,  0.1,   0.0,   5.0, "%"),
        "%soft":      MetricProfile(2.0,  0.5,   0.0,  15.0, "%"),
        "%idle":      MetricProfile(70.0, 8.0,   0.0,  99.9, "%"),
        # Disk - KEY metrics
        "tps":        MetricProfile(800.0,  200.0,  0.0, 20000.0, "/s"),
        "rtps":       MetricProfile(300.0,  100.0,  0.0, 10000.0, "/s"),
        "wtps":       MetricProfile(500.0,  150.0,  0.0, 10000.0, "/s"),
        "bread/s":    MetricProfile(80000.0, 30000.0, 0.0, 5000000.0, "sectors/s"),
        "bwrtn/s":    MetricProfile(120000.0,50000.0, 0.0, 5000000.0, "sectors/s"),
        "await":      MetricProfile(5.0,  3.0,  0.0, 2000.0, "ms"),   # KEY
        "svctm":      MetricProfile(1.0,  0.5,  0.0,  500.0, "ms"),   # KEY
        "%util":      MetricProfile(45.0, 15.0, 0.0,  100.0, "%"),    # KEY
        "rd_sec/s":   MetricProfile(80000.0, 30000.0, 0.0, 5000000.0, "sectors/s"),
        "wr_sec/s":   MetricProfile(120000.0,50000.0, 0.0, 5000000.0, "sectors/s"),
        "avgrq-sz":   MetricProfile(128.0, 32.0,  1.0, 1024.0, "sectors"),
        "avgqu-sz":   MetricProfile(2.0,   1.0,   0.0,  128.0),
        # Network
        "rxpck/s":    MetricProfile(20000.0, 5000.0,  0.0, 1000000.0, "/s"),
        "txpck/s":    MetricProfile(22000.0, 6000.0,  0.0, 1000000.0, "/s"),
        "rxkB/s":     MetricProfile(15000.0, 5000.0,  0.0, 1000000.0, "kB/s"),
        "txkB/s":     MetricProfile(18000.0, 6000.0,  0.0, 1000000.0, "kB/s"),
        "%ifutil":    MetricProfile(35.0,   10.0,    0.0, 100.0,     "%"),
        # Memory
        "kbmemfree":  MetricProfile(0.40, 0.05, 0.01, 0.99, "fraction"),
        "kbmemused":  MetricProfile(0.60, 0.05, 0.01, 0.99, "fraction"),
        "%memused":   MetricProfile(60.0, 5.0,  1.0,  99.9, "%"),
        "kbbuffers":  MetricProfile(0.03, 0.01, 0.0,  0.2,  "fraction"),
        "kbcached":   MetricProfile(0.25, 0.05, 0.0,  0.6,  "fraction"),
        "kbdirty":    MetricProfile(0.02, 0.01, 0.0,  0.2,  "fraction"),
        # Load
        "ldavg-1":    MetricProfile(6.0,  3.0,  0.0, 256.0),
        "ldavg-5":    MetricProfile(5.8,  2.5,  0.0, 256.0),
        "ldavg-15":   MetricProfile(5.5,  2.0,  0.0, 256.0),
        # Paging
        "fault/s":    MetricProfile(500.0, 200.0, 0.0, 50000.0, "/s"),
        "pgpgout/s":  MetricProfile(5000.0, 2000.0, 0.0, 500000.0, "/s"),
        "pgpgin/s":   MetricProfile(2000.0, 1000.0, 0.0, 200000.0, "/s"),
        "proc/s":     MetricProfile(80.0,  20.0, 0.0, 1000.0, "/s"),
        "cswch/s":    MetricProfile(8000.0, 2000.0, 0.0, 200000.0, "/s"),
    }

    p.correlations = [
        # High bwrtn/s -> txkB/s spike (replication traffic)
        CorrelationRule("bwrtn/s", "txkB/s", coefficient=0.12,
                        description="Disk writes -> network TX (replication)"),
        # High %util -> high await
        CorrelationRule("%util", "await", coefficient=0.3,
                        description="Disk util -> await latency"),
        # High tps -> high %util
        CorrelationRule("tps", "%util", coefficient=0.004,
                        description="IOPS -> disk utilization"),
        # High await -> high %iowait
        CorrelationRule("await", "%iowait", coefficient=0.02,
                        description="Disk latency -> CPU iowait"),
    ]
    return p


def build_control_plane_profile() -> NodeProfile:
    """
    Control Plane Node Profile.
    Critical: CPU, ldavg-1, %iowait
    """
    p = NodeProfile(
        name="Control Plane Node",
        node_type="control_plane",
        diurnal_amplitude=0.20,
        weekly_amplitude=0.10,
        cpu_base_util=0.28,
        mem_base_fraction=0.50,
    )

    p.metrics = {
        "%usr":       MetricProfile(20.0, 5.0,  0.0, 80.0, "%"),
        "%nice":      MetricProfile(0.0,  0.01, 0.0,  2.0, "%"),
        "%sys":       MetricProfile(8.0,  2.0,  0.0, 40.0, "%"),
        "%iowait":    MetricProfile(3.0,  2.0,  0.0, 50.0, "%"),   # KEY
        "%steal":     MetricProfile(0.1,  0.1,  0.0,  5.0, "%"),
        "%soft":      MetricProfile(1.5,  0.5,  0.0, 15.0, "%"),
        "%idle":      MetricProfile(67.0, 7.0,  0.0, 99.9, "%"),
        # Load - KEY metrics
        "ldavg-1":    MetricProfile(3.0,  1.5,  0.0, 64.0),   # KEY
        "ldavg-5":    MetricProfile(2.8,  1.3,  0.0, 64.0),
        "ldavg-15":   MetricProfile(2.6,  1.2,  0.0, 64.0),
        # Memory
        "kbmemfree":  MetricProfile(0.50, 0.05, 0.01, 0.99, "fraction"),
        "kbmemused":  MetricProfile(0.50, 0.05, 0.01, 0.99, "fraction"),
        "%memused":   MetricProfile(50.0, 5.0,  1.0,  99.9, "%"),
        "kbbuffers":  MetricProfile(0.02, 0.005, 0.0, 0.1,  "fraction"),
        "kbcached":   MetricProfile(0.12, 0.02,  0.0, 0.4,  "fraction"),
        "kbdirty":    MetricProfile(0.003,0.001, 0.0, 0.05, "fraction"),
        # Disk
        "tps":        MetricProfile(100.0, 30.0, 0.0, 2000.0, "/s"),
        "rtps":       MetricProfile(50.0,  20.0, 0.0, 1000.0, "/s"),
        "wtps":       MetricProfile(50.0,  20.0, 0.0, 1000.0, "/s"),
        "await":      MetricProfile(4.0,   2.0,  0.0, 1000.0, "ms"),
        "svctm":      MetricProfile(0.6,   0.2,  0.0,  100.0, "ms"),
        "%util":      MetricProfile(15.0,  8.0,  0.0,  100.0, "%"),
        "bread/s":    MetricProfile(2000.0, 1000.0, 0.0, 100000.0, "sectors/s"),
        "bwrtn/s":    MetricProfile(3000.0, 1500.0, 0.0, 100000.0, "sectors/s"),
        # Network
        "rxpck/s":    MetricProfile(3000.0, 800.0,  0.0, 500000.0, "/s"),
        "txpck/s":    MetricProfile(3200.0, 900.0,  0.0, 500000.0, "/s"),
        "rxkB/s":     MetricProfile(200.0,  60.0,   0.0,  50000.0, "kB/s"),
        "txkB/s":     MetricProfile(250.0,  80.0,   0.0,  50000.0, "kB/s"),
        "%ifutil":    MetricProfile(5.0,    2.0,    0.0,  100.0,   "%"),
        # Process
        "proc/s":     MetricProfile(60.0,  15.0, 0.0, 500.0, "/s"),
        "cswch/s":    MetricProfile(4000.0, 1000.0, 0.0, 100000.0, "/s"),
        "fault/s":    MetricProfile(800.0, 200.0, 0.0, 50000.0, "/s"),
        "pgpgin/s":   MetricProfile(500.0, 200.0, 0.0, 50000.0, "/s"),
        "pgpgout/s":  MetricProfile(800.0, 300.0, 0.0, 100000.0, "/s"),
    }

    p.correlations = [
        CorrelationRule("ldavg-1", "%iowait", coefficient=0.5,
                        description="High load -> IO wait"),
        CorrelationRule("tps", "ldavg-1", coefficient=0.005,
                        description="Disk IOPS -> load average"),
    ]
    return p


def build_network_profile() -> NodeProfile:
    """
    Network Node Profile.
    Critical: rxpck/s, txpck/s, %soft, %ifutil
    """
    p = NodeProfile(
        name="Network Node",
        node_type="network",
        diurnal_amplitude=0.30,
        weekly_amplitude=0.18,
        cpu_base_util=0.20,
        mem_base_fraction=0.40,
    )

    p.metrics = {
        "%usr":       MetricProfile(8.0,  3.0,  0.0, 40.0, "%"),
        "%nice":      MetricProfile(0.0,  0.01, 0.0,  1.0, "%"),
        "%sys":       MetricProfile(5.0,  2.0,  0.0, 30.0, "%"),
        "%iowait":    MetricProfile(0.5,  0.3,  0.0, 10.0, "%"),
        "%steal":     MetricProfile(0.1,  0.1,  0.0,  5.0, "%"),
        "%irq":       MetricProfile(0.2,  0.1,  0.0,  5.0, "%"),
        "%soft":      MetricProfile(5.0,  2.0,  0.0, 40.0, "%"),   # KEY
        "%idle":      MetricProfile(80.0, 6.0,  0.0, 99.9, "%"),
        # Network - KEY metrics
        "rxpck/s":    MetricProfile(100000.0, 30000.0,  0.0, 10000000.0, "/s"),  # KEY
        "txpck/s":    MetricProfile(110000.0, 35000.0,  0.0, 10000000.0, "/s"),  # KEY
        "rxkB/s":     MetricProfile(80000.0,  25000.0,  0.0, 1000000.0,  "kB/s"),
        "txkB/s":     MetricProfile(90000.0,  28000.0,  0.0, 1000000.0,  "kB/s"),
        "rxcmp/s":    MetricProfile(0.0,  0.0,  0.0, 1000.0, "/s"),
        "txcmp/s":    MetricProfile(0.0,  0.0,  0.0, 1000.0, "/s"),
        "rxmcst/s":   MetricProfile(10.0, 5.0,  0.0, 10000.0, "/s"),
        "rxerr/s":    MetricProfile(0.0,  0.01, 0.0,   100.0, "/s"),
        "txerr/s":    MetricProfile(0.0,  0.01, 0.0,   100.0, "/s"),
        "rxdrop/s":   MetricProfile(0.0,  0.02, 0.0,  1000.0, "/s"),
        "txdrop/s":   MetricProfile(0.0,  0.01, 0.0,   500.0, "/s"),
        "%ifutil":    MetricProfile(70.0, 10.0, 0.0,  100.0, "%"),    # KEY
        # Softnet stats
        "total/s":    MetricProfile(110000.0, 35000.0, 0.0, 10000000.0, "/s"),
        "dropd/s":    MetricProfile(0.0,  10.0, 0.0,   10000.0, "/s"),
        "squeezd/s":  MetricProfile(0.5,  0.5,  0.0,    1000.0, "/s"),
        "rx_rps/s":   MetricProfile(0.0,  0.0,  0.0,  100000.0, "/s"),
        # Memory (light usage)
        "kbmemfree":  MetricProfile(0.60, 0.05, 0.01, 0.99, "fraction"),
        "kbmemused":  MetricProfile(0.40, 0.05, 0.01, 0.99, "fraction"),
        "%memused":   MetricProfile(40.0, 5.0,  1.0,  99.9, "%"),
        "kbbuffers":  MetricProfile(0.01, 0.003, 0.0, 0.05, "fraction"),
        "kbcached":   MetricProfile(0.10, 0.02,  0.0, 0.3,  "fraction"),
        "kbdirty":    MetricProfile(0.001,0.0005,0.0, 0.02, "fraction"),
        # Load
        "ldavg-1":    MetricProfile(2.0,  1.0,  0.0, 32.0),
        "ldavg-5":    MetricProfile(1.9,  0.9,  0.0, 32.0),
        "ldavg-15":   MetricProfile(1.8,  0.8,  0.0, 32.0),
        # Disk (minimal)
        "tps":        MetricProfile(20.0, 8.0,   0.0, 500.0, "/s"),
        "await":      MetricProfile(2.0,  1.0,   0.0, 200.0, "ms"),
        "svctm":      MetricProfile(0.5,  0.2,   0.0,  50.0, "ms"),
        "%util":      MetricProfile(5.0,  3.0,   0.0,  80.0, "%"),
        # Process
        "proc/s":     MetricProfile(30.0,  8.0,  0.0, 300.0, "/s"),
        "cswch/s":    MetricProfile(2000.0, 500.0, 0.0, 50000.0, "/s"),
        "fault/s":    MetricProfile(200.0, 80.0, 0.0, 10000.0, "/s"),
    }

    p.correlations = [
        # High packet rate -> high %soft (software interrupt)
        CorrelationRule("rxpck/s", "%soft", coefficient=0.00003,
                        description="RX packets -> softirq CPU"),
        # High %soft -> high %sys
        CorrelationRule("%soft", "%sys", coefficient=0.4,
                        description="Softirq -> sys CPU"),
        # Saturation -> packet drops
        CorrelationRule("%ifutil", "rxdrop/s", coefficient=0.5,
                        description="Interface saturation -> drops"),
    ]
    return p


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NODE_PROFILES: Dict[str, NodeProfile] = {
    "compute":       build_compute_profile(),
    "ceph_storage":  build_ceph_profile(),
    "control_plane": build_control_plane_profile(),
    "network":       build_network_profile(),
}


def get_profile(node_type: str) -> NodeProfile:
    if node_type not in NODE_PROFILES:
        raise ValueError(f"Unknown node type: {node_type}. Valid: {list(NODE_PROFILES)}")
    return NODE_PROFILES[node_type]


# Hostname prefix mapping
HOSTNAME_PREFIXES: Dict[str, str] = {
    "compute":       "compute",
    "ceph_storage":  "ceph",
    "control_plane": "ctrl",
    "network":       "net",
}
