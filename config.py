"""
Configuration Manager – Pydantic v2 tabanlı SAR Log Data Generator yapılandırma modülü.
"""
from __future__ import annotations

import json
import yaml
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Forward-compatible: new config sections (optional, imported lazily to avoid circular deps)
# DatabaseConfig  -> adapters/database.py
# StreamingConfig -> streaming/streamer.py
# RotationConfig  -> adapters/rotation.py


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    COMPUTE        = "compute"
    CEPH_STORAGE   = "ceph_storage"
    CONTROL_PLANE  = "control_plane"
    NETWORK        = "network"


class AnomalyFrequency(str, Enum):
    NONE   = "none"
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class AnomalySeverity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class ScenarioType(str, Enum):
    STORAGE_CONTENTION  = "storage_contention"
    MEMORY_PRESSURE     = "memory_pressure"
    NETWORK_SATURATION  = "network_saturation"
    CPU_STEAL_SPIKE     = "cpu_steal_spike"
    CASCADING_FAILURE   = "cascading_failure"
    GRADUAL_DEGRADATION = "gradual_degradation"
    BACKUP_STORM        = "backup_storm"


class OutputFormat(str, Enum):
    CSV  = "csv"
    JSON = "json"
    BOTH = "both"


class DataQualityLevel(str, Enum):
    """Data quality level presets for configurable realism."""
    CLEAN    = "clean"       # Minimal noise, smooth signals
    NORMAL   = "normal"      # Default realistic behavior
    NOISY    = "noisy"       # High noise, frequent micro-anomalies
    DEGRADED = "degraded"    # Systematic drift and artifacts


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class NodeConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    type:               NodeType
    count:              int              = Field(default=1,           ge=1,   le=1000)
    base_load:          float            = Field(default=0.4,         ge=0.0, le=1.0)
    anomaly_frequency:  AnomalyFrequency = AnomalyFrequency.LOW
    hostname_prefix:    Optional[str]    = None
    total_memory_kb:    int              = Field(default=131_072_000, ge=1024)
    cpu_count:          int              = Field(default=64,          ge=1)
    disk_count:         int              = Field(default=12,          ge=1)
    network_interfaces: int              = Field(default=2,           ge=1)

    @model_validator(mode="after")
    def _set_hostname_prefix(self) -> "NodeConfig":
        if self.hostname_prefix is None:
            _map = {
                NodeType.COMPUTE:       "compute",
                NodeType.CEPH_STORAGE:  "ceph",
                NodeType.CONTROL_PLANE: "ctrl",
                NodeType.NETWORK:       "net",
            }
            self.hostname_prefix = _map[self.type]
        return self


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    type:              ScenarioType
    start_time:        datetime
    end_time:          Optional[datetime]           = None
    duration_hours:    Optional[float]              = Field(default=None, gt=0)
    severity:          AnomalySeverity              = AnomalySeverity.MEDIUM
    target_node_types: Optional[List[NodeType]]     = None
    ramp_up_minutes:   int                          = Field(default=5, ge=0)
    ramp_down_minutes: int                          = Field(default=5, ge=0)
    parameters:        Dict[str, Any]               = Field(default_factory=dict)

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _parse_datetime(cls, v: Any) -> Any:
        if v is None or isinstance(v, datetime):
            return v
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(v), fmt)
            except ValueError:
                pass
        raise ValueError(f"Cannot parse datetime: {v!r}")

    @model_validator(mode="after")
    def _resolve_end_time(self) -> "ScenarioConfig":
        if self.end_time is None:
            hours = self.duration_hours if self.duration_hours is not None else 1.0
            self.end_time = self.start_time + timedelta(hours=hours)
        return self


class OutputConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    format:          OutputFormat = OutputFormat.CSV
    output_dir:      str          = "./output"
    filename_prefix: str          = "sar_synthetic"
    compress:        bool         = False
    chunk_size:      int          = Field(default=100_000, ge=1_000)
    include_header:  bool         = True


class SimulationConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    start_time:       datetime
    end_time:         datetime
    nodes:            List[NodeConfig]
    interval_seconds: int                  = Field(default=300, ge=1, le=3600)
    scenarios:        List[ScenarioConfig] = Field(default_factory=list)
    output:           OutputConfig         = Field(default_factory=OutputConfig)
    random_seed:      Optional[int]        = None
    diurnal_pattern:  bool                 = True
    weekly_pattern:   bool                 = True
    noise_level:      float                = Field(default=0.05, ge=0.0, le=1.0)
    description:      str                  = ""
    # Opsiyonel genişletilmiş yapılandırmalar (ilgili modüller tarafından okunur)
    database:         Optional[Dict[str, Any]] = None   # adapters/database.py
    streaming:        Optional[Dict[str, Any]] = None   # streaming/streamer.py
    data_quality:     DataQualityLevel         = DataQualityLevel.NORMAL

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _parse_datetime(cls, v: Any) -> Any:
        if isinstance(v, datetime):
            return v
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(v), fmt)
            except ValueError:
                pass
        raise ValueError(f"Cannot parse datetime: {v!r}")

    @model_validator(mode="after")
    def _validate_time_range(self) -> "SimulationConfig":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        return self

    @property
    def total_seconds(self) -> int:
        return int((self.end_time - self.start_time).total_seconds())

    @property
    def num_intervals(self) -> int:
        return self.total_seconds // self.interval_seconds


# ---------------------------------------------------------------------------
# Loader & Helpers
# ---------------------------------------------------------------------------

def load_config(path: Union[str, Path]) -> SimulationConfig:
    """YAML veya JSON dosyasından SimulationConfig yükle ve doğrula."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) if path.suffix in (".yaml", ".yml") else json.load(f)
    sim_data = raw.get("simulation", raw)
    return SimulationConfig(**sim_data)


def get_default_config() -> SimulationConfig:
    """Varsayılan 1-haftalık Telco Cloud simülasyon config'i döndür."""
    return SimulationConfig(
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 7, 23, 59, 59),
        interval_seconds=300,
        description="Default Telco Cloud simulation – 1 week",
        nodes=[
            NodeConfig(type=NodeType.CEPH_STORAGE,  count=3,  base_load=0.6,
                       anomaly_frequency=AnomalyFrequency.LOW,
                       total_memory_kb=262_144_000, cpu_count=32, disk_count=24),
            NodeConfig(type=NodeType.COMPUTE,        count=10, base_load=0.4,
                       anomaly_frequency=AnomalyFrequency.MEDIUM,
                       total_memory_kb=131_072_000, cpu_count=64, disk_count=4),
            NodeConfig(type=NodeType.CONTROL_PLANE,  count=3,  base_load=0.3,
                       anomaly_frequency=AnomalyFrequency.LOW,
                       total_memory_kb=65_536_000,  cpu_count=16, disk_count=2),
            NodeConfig(type=NodeType.NETWORK,        count=2,  base_load=0.35,
                       anomaly_frequency=AnomalyFrequency.LOW,
                       total_memory_kb=32_768_000,  cpu_count=8,  disk_count=2),
        ],
        scenarios=[
            ScenarioConfig(
                type=ScenarioType.STORAGE_CONTENTION,
                start_time=datetime(2024, 1, 3, 14, 0, 0),
                duration_hours=2, severity=AnomalySeverity.HIGH,
                target_node_types=[NodeType.CEPH_STORAGE, NodeType.COMPUTE],
            ),
            ScenarioConfig(
                type=ScenarioType.MEMORY_PRESSURE,
                start_time=datetime(2024, 1, 5, 2, 0, 0),
                duration_hours=3, severity=AnomalySeverity.MEDIUM,
                target_node_types=[NodeType.COMPUTE],
            ),
            ScenarioConfig(
                type=ScenarioType.BACKUP_STORM,
                start_time=datetime(2024, 1, 6, 1, 0, 0),
                duration_hours=4, severity=AnomalySeverity.MEDIUM,
            ),
        ],
        output=OutputConfig(
            format=OutputFormat.BOTH, output_dir="./output",
            compress=False, chunk_size=100_000,
        ),
        random_seed=42, diurnal_pattern=True, weekly_pattern=True, noise_level=0.05,
    )


EXAMPLE_YAML = """\
simulation:
  start_time: "2024-01-01 00:00:00"
  end_time: "2024-01-07 23:59:59"
  interval_seconds: 300
  description: "Telco Cloud 1-week simulation"
  random_seed: 42
  diurnal_pattern: true
  weekly_pattern: true
  noise_level: 0.05

  nodes:
    - type: "ceph_storage"
      count: 3
      base_load: 0.6
      anomaly_frequency: "low"
      total_memory_kb: 262144000
      cpu_count: 32
      disk_count: 24

    - type: "compute"
      count: 10
      base_load: 0.4
      anomaly_frequency: "medium"
      total_memory_kb: 131072000
      cpu_count: 64
      disk_count: 4

    - type: "control_plane"
      count: 3
      base_load: 0.3
      anomaly_frequency: "low"
      total_memory_kb: 65536000
      cpu_count: 16
      disk_count: 2

    - type: "network"
      count: 2
      base_load: 0.35
      anomaly_frequency: "low"
      total_memory_kb: 32768000
      cpu_count: 8
      disk_count: 2

  scenarios:
    - type: "storage_contention"
      start_time: "2024-01-03 14:00:00"
      duration_hours: 2
      severity: "high"
      target_node_types: ["ceph_storage", "compute"]

    - type: "memory_pressure"
      start_time: "2024-01-05 02:00:00"
      duration_hours: 3
      severity: "medium"
      target_node_types: ["compute"]

    - type: "backup_storm"
      start_time: "2024-01-06 01:00:00"
      duration_hours: 4
      severity: "medium"

  output:
    format: "both"
    output_dir: "./output"
    filename_prefix: "sar_synthetic"
    compress: false
    chunk_size: 100000
"""


def write_example_config(path: Union[str, Path] = "config_example.yaml") -> None:
    with open(path, "w") as f:
        f.write(EXAMPLE_YAML)
    print(f"Example config written to: {path}")
