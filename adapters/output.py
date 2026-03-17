"""
Output Adapters - SAR-compatible CSV formatter and hierarchical JSON output.
Supports chunked writing, compression, and streaming modes.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional

import numpy as np
import pandas as pd

from config import OutputConfig, OutputFormat
from engine.generator import SAR_COLUMNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class BaseOutputAdapter:
    def __init__(self, output_cfg: OutputConfig):
        self.cfg = output_cfg
        self.output_dir = Path(output_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _make_path(self, suffix: str, ext: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{self.cfg.filename_prefix}_{suffix}_{ts}.{ext}"
        if self.cfg.compress:
            fname += ".gz"
        return self.output_dir / fname

    def write(self, df: pd.DataFrame) -> List[Path]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CSV Adapter
# ---------------------------------------------------------------------------

class CSVOutputAdapter(BaseOutputAdapter):
    """
    Writes SAR-compatible CSV files.
    Column order matches the official SAR output specification.
    """

    SAR_FLOAT_FMT = "%.2f"

    def write(self, df: pd.DataFrame, suffix: str = "full") -> Path:
        """Write full DataFrame to a single CSV file."""
        path = self._make_path(suffix, "csv")
        logger.info(f"Writing CSV -> {path}  ({len(df):,} rows)")

        # Ensure all SAR columns present
        out_df = self._prepare_dataframe(df)

        if self.cfg.compress:
            with gzip.open(str(path), 'wt', encoding='utf-8') as f:
                out_df.to_csv(f, index=False, float_format=self.SAR_FLOAT_FMT)
        else:
            out_df.to_csv(str(path), index=False, float_format=self.SAR_FLOAT_FMT)

        logger.info(f"CSV written: {path}  ({path.stat().st_size / 1024:.1f} kB)")
        return path

    def write_chunks(
        self,
        chunk_iter: Iterator[pd.DataFrame],
        suffix: str = "chunked",
    ) -> List[Path]:
        """Write chunks to separate files."""
        paths: List[Path] = []
        for i, chunk in enumerate(chunk_iter):
            p = self.write(chunk, suffix=f"{suffix}_{i:04d}")
            paths.append(p)
        return paths

    def write_by_node(self, df: pd.DataFrame) -> List[Path]:
        """Write one CSV file per hostname."""
        paths: List[Path] = []
        for hostname, node_df in df.groupby('hostname'):
            p = self.write(node_df, suffix=f"node_{hostname}")
            paths.append(p)
        return paths

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct column order and fill missing SAR columns."""
        # Add any missing columns
        out = df.copy()
        for col in SAR_COLUMNS:
            if col not in out.columns:
                out[col] = 0
        return out[SAR_COLUMNS]


# ---------------------------------------------------------------------------
# JSON Adapter
# ---------------------------------------------------------------------------

class JSONOutputAdapter(BaseOutputAdapter):
    """
    Writes hierarchical JSON output for modern data pipelines.
    Structure: { metadata, nodes: { hostname: [ {timestamp, metrics}, ... ] } }
    """

    def write(self, df: pd.DataFrame, suffix: str = "full") -> Path:
        path = self._make_path(suffix, "json")
        logger.info(f"Writing JSON -> {path}  ({len(df):,} rows)")

        payload = self._build_payload(df)

        open_fn = gzip.open if self.cfg.compress else open
        mode = 'wt' if self.cfg.compress else 'w'

        with open_fn(str(path), mode, encoding='utf-8') as f:
            json.dump(payload, f, separators=(',', ':'), allow_nan=False,
                      default=self._json_default)

        logger.info(f"JSON written: {path}  ({path.stat().st_size / 1024:.1f} kB)")
        return path

    def write_ndjson(self, df: pd.DataFrame, suffix: str = "ndjson") -> Path:
        """Write newline-delimited JSON (one record per line) for streaming pipelines."""
        path = self._make_path(suffix, "ndjson")
        logger.info(f"Writing NDJSON -> {path}")

        open_fn = gzip.open if self.cfg.compress else open
        mode = 'wt' if self.cfg.compress else 'w'

        with open_fn(str(path), mode, encoding='utf-8') as f:
            for _, row in df.iterrows():
                record = self._row_to_dict(row)
                f.write(json.dumps(record, default=self._json_default) + '\n')

        return path

    def _build_payload(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build hierarchical JSON payload."""
        # Metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_records": len(df),
            "columns": SAR_COLUMNS,
            "nodes": df['hostname'].unique().tolist() if 'hostname' in df.columns else [],
            "time_range": {
                "start": df['DateTime'].min() if 'DateTime' in df.columns else None,
                "end":   df['DateTime'].max() if 'DateTime' in df.columns else None,
            },
        }

        # Node-grouped records
        nodes: Dict[str, Any] = {}
        if 'hostname' in df.columns:
            for hostname, node_df in df.groupby('hostname'):
                nodes[hostname] = {
                    "node_type": self._infer_node_type(hostname),
                    "records": [self._row_to_dict(row) for _, row in node_df.iterrows()],
                }
        else:
            nodes["unknown"] = {
                "records": [self._row_to_dict(row) for _, row in df.iterrows()],
            }

        return {"metadata": metadata, "nodes": nodes}

    def _row_to_dict(self, row: pd.Series) -> Dict[str, Any]:
        """Convert a DataFrame row to a clean dict."""
        d: Dict[str, Any] = {}
        for col in SAR_COLUMNS:
            val = row.get(col, 0)
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val) if not np.isnan(val) and not np.isinf(val) else 0.0
            d[col] = val
        return d

    # -- Nested JSON (document specification format) --------------------

    def write_nested(self, df: pd.DataFrame, suffix: str = "nested") -> Path:
        """Write nested/grouped JSON matching the document specification."""
        path = self._make_path(suffix, "json")
        logger.info(f"Writing nested JSON -> {path}  ({len(df):,} rows)")

        payload = self._build_nested_payload(df)

        open_fn = gzip.open if self.cfg.compress else open
        mode = 'wt' if self.cfg.compress else 'w'
        with open_fn(str(path), mode, encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=self._json_default)

        logger.info(f"Nested JSON written: {path}  ({path.stat().st_size / 1024:.1f} kB)")
        return path

    def _build_nested_payload(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build document-spec nested JSON structure with metric groups."""
        cpu_cols = ['%usr', '%nice', '%sys', '%iowait', '%steal',
                    '%irq', '%soft', '%guest', '%gnice', '%idle']
        mem_cols = ['kbmemfree', 'kbmemused', '%memused',
                    'kbbuffers', 'kbcached', 'kbcommit', '%commit',
                    'kbactive', 'kbinact', 'kbdirty']
        disk_cols = ['tps', 'rtps', 'wtps', 'bread/s', 'bwrtn/s',
                     'await', 'svctm', '%util', 'rd_sec/s', 'wr_sec/s',
                     'avgrq-sz', 'avgqu-sz']
        net_cols = ['rxpck/s', 'txpck/s', 'rxkB/s', 'txkB/s',
                    'rxcmp/s', 'txcmp/s', '%ifutil']

        records = []
        for _, row in df.iterrows():
            record = {
                "timestamp": row.get("DateTime", ""),
                "hostname": row.get("hostname", ""),
                "node_type": self._infer_node_type(str(row.get("hostname", ""))),
                "metrics": {
                    "cpu":     {c: self._safe_val(row, c) for c in cpu_cols if c in df.columns},
                    "memory":  {c: self._safe_val(row, c) for c in mem_cols if c in df.columns},
                    "disk":    {c: self._safe_val(row, c) for c in disk_cols if c in df.columns},
                    "network": {c: self._safe_val(row, c) for c in net_cols if c in df.columns},
                },
            }
            records.append(record)

        return {"records": records, "total": len(records)}

    @staticmethod
    def _safe_val(row, col):
        """Extract a JSON-safe scalar value from a row."""
        v = row.get(col, 0)
        if hasattr(v, 'item'):
            v = v.item()
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return 0.0
        return v

    @staticmethod
    def _infer_node_type(hostname: str) -> str:
        """Infer node type from hostname prefix."""
        if hostname.startswith("compute"):
            return "compute"
        elif hostname.startswith("ceph"):
            return "ceph_storage"
        elif hostname.startswith("ctrl"):
            return "control_plane"
        elif hostname.startswith("net"):
            return "network"
        return "unknown"

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Combined adapter
# ---------------------------------------------------------------------------

class OutputManager:
    """
    Manages all output adapters based on configuration.
    Provides a unified write interface.
    """

    def __init__(self, output_cfg: OutputConfig):
        self.cfg = output_cfg
        self.csv_adapter  = CSVOutputAdapter(output_cfg)
        self.json_adapter = JSONOutputAdapter(output_cfg)

    def write(self, df: pd.DataFrame) -> Dict[str, List[Path]]:
        """Write data in all configured formats. Returns paths written."""
        results: Dict[str, List[Path]] = {}
        fmt = self.cfg.format

        if fmt in (OutputFormat.CSV, OutputFormat.BOTH):
            csv_path = self.csv_adapter.write(df)
            results['csv'] = [csv_path]
            logger.info(f"CSV output: {csv_path}")

        if fmt in (OutputFormat.JSON, OutputFormat.BOTH):
            json_path = self.json_adapter.write(df)
            results['json'] = [json_path]
            logger.info(f"JSON output: {json_path}")

        return results

    def write_by_node(self, df: pd.DataFrame) -> Dict[str, List[Path]]:
        """Write per-node files."""
        results: Dict[str, List[Path]] = {}
        fmt = self.cfg.format

        if fmt in (OutputFormat.CSV, OutputFormat.BOTH):
            paths = self.csv_adapter.write_by_node(df)
            results['csv'] = paths

        if fmt in (OutputFormat.JSON, OutputFormat.BOTH):
            paths = []
            for hostname, node_df in df.groupby('hostname'):
                p = self.json_adapter.write(node_df, suffix=f"node_{hostname}")
                paths.append(p)
            results['json'] = paths

        return results

    def write_nested_json(self, df: pd.DataFrame) -> Path:
        """Write nested JSON format matching the document specification."""
        return self.json_adapter.write_nested(df)

    def print_summary(self, results: Dict[str, List[Path]]) -> None:
        """Print a summary of written files."""
        total_size = 0
        print("\n--- Output Summary ---")
        for fmt, paths in results.items():
            for p in paths:
                size_kb = Path(p).stat().st_size / 1024
                total_size += size_kb
                print(f"  [{fmt.upper()}] {p}  ({size_kb:.1f} kB)")
        print(f"  Total: {total_size / 1024:.2f} MB")
        print("----------------------\n")
