#!/usr/bin/env python3
"""Streamlit GUI for SAR Generator — full feature parity with CLI.

Run with:  streamlit run app.py
The CLI (python main.py) is completely unchanged.
"""
from __future__ import annotations

import io
import json
import logging
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import streamlit as st

# ── Page config — must be the first Streamlit call ──────────────────────────
st.set_page_config(
    page_title="SAR Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #162032 100%);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(.stBadge) {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] h3 { color: #7fb3d3 !important; font-size: 1rem; }
[data-testid="stSidebar"] .stButton > button {
    background: #1e3a5f; color: #e2e8f0;
    border: 1px solid #2e5a8a; border-radius: 7px;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #1d4ed8; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white; border: none; font-size: 1rem; font-weight: 600;
}
.block-container { padding-top: 1.25rem; }
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #0f3460 55%, #0d1b2a 100%);
    border-radius: 14px; padding: 1.5rem 2rem; margin-bottom: 1.25rem;
    border: 1px solid #1e3a5f;
    box-shadow: 0 4px 24px rgba(0,0,0,.4);
}
.hero h1 { color: #e2e8f0; font-size: 2rem; margin: 0; font-weight: 800; letter-spacing: -.5px; }
.hero p  { color: #7fb3d3; margin: .35rem 0 0; font-size: .9rem; }
.hero code { background: #1e3a5f; padding: 1px 6px; border-radius: 4px; font-size: .8rem; color: #93c5fd; }
div[data-testid="metric-container"] {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: .75rem 1rem;
}
div[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: .78rem; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important; font-size: 1.5rem !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #1e293b; border-radius: 8px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] { color: #94a3b8; border-radius: 6px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #2563eb !important; color: #fff !important; }
.section-head {
    font-size: .95rem; font-weight: 700; color: #60a5fa;
    border-bottom: 1px solid #334155; padding-bottom: .25rem; margin: 1rem 0 .6rem;
}
</style>
""", unsafe_allow_html=True)

# ── Path fix ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    AnomalyFrequency, AnomalySeverity, DataQualityLevel, EXAMPLE_YAML,
    NodeConfig, NodeType, OutputConfig, OutputFormat,
    ScenarioConfig, ScenarioType, SimulationConfig,
)

# ── Enum option lists ────────────────────────────────────────────────────────
_NODE_TYPES     = [e.value for e in NodeType]
_ANOMALY_FREQ   = [e.value for e in AnomalyFrequency]
_SEVERITIES     = [e.value for e in AnomalySeverity]
_SCENARIO_TYPES = [e.value for e in ScenarioType]
_DATA_QUALITY   = [e.value for e in DataQualityLevel]
_OUTPUT_FORMATS = [e.value for e in OutputFormat]
_STREAM_MODES   = ["stdout", "socket", "websocket"]
_INTERVALS      = [60, 120, 300, 600, 900, 1800, 3600]


# ── Session-state bootstrap ──────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict = {
        "df":                None,
        "sim_config":        None,
        "gen_stats":         {},
        "ng_counter":        4,
        "sc_counter":        3,
        "_config_file_hash": None,
        "node_groups": [
            {"id": 0, "type": "ceph_storage",  "count": 3,  "base_load": 0.60, "anomaly_freq": "low"},
            {"id": 1, "type": "compute",        "count": 10, "base_load": 0.40, "anomaly_freq": "medium"},
            {"id": 2, "type": "control_plane",  "count": 3,  "base_load": 0.30, "anomaly_freq": "low"},
            {"id": 3, "type": "network",        "count": 2,  "base_load": 0.35, "anomaly_freq": "low"},
        ],
        "scenarios": [
            {"id": 0, "type": "storage_contention", "start": datetime(2024, 1, 3, 14),
             "duration": 2.0, "severity": "high",   "targets": ["ceph_storage", "compute"]},
            {"id": 1, "type": "memory_pressure",    "start": datetime(2024, 1, 5, 2),
             "duration": 3.0, "severity": "medium", "targets": ["compute"]},
            {"id": 2, "type": "backup_storm",       "start": datetime(2024, 1, 6, 1),
             "duration": 4.0, "severity": "medium", "targets": []},
        ],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── Config loader ─────────────────────────────────────────────────────────────
def _load_config_into_state(cfg: SimulationConfig) -> None:
    """Populate all sidebar widget states from a loaded SimulationConfig."""
    # Simulation period
    st.session_state["start_date"]   = cfg.start_time.date()
    st.session_state["end_date"]     = cfg.end_time.date()
    # Snap interval to nearest valid option
    closest = min(_INTERVALS, key=lambda x: abs(x - cfg.interval_seconds))
    st.session_state["interval_sec"] = closest
    # Advanced
    st.session_state["seed"]         = int(cfg.random_seed) if cfg.random_seed is not None else 42
    st.session_state["noise"]        = float(cfg.noise_level)
    st.session_state["diurnal"]      = bool(cfg.diurnal_pattern)
    st.session_state["weekly"]       = bool(cfg.weekly_pattern)
    st.session_state["data_quality"] = getattr(cfg.data_quality, "value", str(cfg.data_quality))
    # Output
    st.session_state["out_fmt"]      = getattr(cfg.output.format, "value", str(cfg.output.format))
    st.session_state["out_dir"]      = cfg.output.output_dir
    st.session_state["compress"]     = bool(cfg.output.compress)

    # Node groups — clear old widget keys, install new entries with fresh IDs
    for ng in st.session_state.node_groups:
        _id = ng["id"]
        for suf in ["t", "c", "l", "f"]:
            st.session_state.pop(f"ng_{suf}{_id}", None)

    base = st.session_state.ng_counter
    new_ngs = []
    for i, nc in enumerate(cfg.nodes):
        _id = base + i
        new_ngs.append({
            "id":           _id,
            "type":         getattr(nc.type, "value", str(nc.type)),
            "count":        nc.count,
            "base_load":    nc.base_load,
            "anomaly_freq": getattr(nc.anomaly_frequency, "value", str(nc.anomaly_frequency)),
        })
    st.session_state.ng_counter = base + len(cfg.nodes)
    st.session_state.node_groups = new_ngs

    # Scenarios — same pattern
    for sc in st.session_state.scenarios:
        _id = sc["id"]
        for suf in ["t", "s", "d", "ti", "dur", "tg"]:
            st.session_state.pop(f"sc_{suf}{_id}", None)

    base = st.session_state.sc_counter
    new_scs = []
    for i, sc in enumerate(cfg.scenarios):
        _id = base + i
        dur_h = (sc.end_time - sc.start_time).total_seconds() / 3600
        new_scs.append({
            "id":       _id,
            "type":     getattr(sc.type, "value", str(sc.type)),
            "start":    sc.start_time,
            "duration": round(dur_h, 2),
            "severity": getattr(sc.severity, "value", str(sc.severity)),
            "targets":  [getattr(t, "value", str(t)) for t in (sc.target_node_types or [])],
        })
    st.session_state.sc_counter = base + len(cfg.scenarios)
    st.session_state.scenarios = new_scs


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> Tuple[Optional[SimulationConfig], bool]:
    """Renders sidebar. Returns (config | None, generate_clicked)."""

    with st.sidebar:
        st.markdown("### ⚙ Configuration")
        st.markdown("---")

        # ── YAML / JSON config upload ─────────────────────────────
        st.markdown("**📂 Load Config File**")
        uploaded = st.file_uploader(
            "Upload YAML or JSON config", type=["yaml", "yml", "json"],
            key="config_upload", label_visibility="collapsed",
        )
        if uploaded is not None:
            _hash = hash(uploaded.getvalue())
            if _hash != st.session_state["_config_file_hash"]:
                if st.button("Apply Uploaded Config", use_container_width=True, key="apply_cfg"):
                    try:
                        import yaml as _yaml
                        raw_bytes = uploaded.getvalue()
                        if uploaded.name.endswith(".json"):
                            raw = json.loads(raw_bytes)
                        else:
                            raw = _yaml.safe_load(raw_bytes)
                        sim_data = raw.get("simulation", raw)
                        cfg_loaded = SimulationConfig(**sim_data)
                        _load_config_into_state(cfg_loaded)
                        st.session_state["_config_file_hash"] = _hash
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to load config: {exc}")
            else:
                st.caption("✅ Config applied")

        st.markdown("---")

        # ── Simulation Period ─────────────────────────────────────
        st.markdown("**📅 Simulation Period**")
        c1, c2 = st.columns(2)
        start_date: date = c1.date_input("Start", date(2024, 1, 1), key="start_date")
        end_date:   date = c2.date_input("End",   date(2024, 1, 7), key="end_date")
        interval = st.select_slider(
            "Interval (seconds)", _INTERVALS, value=300, key="interval_sec"
        )

        # ── Advanced ─────────────────────────────────────────────
        with st.expander("🔧 Advanced Settings"):
            seed    = st.number_input("Random Seed", 0, 99999, 42, step=1, key="seed")
            noise   = st.slider("Noise Level", 0.0, 1.0, 0.05, 0.01, key="noise")
            dq      = st.selectbox("Data Quality", _DATA_QUALITY, index=1, key="data_quality")
            ca, cb  = st.columns(2)
            diurnal = ca.checkbox("Diurnal", value=True, key="diurnal")
            weekly  = cb.checkbox("Weekly",  value=True, key="weekly")
            verbose = st.checkbox("Verbose / Debug Logging", value=False, key="verbose")

        # ── Node Groups ───────────────────────────────────────────
        st.markdown("**🖥 Node Groups**")
        _ng_remove: List[int] = []
        for ng in st.session_state.node_groups:
            _id = ng["id"]
            with st.expander(f"{ng['type']}  ×{ng['count']}", expanded=False):
                try:   _ti = _NODE_TYPES.index(ng["type"])
                except ValueError: _ti = 0
                ng["type"] = st.selectbox("Type", _NODE_TYPES, index=_ti, key=f"ng_t{_id}")

                ca, cb = st.columns(2)
                ng["count"]        = ca.number_input("Count", 1, 1000, int(ng["count"]), 1, key=f"ng_c{_id}")
                ng["base_load"]    = cb.slider("Base Load", 0.0, 1.0, float(ng["base_load"]), 0.05, key=f"ng_l{_id}")

                try:   _fi = _ANOMALY_FREQ.index(ng["anomaly_freq"])
                except ValueError: _fi = 0
                ng["anomaly_freq"] = st.selectbox("Anomaly Freq", _ANOMALY_FREQ, index=_fi, key=f"ng_f{_id}")

                if st.button("Remove", key=f"ng_rm{_id}", use_container_width=True):
                    _ng_remove.append(_id)

        if _ng_remove:
            st.session_state.node_groups = [
                g for g in st.session_state.node_groups if g["id"] not in _ng_remove
            ]
            st.rerun()

        if st.button("➕ Add Node Group", use_container_width=True, key="add_ng"):
            nid = st.session_state.ng_counter
            st.session_state.ng_counter += 1
            st.session_state.node_groups.append(
                {"id": nid, "type": "compute", "count": 1, "base_load": 0.4, "anomaly_freq": "low"}
            )
            st.rerun()

        # ── Scenarios ─────────────────────────────────────────────
        st.markdown("**⚡ Anomaly Scenarios**")
        _sc_remove: List[int] = []
        for sc in st.session_state.scenarios:
            _id = sc["id"]
            with st.expander(f"{sc['type']}  [{sc['severity']}]", expanded=False):
                try:   _ti = _SCENARIO_TYPES.index(sc["type"])
                except ValueError: _ti = 0
                sc["type"] = st.selectbox("Scenario", _SCENARIO_TYPES, index=_ti, key=f"sc_t{_id}")

                try:   _si = _SEVERITIES.index(sc["severity"])
                except ValueError: _si = 1
                sc["severity"] = st.selectbox("Severity", _SEVERITIES, index=_si, key=f"sc_s{_id}")

                ca, cb = st.columns(2)
                _sd = ca.date_input("Date", sc["start"].date(), key=f"sc_d{_id}")
                _st = cb.time_input("Time", sc["start"].time(), key=f"sc_ti{_id}")
                sc["start"] = datetime.combine(_sd, _st)

                sc["duration"] = st.number_input(
                    "Duration (hours)", 0.5, 72.0, float(sc["duration"]), 0.5, key=f"sc_dur{_id}"
                )
                _def = [t for t in sc.get("targets", []) if t in _NODE_TYPES]
                sc["targets"] = st.multiselect(
                    "Target nodes (empty = all)", _NODE_TYPES, default=_def, key=f"sc_tg{_id}"
                )
                if st.button("Remove", key=f"sc_rm{_id}", use_container_width=True):
                    _sc_remove.append(_id)

        if _sc_remove:
            st.session_state.scenarios = [
                s for s in st.session_state.scenarios if s["id"] not in _sc_remove
            ]
            st.rerun()

        if st.button("➕ Add Scenario", use_container_width=True, key="add_sc"):
            sid = st.session_state.sc_counter
            st.session_state.sc_counter += 1
            st.session_state.scenarios.append({
                "id": sid, "type": "cpu_steal_spike",
                "start": datetime(start_date.year, start_date.month, start_date.day),
                "duration": 2.0, "severity": "medium", "targets": [],
            })
            st.rerun()

        # ── Output Settings ───────────────────────────────────────
        with st.expander("📁 Output Settings"):
            out_fmt  = st.selectbox("Format", _OUTPUT_FORMATS, key="out_fmt")
            out_dir  = st.text_input("Output Directory", "./output", key="out_dir")
            compress = st.checkbox("Compress (gzip)", value=False, key="compress")
            by_node  = st.checkbox(
                "Write per-node files (--by-node)", value=False, key="by_node",
                help="One file per hostname instead of a single combined file.",
            )

        # ── Streaming Settings ────────────────────────────────────
        with st.expander("📡 Streaming Settings"):
            stream_enabled = st.checkbox("Enable streaming", value=False, key="stream_enabled")
            if stream_enabled:
                stream_mode = st.selectbox("Mode", _STREAM_MODES, key="stream_mode",
                                           help="stdout → terminal; socket/websocket → network clients")
                ca, cb = st.columns(2)
                stream_host = ca.text_input("Host", "0.0.0.0", key="stream_host")
                stream_port = cb.number_input("Port", 1024, 65535, 9000, step=1, key="stream_port")
                stream_fmt  = st.selectbox("Record format", ["json", "csv"], key="stream_fmt")
                stream_rows = st.number_input("Flush interval (rows)", 10, 10000, 100, step=10,
                                              key="stream_rows")
                if stream_mode != "stdout":
                    st.caption(f"Connect externally to **{stream_host}:{stream_port}** after clicking Generate.")

        st.markdown("---")

        # ── Utility buttons ───────────────────────────────────────
        st.download_button(
            "⬇ Download Example Config (YAML)",
            data=EXAMPLE_YAML,
            file_name="config_example.yaml",
            mime="text/yaml",
            use_container_width=True,
            key="dl_example_cfg",
        )

        gen_clicked = st.button(
            "🚀  Generate SAR Data", type="primary", use_container_width=True, key="gen_btn"
        )

    # ── Build SimulationConfig ────────────────────────────────────────────────
    start_dt = datetime(start_date.year, start_date.month, start_date.day, 0,  0,  0)
    end_dt   = datetime(end_date.year,   end_date.month,   end_date.day,  23, 59, 59)

    try:
        node_cfgs = [
            NodeConfig(
                type=NodeType(ng["type"]),
                count=int(ng["count"]),
                base_load=float(ng["base_load"]),
                anomaly_frequency=AnomalyFrequency(ng["anomaly_freq"]),
            )
            for ng in st.session_state.node_groups
        ]

        scen_cfgs = []
        for sc in st.session_state.scenarios:
            tgts = [NodeType(t) for t in sc.get("targets", [])] or None
            scen_cfgs.append(ScenarioConfig(
                type=ScenarioType(sc["type"]),
                start_time=sc["start"],
                duration_hours=float(sc["duration"]),
                severity=AnomalySeverity(sc["severity"]),
                target_node_types=tgts,
            ))

        cfg = SimulationConfig(
            start_time=start_dt,
            end_time=end_dt,
            interval_seconds=int(st.session_state.interval_sec),
            nodes=node_cfgs,
            scenarios=scen_cfgs,
            output=OutputConfig(
                format=OutputFormat(st.session_state.out_fmt),
                output_dir=st.session_state.out_dir,
                compress=bool(st.session_state.compress),
            ),
            random_seed=int(st.session_state.seed),
            noise_level=float(st.session_state.noise),
            diurnal_pattern=bool(st.session_state.diurnal),
            weekly_pattern=bool(st.session_state.weekly),
            data_quality=DataQualityLevel(st.session_state.data_quality),
        )
    except Exception as exc:
        st.sidebar.error(f"Config error: {exc}")
        return None, gen_clicked

    return cfg, gen_clicked


# ── Cached plot generator ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _generate_all_plots(df: pd.DataFrame, hostname: Optional[str]) -> Dict[str, bytes]:
    from validation.plots import (
        plot_cpu_timeseries, plot_memory_trends,
        plot_network_throughput, plot_disk_io, plot_anomaly_distribution,
    )
    plots: Dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tp = Path(tmpdir)
        for name, fn in [
            ("cpu",     lambda: plot_cpu_timeseries(df,    hostname=hostname, output_dir=tp)),
            ("memory",  lambda: plot_memory_trends(df,     hostname=hostname, output_dir=tp)),
            ("network", lambda: plot_network_throughput(df, hostname=hostname, output_dir=tp)),
            ("disk",    lambda: plot_disk_io(df,           hostname=hostname, output_dir=tp)),
            ("anomaly", lambda: plot_anomaly_distribution(df,                 output_dir=tp)),
        ]:
            try:
                path = fn()
                plots[name] = Path(path).read_bytes()
            except Exception:
                plots[name] = b""
    return plots


# ── Shared helper: resolve data source (generated or uploaded CSV) ────────────
def _resolve_df(tab_key: str) -> Optional[pd.DataFrame]:
    """
    Renders a data-source selector and returns the active DataFrame.
    Returns None if no data is available for the chosen source.
    """
    has_generated = st.session_state.df is not None
    source = st.radio(
        "Data source",
        ["Generated data", "Upload existing CSV"],
        horizontal=True,
        key=f"src_{tab_key}",
        help="Use data generated in this session, or upload any SAR CSV file.",
    )

    if source == "Upload existing CSV":
        uploaded = st.file_uploader(
            "SAR CSV file", type="csv", key=f"upload_{tab_key}",
            label_visibility="collapsed",
        )
        if uploaded is None:
            st.info("Upload a SAR CSV file to continue.")
            return None
        return pd.read_csv(uploaded)
    else:
        if not has_generated:
            st.info("No data generated yet. Click **Generate SAR Data** in the sidebar.")
            return None
        return st.session_state.df


# ── Tab: Overview ─────────────────────────────────────────────────────────────
def _tab_overview(cfg: Optional[SimulationConfig]) -> None:
    if cfg is None:
        st.warning("Configuration has errors — please check the sidebar.")
        return

    total_nodes = sum(n.count for n in cfg.nodes)
    est_rows    = cfg.num_intervals * total_nodes
    duration_d  = cfg.total_seconds / 86_400

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration",    f"{duration_d:.1f} days")
    c2.metric("Total Nodes", f"{total_nodes:,}")
    c3.metric("Est. Rows",   f"{est_rows:,}")
    c4.metric("Scenarios",   str(len(cfg.scenarios)))

    # ── Validate Config ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-head">Configuration Validation (--validate-only)</div>',
                unsafe_allow_html=True)

    col_v, col_e = st.columns([1, 3])
    if col_v.button("✅ Validate Config", use_container_width=True, key="btn_validate_cfg"):
        st.session_state["_cfg_valid_result"] = cfg   # already valid if we got here

    if "_cfg_valid_result" in st.session_state:
        vcfg = st.session_state["_cfg_valid_result"]
        st.success(
            f"**Configuration is valid.** "
            f"{sum(n.count for n in vcfg.nodes):,} nodes · "
            f"{vcfg.num_intervals:,} intervals · "
            f"~{vcfg.num_intervals * sum(n.count for n in vcfg.nodes):,} rows"
        )

    st.markdown("---")

    st.markdown('<div class="section-head">Node Groups</div>', unsafe_allow_html=True)
    ng_rows = [
        {
            "Type":         getattr(nc.type,              "value", str(nc.type)),
            "Count":        nc.count,
            "Base Load":    f"{nc.base_load:.0%}",
            "Anomaly Freq": getattr(nc.anomaly_frequency, "value", str(nc.anomaly_frequency)),
            "Est. Rows":    f"{cfg.num_intervals * nc.count:,}",
        }
        for nc in cfg.nodes
    ]
    st.dataframe(pd.DataFrame(ng_rows), use_container_width=True, hide_index=True)

    if cfg.scenarios:
        st.markdown('<div class="section-head">Anomaly Scenarios</div>', unsafe_allow_html=True)
        sc_rows = []
        for sc in cfg.scenarios:
            dur_h = (sc.end_time - sc.start_time).total_seconds() / 3600
            sc_rows.append({
                "Type":     getattr(sc.type,     "value", str(sc.type)),
                "Start":    sc.start_time.strftime("%Y-%m-%d %H:%M"),
                "Duration": f"{dur_h:.1f} h",
                "Severity": getattr(sc.severity, "value", str(sc.severity)),
                "Targets":  (", ".join(getattr(t, "value", str(t)) for t in sc.target_node_types)
                             if sc.target_node_types else "All nodes"),
            })
        st.dataframe(pd.DataFrame(sc_rows), use_container_width=True, hide_index=True)

    if st.session_state.gen_stats:
        gs = st.session_state.gen_stats
        st.success(
            f"Last run: **{gs['rows']:,} rows** in **{gs['elapsed']:.2f}s** "
            f"({gs['rows_per_sec']:,.0f} rows/s)"
        )
    else:
        st.info("Click **Generate SAR Data** in the sidebar to start.")


# ── Tab: Results ──────────────────────────────────────────────────────────────
def _tab_results(df: Optional[pd.DataFrame]) -> None:
    if df is None:
        st.info("No data generated yet. Click **Generate SAR Data** in the sidebar.")
        return

    gs = st.session_state.gen_stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",     f"{len(df):,}")
    c2.metric("Columns",  str(len(df.columns)))
    c3.metric("Nodes",    str(df["hostname"].nunique()) if "hostname" in df.columns else "—")
    c4.metric("Gen Time", f"{gs.get('elapsed', 0):.2f}s")

    st.markdown('<div class="section-head">Data Preview</div>', unsafe_allow_html=True)

    if "hostname" in df.columns:
        hosts    = ["All"] + sorted(df["hostname"].unique())
        sel_host = st.selectbox("Filter by hostname", hosts, key="res_host")
        show_df  = df[df["hostname"] == sel_host] if sel_host != "All" else df
    else:
        show_df = df

    st.dataframe(show_df.head(500), use_container_width=True)

    key_cols = [c for c in [
        "%usr", "%sys", "%iowait", "%steal", "%idle",
        "%memused", "%swpused", "tps", "await", "%util",
        "rxkB/s", "txkB/s", "ldavg-1",
    ] if c in df.columns]

    if key_cols:
        st.markdown('<div class="section-head">Quick Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df[key_cols].describe().round(3), use_container_width=True)


# ── Tab: Visualize ────────────────────────────────────────────────────────────
def _tab_visualize() -> None:
    df = _resolve_df("viz")
    if df is None:
        return

    hosts    = sorted(df["hostname"].unique()) if "hostname" in df.columns else []
    hostname = st.selectbox("Select Host", hosts, key="viz_host") if hosts else None

    _plot_labels = {
        "CPU Usage":            "cpu",
        "Memory Trends":        "memory",
        "Network Throughput":   "network",
        "Disk I/O":             "disk",
        "Anomaly Distribution": "anomaly",
    }

    selected = st.multiselect(
        "Charts", list(_plot_labels.keys()),
        default=list(_plot_labels.keys()), key="viz_charts",
    )

    if not selected:
        st.info("Select at least one chart above.")
        return

    with st.spinner("Rendering plots…"):
        all_plots = _generate_all_plots(df, hostname)

    for label in selected:
        key   = _plot_labels[label]
        img_b = all_plots.get(key, b"")
        if img_b:
            st.markdown(f'<div class="section-head">{label}</div>', unsafe_allow_html=True)
            st.image(img_b, use_container_width=True)
        else:
            st.warning(f"Could not render {label}.")


# ── Tab: Validate ─────────────────────────────────────────────────────────────
def _tab_validate() -> None:
    df = _resolve_df("val")
    if df is None:
        return

    c1, c2 = st.columns(2)

    if c1.button("Run Statistical Validation", use_container_width=True, key="btn_val"):
        from validation.statistics import StatisticalValidator
        with st.spinner("Validating…"):
            report = StatisticalValidator(df).run_all()
        st.session_state["_val_report"] = report

    if c2.button("Compare vs Reference Profiles", use_container_width=True, key="btn_cmp"):
        from validation.comparison import SARPatternComparator
        with st.spinner("Comparing…"):
            report = SARPatternComparator(df).compare_builtin()
        st.session_state["_cmp_report"] = report

    if "_val_report" in st.session_state:
        st.markdown("---")
        _render_validation_report(st.session_state["_val_report"])

    if "_cmp_report" in st.session_state:
        st.markdown("---")
        _render_comparison_report(st.session_state["_cmp_report"])


def _render_validation_report(report) -> None:
    status = "✅ PASSED" if report.passed else "❌ FAILED"
    d      = report.to_dict()
    total  = (len(d.get("metric_checks", [])) +
              len(d.get("correlation_checks", [])) +
              len(d.get("anomaly_checks", [])))
    passed = (sum(1 for c in d.get("metric_checks",     []) if c["passed"]) +
              sum(1 for c in d.get("correlation_checks", []) if c["passed"]) +
              sum(1 for c in d.get("anomaly_checks",     []) if c["passed"]))

    st.markdown(
        f'<div class="section-head">Statistical Validation — {status} &nbsp;({passed}/{total})</div>',
        unsafe_allow_html=True,
    )

    if d.get("metric_checks"):
        rows = [{
            "Metric":      c["metric"],
            "Pass":        "✅" if c["passed"] else "❌",
            "Actual Mean": round(c["actual_mean"], 3),
            "Actual Std":  round(c["actual_std"],  3),
            "Details":     c.get("details", ""),
        } for c in d["metric_checks"]]
        st.markdown("**Metric Checks**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if d.get("correlation_checks"):
        rows = [{
            "Metric A":  c["metric_a"],
            "Metric B":  c["metric_b"],
            "r":         round(c["correlation"], 3),
            "Threshold": c["threshold"],
            "Direction": c["direction"],
            "Pass":      "✅" if c["passed"] else "❌",
        } for c in d["correlation_checks"]]
        st.markdown("**Correlation Checks**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if d.get("anomaly_checks"):
        rows = [{
            "Host":      c["hostname"],
            "Metric":    c["metric"],
            "Anomaly %": f"{c['anomaly_fraction'] * 100:.2f}%",
            "Expected":  f"{c['expected_min']*100:.1f}–{c['expected_max']*100:.1f}%",
            "Pass":      "✅" if c["passed"] else "❌",
        } for c in d["anomaly_checks"]]
        st.markdown("**Anomaly Frequency Checks**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_comparison_report(report) -> None:
    status  = "✅ PASSED" if report.passed else "❌ FAILED"
    d       = report.to_dict()
    results = d.get("results", [])
    passed  = sum(1 for r in results if r["passed"])

    st.markdown(
        f'<div class="section-head">Reference Comparison — {status} &nbsp;({passed}/{len(results)})</div>',
        unsafe_allow_html=True,
    )

    if not results:
        st.info("No comparison results available.")
        return

    rows = [{
        "Node Type":  r["node_type"],
        "Metric":     r["metric"],
        "Synth Mean": round(r["synthetic_mean"], 2),
        "Ref Mean":   round(r["reference_mean"], 2),
        "Diff %":     f"{r['mean_diff_pct']:+.1f}%",
        "KS stat":    round(r["ks_statistic"], 3),
        "KS p-val":   round(r["ks_pvalue"], 4),
        "Pass":       "✅" if r["passed"] else "❌",
    } for r in results]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if d.get("summary"):
        st.caption(d["summary"])


# ── Tab: Download ─────────────────────────────────────────────────────────────
def _tab_download(df: Optional[pd.DataFrame], cfg: Optional[SimulationConfig]) -> None:
    if df is None:
        st.info("No data generated yet. Click **Generate SAR Data** in the sidebar.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.markdown('<div class="section-head">In-Browser Download</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇  Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"sar_synthetic_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_csv",
        )

    with c2:
        meta = {}
        if cfg:
            meta = {
                "start_time": str(cfg.start_time),
                "end_time":   str(cfg.end_time),
                "total_rows": len(df),
                "nodes":      sum(n.count for n in cfg.nodes),
            }
        json_buf = io.StringIO()
        json.dump({"metadata": meta, "data": df.to_dict(orient="records")},
                  json_buf, default=str, indent=2)
        st.download_button(
            "⬇  Download JSON",
            data=json_buf.getvalue(),
            file_name=f"sar_synthetic_{ts}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json",
        )

    st.markdown('<div class="section-head">Save to Disk</div>', unsafe_allow_html=True)

    active_cfg = cfg or st.session_state.sim_config
    if active_cfg is None:
        st.warning("No config available — regenerate data first.")
        return

    by_node = st.session_state.get("by_node", False)
    mode_label = "per-node files (--by-node)" if by_node else "single combined file"
    st.caption(
        f"Output directory: **{active_cfg.output.output_dir}** · mode: **{mode_label}**"
    )

    if st.button("💾  Save to Output Directory", use_container_width=True, key="btn_save"):
        from adapters.output import OutputManager
        Path(active_cfg.output.output_dir).mkdir(parents=True, exist_ok=True)
        with st.spinner("Writing files…"):
            mgr = OutputManager(active_cfg.output)
            if by_node:
                paths = mgr.write_by_node(df)
            else:
                paths = mgr.write(df)
        saved = [str(p) for ps in paths.values() for p in ps]
        st.success(
            f"Saved {len(saved)} file(s):\n" + "\n".join(f"  • {p}" for p in saved)
        )


# ── Tab: Benchmark ────────────────────────────────────────────────────────────
def _tab_benchmark() -> None:
    st.markdown('<div class="section-head">Performance Benchmark (benchmark.py)</div>',
                unsafe_allow_html=True)
    st.caption(
        "Measures rows/second and memory usage using chunk-based generation. "
        "Equivalent to `python benchmark.py --nodes N --days D …`"
    )

    c1, c2, c3 = st.columns(3)
    bm_nodes    = c1.number_input("Total Nodes",  1, 1000, 18,     step=1,    key="bm_nodes")
    bm_days     = c2.number_input("Days",         1, 90,   7,      step=1,    key="bm_days")
    bm_interval = c3.selectbox("Interval (sec)", _INTERVALS, index=2,          key="bm_interval")

    c4, c5 = st.columns(2)
    bm_warmup = c4.number_input("Warmup runs", 0, 5,      1, step=1,           key="bm_warmup")
    bm_chunk  = c5.number_input("Chunk size",  1000, 500_000, 50_000, step=1000, key="bm_chunk")

    if st.button("▶  Run Benchmark", type="primary", use_container_width=True, key="bm_run"):
        from benchmark.performance import PerformanceBenchmark
        with st.spinner(
            f"Benchmarking {bm_nodes} nodes × {bm_days} days "
            f"(warmup: {bm_warmup})…"
        ):
            bm = PerformanceBenchmark(
                total_nodes=int(bm_nodes),
                days=int(bm_days),
                interval_seconds=int(bm_interval),
                warmup_runs=int(bm_warmup),
                chunk_size=int(bm_chunk),
            )
            result = bm.run()
        st.session_state["_bm_result"] = result

    if "_bm_result" in st.session_state:
        _render_benchmark_result(st.session_state["_bm_result"])


def _render_benchmark_result(result) -> None:
    d = result.to_dict()

    rps = d["rows_per_second"]
    if rps >= 1_000_000:
        grade, color = "Excellent  (> 1 M rows/s)",  "#34d399"
    elif rps >= 500_000:
        grade, color = "Very Good  (> 500 K rows/s)", "#60a5fa"
    elif rps >= 100_000:
        grade, color = "Good  (> 100 K rows/s)",      "#fbbf24"
    elif rps >= 10_000:
        grade, color = "Fair  (> 10 K rows/s)",       "#f97316"
    else:
        grade, color = "Low  (< 10 K rows/s)",        "#f87171"

    st.markdown(
        f'<div class="section-head">Results — '
        f'<span style="color:{color}">{grade}</span></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows/second",  f"{rps:,.0f}")
    c2.metric("Total rows",   f"{d['total_rows']:,}")
    c3.metric("Total time",   f"{d['total_seconds']:.2f}s")
    c4.metric("Peak memory",  f"{d['peak_memory_mb']:.1f} MB")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg memory",   f"{d['avg_memory_mb']:.1f} MB")
    c6.metric("Chunks",       str(len(d.get("chunk_times_s", []))))
    c7.metric("Chunk p50",    f"{result.p50_chunk_s * 1000:.1f} ms")
    c8.metric("Chunk p95",    f"{result.p95_chunk_s * 1000:.1f} ms")

    # Config echo
    st.caption(
        f"Config: {d['total_nodes']} nodes · {d['days']} days · "
        f"{d['interval_seconds']}s interval · {d['warmup_runs']} warmup run(s)"
    )

    # Download result JSON
    st.download_button(
        "⬇  Download Benchmark JSON",
        data=json.dumps(d, indent=2, default=str),
        file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key="dl_bm",
    )


# ── Main layout ───────────────────────────────────────────────────────────────
def main() -> None:
    st.markdown("""
    <div class="hero">
        <h1>📊 SAR Generator</h1>
        <p>Synthetic SAR log-data generator for <strong>Telco Cloud</strong> environments
           (OpenStack&thinsp;/&thinsp;Ceph) &mdash; full-featured GUI
           &nbsp;|&nbsp; CLI: <code>python main.py</code>
           &nbsp;|&nbsp; Benchmark: <code>python benchmark.py</code></p>
    </div>
    """, unsafe_allow_html=True)

    cfg, gen_clicked = _render_sidebar()

    # ── Apply verbose logging ─────────────────────────────────────────────
    if st.session_state.get("verbose"):
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # ── Run generation ────────────────────────────────────────────────────
    if gen_clicked:
        if cfg is None:
            st.error("Fix configuration errors in the sidebar before generating.")
        else:
            from engine.generator import SARDataGenerator

            _streamer = None
            if st.session_state.get("stream_enabled"):
                try:
                    from streaming.streamer import SARStreamer, StreamingConfig
                    _stream_cfg = StreamingConfig(
                        enabled=True,
                        mode=st.session_state.get("stream_mode", "stdout"),
                        host=st.session_state.get("stream_host", "0.0.0.0"),
                        port=int(st.session_state.get("stream_port", 9000)),
                        record_format=st.session_state.get("stream_fmt", "json"),
                        flush_interval_rows=int(st.session_state.get("stream_rows", 100)),
                    )
                    _streamer = SARStreamer(_stream_cfg)
                    _streamer.start()
                    mode = st.session_state.get("stream_mode", "stdout")
                    st.info(f"Streaming active — mode: **{mode}**")
                except Exception as exc:
                    st.warning(f"Streaming setup failed: {exc}")
                    _streamer = None

            with st.spinner("⚙  Generating SAR data — please wait…"):
                t0 = time.perf_counter()
                try:
                    df      = SARDataGenerator(cfg).generate_all()
                    elapsed = time.perf_counter() - t0

                    if _streamer:
                        _streamer.send_chunk(df)
                        _streamer.stop()

                    st.session_state.df         = df
                    st.session_state.sim_config = cfg
                    st.session_state.gen_stats  = {
                        "rows":         len(df),
                        "elapsed":      elapsed,
                        "rows_per_sec": len(df) / max(elapsed, 0.001),
                    }
                    _generate_all_plots.clear()
                    for _k in ("_val_report", "_cmp_report"):
                        st.session_state.pop(_k, None)

                    st.toast(f"Generated {len(df):,} rows in {elapsed:.1f}s", icon="✅")
                except Exception as exc:
                    if _streamer:
                        try: _streamer.stop()
                        except Exception: pass
                    st.error(f"Generation failed: {exc}")

    # ── Tabs ──────────────────────────────────────────────────────────────
    t_ov, t_res, t_viz, t_val, t_dl, t_bm = st.tabs([
        "📋 Overview",
        "📊 Results",
        "📈 Visualize",
        "🔍 Validate",
        "⬇ Download",
        "🏎 Benchmark",
    ])

    with t_ov:  _tab_overview(cfg)
    with t_res: _tab_results(st.session_state.df)
    with t_viz: _tab_visualize()
    with t_val: _tab_validate()
    with t_dl:  _tab_download(st.session_state.df, cfg or st.session_state.sim_config)
    with t_bm:  _tab_benchmark()


if __name__ == "__main__":
    main()
