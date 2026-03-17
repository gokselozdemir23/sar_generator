#!/usr/bin/env python3
"""
Synthetic SAR Log Data Generator - Main Entry Point
Telco Cloud (OpenStack / CEPH) SAR metrics simulation.

Usage:
    python main.py                             # Use built-in default config
    python main.py --config config.yaml        # Use YAML config file
    python main.py --config config.yaml --output-dir ./my_output
    python main.py --write-example-config      # Dump example YAML config
    python main.py --validate-only             # Validate config without generating
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sar_generator")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=False),
    default=None,
    help="Path to YAML/JSON simulation config file. Uses built-in defaults if omitted.",
)
@click.option(
    "--output-dir", "-o",
    default=None,
    help="Override output directory from config.",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["csv", "json", "both"], case_sensitive=False),
    default=None,
    help="Override output format from config.",
)
@click.option(
    "--write-example-config",
    is_flag=True,
    default=False,
    help="Write an example YAML config to 'config_example.yaml' and exit.",
)
@click.option(
    "--validate-only",
    is_flag=True,
    default=False,
    help="Validate configuration only, do not generate data.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable DEBUG logging.",
)
@click.option(
    "--by-node",
    is_flag=True,
    default=False,
    help="Write one file per node instead of a combined file.",
)
@click.option(
    "--stream",
    is_flag=True,
    default=False,
    help="Stream output in real-time (config.streaming must be enabled).",
)
@click.option(
    "--visualize",
    default=None,
    metavar="CSV_PATH",
    help="Visualize a generated CSV file and save plots to ./plots.",
)
@click.option(
    "--validate-data",
    default=None,
    metavar="CSV_PATH",
    help="Run statistical validation on a generated CSV file.",
)
@click.option(
    "--plots-dir",
    default="./plots",
    help="Directory for --visualize output (default: ./plots).",
)
@click.option(
    "--compare-reference",
    default=None,
    metavar="CSV_PATH",
    help="Compare generated CSV against real SAR reference profiles.",
)
def main(
    config: Optional[str],
    output_dir: Optional[str],
    format: Optional[str],
    write_example_config: bool,
    validate_only: bool,
    verbose: bool,
    by_node: bool,
    stream: bool,
    visualize: Optional[str],
    validate_data: Optional[str],
    plots_dir: str,
    compare_reference: Optional[str],
) -> None:
    """Synthetic SAR Log Data Generator for Telco Cloud environments."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # -- Standalone: visualize a CSV file --
    if visualize:
        try:
            from validation.plots import generate_all_plots
            click.echo(f"🖼️  Grafik üretiliyor: {visualize} → {plots_dir}")
            paths = generate_all_plots(visualize, plots_dir)
            for p in paths:
                click.echo(f"  ✓ {p}")
            click.echo(f"\n{len(paths)} grafik üretildi.")
        except ImportError as exc:
            click.echo(f"❌ {exc}", err=True)
        sys.exit(0)

    # -- Standalone: validate a CSV file --
    if validate_data:
        import pandas as _pd
        from validation.statistics import StatisticalValidator
        click.echo(f"🔍 Doğrulama: {validate_data}")
        _df = _pd.read_csv(validate_data)
        report = StatisticalValidator(_df).run_all()
        report.print_summary()
        sys.exit(0 if report.passed else 1)

    # -- Standalone: compare against reference SAR patterns --
    if compare_reference:
        import pandas as _pd
        from validation.comparison import SARPatternComparator
        click.echo(f"📊 Karşılaştırma: {compare_reference}")
        _df = _pd.read_csv(compare_reference)
        report = SARPatternComparator(_df).compare_builtin()
        report.print_summary()
        sys.exit(0 if report.passed else 1)

    # -- Write example config and exit --
    if write_example_config:
        from config import write_example_config as _write
        _write("config_example.yaml")
        click.echo("Example config written to config_example.yaml")
        sys.exit(0)

    # -- Load configuration --
    if config:
        from config import load_config
        click.echo(f"Loading config from: {config}")
        sim_config = load_config(config)
    else:
        from config import get_default_config
        click.echo("No config specified - using built-in defaults.")
        sim_config = get_default_config()

    # -- Apply CLI overrides --
    if output_dir:
        sim_config.output.output_dir = output_dir
    if format:
        sim_config.output.format = format

    # -- Print simulation summary --
    _print_simulation_summary(sim_config)

    # -- Validate only --
    if validate_only:
        click.echo("\n✅ Configuration is valid.")
        sys.exit(0)

    # -- Generate data --
    click.echo("\n🚀 Starting data generation...")
    t0 = time.perf_counter()

    from engine.generator import SARDataGenerator
    from adapters.output import OutputManager

    generator = SARDataGenerator(sim_config)
    output_mgr = OutputManager(sim_config.output)

    # -- Setup optional streamer --
    _streamer = None
    _raw_cfg = None
    if stream:
        try:
            import yaml as _yaml
            if config:
                with open(config) as _f:
                    _raw_cfg = _yaml.safe_load(_f)
            if _raw_cfg and "streaming" in _raw_cfg:
                from streaming.streamer import SARStreamer, StreamingConfig
                _stream_cfg = StreamingConfig(**_raw_cfg["streaming"])
                _streamer = SARStreamer(_stream_cfg)
                _streamer.start()
                click.echo(f"📡 Streaming aktif: {_stream_cfg.mode}")
            else:
                click.echo("⚠️  --stream verildi ama config'de [streaming] bölümü yok.", err=True)
        except ImportError as exc:
            click.echo(f"❌ Streaming modülü yüklenemedi: {exc}", err=True)

    if by_node:
        # Per-node file output via chunk iteration
        all_paths = {'csv': [], 'json': []}
        for chunk_df in generator.generate_chunks(sim_config.output.chunk_size):
            results = output_mgr.write_by_node(chunk_df)
            for fmt, paths in results.items():
                all_paths.setdefault(fmt, []).extend(paths)
            if _streamer:
                _streamer.send_chunk(chunk_df)
    else:
        # Single combined output
        df = generator.generate_all()
        elapsed_gen = time.perf_counter() - t0
        rows_per_sec = len(df) / max(elapsed_gen, 0.001)
        click.echo(
            f"\n✅ Generated {len(df):,} rows in {elapsed_gen:.2f}s "
            f"({rows_per_sec:,.0f} rows/s)"
        )
        if _streamer:
            _streamer.send_chunk(df)

        click.echo("💾 Writing output files...")
        all_paths = output_mgr.write(df)

    if _streamer:
        _streamer.stop()

    elapsed_total = time.perf_counter() - t0
    output_mgr.print_summary(all_paths)
    click.echo(f"⏱️  Total time: {elapsed_total:.2f}s")
    click.echo("✅ Done.")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_simulation_summary(config) -> None:
    duration_days = config.total_seconds / 86400
    total_nodes = sum(n.count for n in config.nodes)
    total_rows = config.num_intervals * total_nodes

    def ev(e):
        return e.value if hasattr(e, "value") else str(e)

    click.echo("\n=== Simulation Configuration ===")
    click.echo(f"  Period    : {config.start_time} → {config.end_time}  ({duration_days:.1f} days)")
    click.echo(f"  Interval  : {config.interval_seconds}s  ({config.num_intervals:,} timestamps)")
    click.echo(f"  Nodes     : {total_nodes}  (across {len(config.nodes)} node groups)")
    click.echo(f"  Scenarios : {len(config.scenarios)}")
    click.echo(f"  Est. rows : {total_rows:,}")
    click.echo(f"  Output    : {config.output.output_dir}  [{ev(config.output.format)}]")
    if config.description:
        click.echo(f"  Desc.     : {config.description}")
    click.echo()

    click.echo("Node Groups:")
    for nc in config.nodes:
        click.echo(
            f"  {ev(nc.type):<18} x{nc.count:<4}  base_load={nc.base_load}  "
            f"anomaly={ev(nc.anomaly_frequency)}"
        )

    if config.scenarios:
        click.echo("\nScenarios:")
        for sc in config.scenarios:
            targets = [ev(t) for t in sc.target_node_types] if sc.target_node_types else ["all"]
            click.echo(
                f"  {ev(sc.type):<25} {sc.start_time}  "
                f"severity={ev(sc.severity)}  targets={targets}"
            )
    click.echo("================================")


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------

def run_simulation(config=None) -> "pd.DataFrame":
    """
    Programmatic entry point. Returns generated DataFrame.

    Example:
        from main import run_simulation
        from config import get_default_config
        df = run_simulation(get_default_config())
    """
    import pandas as pd
    from engine.generator import SARDataGenerator

    if config is None:
        from config import get_default_config
        config = get_default_config()

    gen = SARDataGenerator(config)
    return gen.generate_all()


if __name__ == "__main__":
    main()
