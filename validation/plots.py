"""
Visualization Tools – SAR veri setini matplotlib ile görselleştirir.

Mevcut pipeline'a dokunmaz; üretilmiş CSV/DataFrame'i girdi olarak alır.

CLI:
    python main.py --visualize output/sar_synthetic_full.csv
    python -m validation.plots output/sar_synthetic_full.csv --output-dir ./plots

Gereksinim: pip install matplotlib
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# matplotlib'i lazy import et (isteğe bağlı bağımlılık)
def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless / non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        return plt, mdates
    except ImportError as exc:
        raise ImportError(
            "matplotlib gerekli: pip install matplotlib"
        ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_df(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """CSV dosyası veya DataFrame kabul eder."""
    if isinstance(source, pd.DataFrame):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    return pd.read_csv(path)


def _parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """DateTime kolonunu parse et ve index olarak ata."""
    df = df.copy()
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df


def _save_or_show(plt, path: Optional[Path], filename: str) -> Path:
    """Dosyaya kaydet ya da göster."""
    if path:
        out = path / filename
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Grafik kaydedildi: %s", out)
        return out
    else:
        plt.tight_layout()
        plt.show()
        return Path(filename)


# ---------------------------------------------------------------------------
# Plot Functions
# ---------------------------------------------------------------------------

def plot_cpu_timeseries(
    source: Union[str, Path, pd.DataFrame],
    hostname: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> Path:
    """
    CPU kullanım zaman serisi grafiği (%usr, %sys, %iowait, %steal, %idle).

    Args:
        source:     CSV dosya yolu veya DataFrame.
        hostname:   Belirli bir host için filtrele. None ise ilk host seçilir.
        output_dir: Grafik kaydedileceği dizin. None ise ekranda gösterilir.
        figsize:    Grafik boyutu (genişlik, yükseklik) inç.

    Returns:
        Kaydedilen dosyanın yolu.
    """
    plt, mdates = _require_matplotlib()
    df = _parse_datetime(_load_df(source))

    if hostname is None and "hostname" in df.columns:
        hostname = df["hostname"].iloc[0]
    if hostname:
        df = df[df["hostname"] == hostname]

    fig, ax = plt.subplots(figsize=figsize)
    cpu_cols = [c for c in ["%usr", "%sys", "%iowait", "%steal", "%idle"] if c in df.columns]

    for col in cpu_cols:
        ax.plot(df["DateTime"], df[col], label=col, linewidth=0.8)

    ax.set_title(f"CPU Kullanımı — {hostname or 'Tüm Hostlar'}")
    ax.set_xlabel("Zaman")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)

    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"cpu_timeseries_{hostname or 'all'}.png"
    return _save_or_show(plt, out_dir, fname)


def plot_memory_trends(
    source: Union[str, Path, pd.DataFrame],
    hostname: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> Path:
    """
    Bellek kullanım trendi grafiği (%memused, %swpused).

    Args:
        source:     CSV dosya yolu veya DataFrame.
        hostname:   Belirli bir host için filtrele.
        output_dir: Grafik kaydedileceği dizin.
        figsize:    Grafik boyutu.

    Returns:
        Kaydedilen dosyanın yolu.
    """
    plt, mdates = _require_matplotlib()
    df = _parse_datetime(_load_df(source))

    if hostname is None and "hostname" in df.columns:
        hostname = df["hostname"].iloc[0]
    if hostname:
        df = df[df["hostname"] == hostname]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # RAM
    if "%memused" in df.columns:
        axes[0].fill_between(df["DateTime"], df["%memused"], alpha=0.5, color="steelblue")
        axes[0].plot(df["DateTime"], df["%memused"], color="steelblue", linewidth=0.8)
        axes[0].set_ylabel("%memused")
        axes[0].set_ylim(0, 105)
        axes[0].set_title(f"Bellek Kullanımı — {hostname or 'Tüm Hostlar'}")
        axes[0].grid(True, alpha=0.3)

    # Swap
    if "%swpused" in df.columns:
        axes[1].fill_between(df["DateTime"], df["%swpused"], alpha=0.5, color="coral")
        axes[1].plot(df["DateTime"], df["%swpused"], color="coral", linewidth=0.8)
        axes[1].set_ylabel("%swpused")
        axes[1].set_ylim(0, 105)
        axes[1].grid(True, alpha=0.3)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate()

    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"memory_trends_{hostname or 'all'}.png"
    return _save_or_show(plt, out_dir, fname)


def plot_network_throughput(
    source: Union[str, Path, pd.DataFrame],
    hostname: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> Path:
    """
    Ağ throughput grafiği (rxkB/s ve txkB/s).

    Args:
        source:     CSV dosya yolu veya DataFrame.
        hostname:   Belirli bir host için filtrele.
        output_dir: Grafik kaydedileceği dizin.
        figsize:    Grafik boyutu.

    Returns:
        Kaydedilen dosyanın yolu.
    """
    plt, mdates = _require_matplotlib()
    df = _parse_datetime(_load_df(source))

    if hostname is None and "hostname" in df.columns:
        hostname = df["hostname"].iloc[0]
    if hostname:
        df = df[df["hostname"] == hostname]

    fig, ax = plt.subplots(figsize=figsize)

    if "rxkB/s" in df.columns:
        ax.plot(df["DateTime"], df["rxkB/s"], label="rxkB/s", color="green", linewidth=0.8)
    if "txkB/s" in df.columns:
        ax.plot(df["DateTime"], df["txkB/s"], label="txkB/s", color="orange", linewidth=0.8)

    ax.set_title(f"Ağ Throughput — {hostname or 'Tüm Hostlar'}")
    ax.set_xlabel("Zaman")
    ax.set_ylabel("kB/s")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)

    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"network_throughput_{hostname or 'all'}.png"
    return _save_or_show(plt, out_dir, fname)


def plot_anomaly_distribution(
    source: Union[str, Path, pd.DataFrame],
    metrics: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> Path:
    """
    Seçili metriklerin kutu grafiği (boxplot) — anomali dağılımını görselleştirir.

    Args:
        source:     CSV dosya yolu veya DataFrame.
        metrics:    Görselleştirilecek metrik listesi.
        output_dir: Grafik kaydedileceği dizin.
        figsize:    Grafik boyutu.

    Returns:
        Kaydedilen dosyanın yolu.
    """
    plt, _ = _require_matplotlib()
    df = _load_df(source)

    default_metrics = [
        m for m in ["%iowait", "%steal", "await", "%util", "ldavg-1", "%memused"]
        if m in df.columns
    ]
    targets = metrics or default_metrics

    fig, axes = plt.subplots(1, len(targets), figsize=figsize, sharey=False)
    if len(targets) == 1:
        axes = [axes]

    for ax, metric in zip(axes, targets):
        data = df[metric].dropna()
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="lightblue", alpha=0.7))
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Anomali Dağılımı (Boxplot)", fontsize=12, y=1.02)

    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    fname = "anomaly_distribution.png"
    return _save_or_show(plt, out_dir, fname)


def plot_disk_io(
    source: Union[str, Path, pd.DataFrame],
    hostname: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 7),
) -> Path:
    """
    Disk I/O grafiği (tps, await, %util).

    Args:
        source:     CSV dosya yolu veya DataFrame.
        hostname:   Belirli bir host için filtrele.
        output_dir: Grafik kaydedileceği dizin.
        figsize:    Grafik boyutu.

    Returns:
        Kaydedilen dosyanın yolu.
    """
    plt, mdates = _require_matplotlib()
    df = _parse_datetime(_load_df(source))

    if hostname is None and "hostname" in df.columns:
        hostname = df["hostname"].iloc[0]
    if hostname:
        df = df[df["hostname"] == hostname]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for ax, (col, color, ylabel) in zip(axes, [
        ("tps",   "royalblue", "tps"),
        ("await", "tomato",    "await (ms)"),
        ("%util", "purple",    "%util"),
    ]):
        if col in df.columns:
            ax.plot(df["DateTime"], df[col], color=color, linewidth=0.8)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    axes[0].set_title(f"Disk I/O — {hostname or 'Tüm Hostlar'}")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate()

    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"disk_io_{hostname or 'all'}.png"
    return _save_or_show(plt, out_dir, fname)


def generate_all_plots(
    source: Union[str, Path, pd.DataFrame],
    output_dir: Union[str, Path],
    hostname: Optional[str] = None,
) -> List[Path]:
    """
    Tüm grafikleri üretir ve output_dir'e kaydeder.

    Args:
        source:     CSV dosya yolu veya DataFrame.
        output_dir: Grafiklerin kaydedileceği dizin.
        hostname:   Belirli bir host için filtrele (None ise ilk host).

    Returns:
        Üretilen dosya yollarının listesi.
    """
    out_dir = Path(output_dir)
    paths: List[Path] = []

    # Functions that accept hostname parameter
    host_funcs = [
        plot_cpu_timeseries,
        plot_memory_trends,
        plot_network_throughput,
        plot_disk_io,
    ]
    for fn in host_funcs:
        try:
            p = fn(source=source, output_dir=out_dir, hostname=hostname)
            paths.append(p)
        except Exception as exc:
            logger.warning("%s grafiği üretilirken hata: %s", fn.__name__, exc)

    # Anomaly distribution does not filter by hostname (shows all)
    try:
        p = plot_anomaly_distribution(source=source, output_dir=out_dir)
        paths.append(p)
    except Exception as exc:
        logger.warning("plot_anomaly_distribution grafiği üretilirken hata: %s", exc)

    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """python -m validation.plots <csv> [--output-dir ./plots] [--host hostname]"""
    import argparse

    parser = argparse.ArgumentParser(description="SAR veri görselleştirme aracı")
    parser.add_argument("csv_path", help="SAR CSV dosyası")
    parser.add_argument("--output-dir", default="./plots", help="Grafik çıktı dizini")
    parser.add_argument("--host", default=None, help="Filtre için hostname")
    args = parser.parse_args()

    print(f"Grafik üretiliyor: {args.csv_path} → {args.output_dir}")
    paths = generate_all_plots(args.csv_path, args.output_dir, hostname=args.host)
    for p in paths:
        print(f"  ✓ {p}")
    print(f"\n{len(paths)} grafik üretildi.")


if __name__ == "__main__":
    _cli_main()
