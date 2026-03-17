#!/usr/bin/env python3
"""
SAR Generator – Performans Benchmark CLI

Kullanım:
    python benchmark.py
    python benchmark.py --nodes 100 --days 7
    python benchmark.py --nodes 50 --days 1 --interval 60
    python benchmark.py --json --out result.json
"""
import sys
from pathlib import Path

# Proje kökünü PYTHONPATH'e ekle
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.performance import _cli_main

if __name__ == "__main__":
    _cli_main()
