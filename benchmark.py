#!/usr/bin/env python3
import sys
from pathlib import Path

# Proje kökünü PYTHONPATH'e ekle
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.performance import _cli_main

if __name__ == "__main__":
    _cli_main()
