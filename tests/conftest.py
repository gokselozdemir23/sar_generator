"""
Shared pytest configuration and markers.
Mevcut testlere dokunmaz; yalnızca ortak yapılandırma sağlar.
"""
import sys
from pathlib import Path

# Proje kökünü path'e ekle (tüm testlerde geçerli)
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Özel pytest marker tanımlamaları."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow-running (deselect with -m 'not slow')"
    )
