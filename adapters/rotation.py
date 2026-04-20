"""
Kullanım (config.yaml):
    output:
      format: csv
      rotation:
        strategy: size
        max_mb: 100
        interval_minutes: 60
        compress: true
"""
from __future__ import annotations

import gzip
import logging
import os
import shutil
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import IO, Iterator, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class RotationStrategy(str, Enum):
    SIZE   = "size"
    TIME   = "time"
    HYBRID = "hybrid"


class RotationConfig(BaseModel):
    """Dosya rotasyon yapılandırması."""
    model_config = ConfigDict(use_enum_values=False)

    strategy:         RotationStrategy = RotationStrategy.SIZE
    max_mb:           float            = Field(default=100.0,  gt=0)
    interval_minutes: int              = Field(default=60,     ge=1)
    compress:         bool             = False
    keep_last_n:      Optional[int]    = Field(default=None,   ge=1)


# Rotating File Handle

class RotatingFileHandle:
    """
    Tek bir dosyayı temsil eder; rotasyon eşiği aşılınca otomatik olarak
    yeni bir dosya açar.

    Boyut ve zaman kontrolünü iç içe yönetir; dışarıdan sadece
    ``write(data)`` çağrısı yeterlidir.
    """

    def __init__(
        self,
        base_path: Path,
        config: RotationConfig,
        extension: str = ".csv",
    ):
        self._base     = base_path
        self._config   = config
        self._ext      = extension
        self._index    = 0
        self._size_b   = 0
        self._opened_at= time.monotonic()
        self._fh: Optional[IO] = None
        self._rotated_files: List[Path] = []
        self._open_new()


    @property
    def current_path(self) -> Path:
        """Şu an yazılan dosyanın yolu."""
        return self._current_path

    @property
    def rotated_files(self) -> List[Path]:
        """Rotation tamamlanmış (kapalı) dosyalar."""
        return list(self._rotated_files)


    def write(self, data: Union[str, bytes]) -> None:
        """Veriyi yaz; gerekirse önce rotasyon tetikle."""
        if self._should_rotate():
            self._rotate()

        raw = data.encode() if isinstance(data, str) else data
        self._fh.write(raw)  # type: ignore[arg-type]
        self._size_b += len(raw)

    def flush(self) -> None:
        """Tamponu diske boşalt."""
        if self._fh:
            self._fh.flush()

    def close(self) -> None:
        """Mevcut dosyayı kapat; sıkıştırma varsa uygula."""
        self._close_current()


    def _build_path(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self._base.stem}_{ts}_{self._index:04d}{self._ext}"
        return self._base.parent / name

    def _open_new(self) -> None:
        self._current_path = self._build_path()
        self._current_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._current_path, "ab")
        self._size_b = 0
        self._opened_at = time.monotonic()
        logger.debug("Yeni rotasyon dosyası açıldı: %s", self._current_path)

    def _close_current(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
        path = self._current_path
        if self._config.compress and path.exists():
            gz_path = Path(str(path) + ".gz")
            with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            path.unlink()
            self._rotated_files.append(gz_path)
            logger.debug("Sıkıştırıldı: %s → %s", path, gz_path)
        elif path.exists():
            self._rotated_files.append(path)

    def _should_rotate(self) -> bool:
        strategy = self._config.strategy
        max_b = self._config.max_mb * 1024 * 1024
        max_s = self._config.interval_minutes * 60
        elapsed = time.monotonic() - self._opened_at

        if strategy == RotationStrategy.SIZE:
            return self._size_b >= max_b
        elif strategy == RotationStrategy.TIME:
            return elapsed >= max_s
        else:  # HYBRID
            return self._size_b >= max_b or elapsed >= max_s

    def _rotate(self) -> None:
        logger.info(
            "Rotasyon: %s (%.1f MB / %.1f dk)",
            self._current_path,
            self._size_b / (1024 * 1024),
            (time.monotonic() - self._opened_at) / 60,
        )
        self._close_current()
        self._index += 1
        self._open_new()
        self._prune_old_files()

    def _prune_old_files(self) -> None:
        """keep_last_n varsa eskilerini sil."""
        if self._config.keep_last_n is None:
            return
        while len(self._rotated_files) > self._config.keep_last_n:
            old = self._rotated_files.pop(0)
            try:
                old.unlink(missing_ok=True)
                logger.info("Eski rotasyon dosyası silindi: %s", old)
            except OSError as exc:
                logger.warning("Silinemedi: %s — %s", old, exc)


# Rotating CSV Adapter

class RotatingCSVAdapter:
    """
    Rotasyon destekli CSV yazıcı.

    Mevcut CSVOutputAdapter'ın drop-in alternatifi;
    aynı ``write(df)`` arayüzünü sunar.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        filename_prefix: str,
        rotation_config: RotationConfig,
        float_fmt: str = "%.2f",
    ):
        self._dir      = Path(output_dir)
        self._prefix   = filename_prefix
        self._rot_cfg  = rotation_config
        self._float_fmt = float_fmt
        self._handle: Optional[RotatingFileHandle] = None
        self._header_written = False


    def open(self) -> None:
        """Dosya handle'ı başlat."""
        base = self._dir / self._prefix
        self._handle = RotatingFileHandle(base, self._rot_cfg, ".csv")
        self._header_written = False

    def write(self, df: pd.DataFrame) -> None:
        """
        DataFrame chunk'ını rotasyon-destekli dosyaya yaz.

        Args:
            df: Yazılacak SAR DataFrame chunk'ı.
        """
        if self._handle is None:
            raise RuntimeError("open() çağrılmadan write() kullanılamaz.")

        buf = _df_to_csv_buf(df, header=not self._header_written)
        self._handle.write(buf)
        self._header_written = True

    def flush(self) -> None:
        """Tamponu diske boşalt."""
        if self._handle:
            self._handle.flush()

    def close(self) -> Tuple[Path, List[Path]]:
        """
        Yazar'ı kapat.

        Returns:
            (aktif_dosya, rotasyon_tamamlanmış_dosyalar)
        """
        if self._handle:
            self._handle.close()
            return self._handle.current_path, self._handle.rotated_files
        return Path("/dev/null"), []

    def __enter__(self) -> "RotatingCSVAdapter":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()


# Rotating JSON (NDJSON) Adapter

class RotatingJSONAdapter:
    """
    Rotasyon destekli NDJSON (newline-delimited JSON) yazıcı.
    Her satır ayrı bir JSON nesnesi.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        filename_prefix: str,
        rotation_config: RotationConfig,
    ):
        self._dir     = Path(output_dir)
        self._prefix  = filename_prefix
        self._rot_cfg = rotation_config
        self._handle: Optional[RotatingFileHandle] = None

    def open(self) -> None:
        """Dosya handle'ı başlat."""
        base = self._dir / self._prefix
        self._handle = RotatingFileHandle(base, self._rot_cfg, ".ndjson")

    def write(self, df: pd.DataFrame) -> None:
        """
        DataFrame chunk'ını NDJSON formatında rotasyonlu dosyaya yaz.

        Args:
            df: Yazılacak SAR DataFrame chunk'ı.
        """
        if self._handle is None:
            raise RuntimeError("open() çağrılmadan write() kullanılamaz.")

        import json
        import numpy as np

        def _default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return str(o)

        lines = []
        for _, row in df.iterrows():
            lines.append(json.dumps(dict(row), default=_default))

        self._handle.write(("\n".join(lines) + "\n"))

    def flush(self) -> None:
        if self._handle:
            self._handle.flush()

    def close(self) -> Tuple[Path, List[Path]]:
        if self._handle:
            self._handle.close()
            return self._handle.current_path, self._handle.rotated_files
        return Path("/dev/null"), []

    def __enter__(self) -> "RotatingJSONAdapter":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()


# Helper

def _df_to_csv_buf(df: pd.DataFrame, header: bool = True) -> str:
    """DataFrame'i CSV string'e dönüştür."""
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=header, float_format="%.2f")
    return buf.getvalue()
