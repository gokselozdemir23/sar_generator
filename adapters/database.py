"""
Database Output Adapters – Pluggable writers for InfluxDB, PostgreSQL ve Prometheus.

Her adapter, mevcut CSV/JSON pipeline'ına dokunmadan bağımsız çalışır.
Adapter'lar lazy import kullanır: driver kurulu değilse yalnızca kullanımda hata verir.

Kullanım (config.yaml):
    database:
      influxdb:
        enabled: true
        host: localhost
        port: 8086
        token: "my-token"
        org: "my-org"
        bucket: "sar_metrics"
        batch_size: 5000
      postgresql:
        enabled: true
        host: localhost
        port: 5432
        database: sar_metrics
        user: postgres
        password: secret
        table: sar_data
        pool_size: 5
      prometheus:
        enabled: true
        port: 9100
        prefix: sar
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config Models
# ---------------------------------------------------------------------------

class InfluxDBConfig(BaseModel):
    """InfluxDB v2 bağlantı yapılandırması."""
    model_config = ConfigDict(use_enum_values=False)

    enabled:    bool = False
    host:       str  = "localhost"
    port:       int  = Field(default=8086, ge=1, le=65535)
    token:      str  = ""
    org:        str  = "my-org"
    bucket:     str  = "sar_metrics"
    batch_size: int  = Field(default=5_000, ge=1)
    retries:    int  = Field(default=3, ge=0)
    timeout_s:  int  = Field(default=10, ge=1)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class PostgreSQLConfig(BaseModel):
    """PostgreSQL bağlantı yapılandırması."""
    model_config = ConfigDict(use_enum_values=False)

    enabled:   bool = False
    host:      str  = "localhost"
    port:      int  = Field(default=5432, ge=1, le=65535)
    database:  str  = "sar_metrics"
    user:      str  = "postgres"
    password:  str  = ""
    table:     str  = "sar_data"
    pool_size: int  = Field(default=5, ge=1, le=50)
    retries:   int  = Field(default=3, ge=0)
    timeout_s: int  = Field(default=10, ge=1)


class PrometheusConfig(BaseModel):
    """Prometheus exporter yapılandırması."""
    model_config = ConfigDict(use_enum_values=False)

    enabled: bool = False
    port:    int  = 9100
    prefix:  str  = "sar"
    # Hangi metrikler expose edilecek (boşsa hepsi)
    metrics: Optional[List[str]] = None

    def __init__(self, **kw):
        super().__init__(**kw)
        if self.metrics is None:
            self.metrics = []


class DatabaseConfig(BaseModel):
    """Tüm database adapter'larının üst yapılandırması."""
    model_config = ConfigDict(use_enum_values=False)

    influxdb:   Optional[InfluxDBConfig]   = None
    postgresql: Optional[PostgreSQLConfig] = None
    prometheus: Optional[PrometheusConfig] = None

    def __init__(self, **kw):
        super().__init__(**kw)
        if self.influxdb is None:
            self.influxdb = InfluxDBConfig()
        if self.postgresql is None:
            self.postgresql = PostgreSQLConfig()
        if self.prometheus is None:
            self.prometheus = PrometheusConfig()


# ---------------------------------------------------------------------------
# Base Writer Interface
# ---------------------------------------------------------------------------

class BaseWriter(ABC):
    """Tüm database writer'larının uyması gereken arayüz."""

    def __init__(self, config: Any):
        self.config = config
        self._connected = False

    @abstractmethod
    def connect(self) -> None:
        """Bağlantı kur. Bağlanamasa RuntimeError fırlatır."""

    @abstractmethod
    def write(self, df: pd.DataFrame) -> int:
        """
        DataFrame'i hedefe yaz.

        Returns:
            Yazılan satır sayısı.
        """

    @abstractmethod
    def close(self) -> None:
        """Bağlantıyı kapat ve kaynakları serbest bırak."""

    def write_with_retry(self, df: pd.DataFrame) -> int:
        """
        Retry logic ile yazma. Her deneme arasında üstel bekleme uygular.

        Returns:
            Başarıyla yazılan satır sayısı.
        Raises:
            RuntimeError: Tüm denemeler başarısız olursa.
        """
        retries = getattr(self.config, "retries", 3)
        last_exc: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                return self.write(df)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "%s write attempt %d/%d failed: %s — retrying in %ds",
                    type(self).__name__, attempt + 1, retries + 1, exc, wait,
                )
                if attempt < retries:
                    time.sleep(wait)

        raise RuntimeError(
            f"{type(self).__name__} failed after {retries + 1} attempts"
        ) from last_exc

    def __enter__(self) -> "BaseWriter":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# InfluxDB Writer
# ---------------------------------------------------------------------------

class InfluxDBWriter(BaseWriter):
    """
    InfluxDB v2 (line-protocol) writer.

    Gereksinim: pip install influxdb-client
    Her SAR satırı bir InfluxDB Point'e dönüştürülür.
    Measurement adı: hostname (örn. "compute-01").
    """

    # SAR kolonlarından InfluxDB tag olacaklar
    _TAG_COLUMNS = {"hostname", "CPU"}
    # Sayısal olmayan kolonlar (skip)
    _SKIP_COLUMNS = {"DateTime"} | _TAG_COLUMNS

    def __init__(self, config: InfluxDBConfig):
        super().__init__(config)
        self._client = None
        self._write_api = None

    def connect(self) -> None:
        """InfluxDB bağlantısını kur."""
        try:
            from influxdb_client import InfluxDBClient  # type: ignore
            from influxdb_client.client.write_api import SYNCHRONOUS  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "influxdb-client paketi gerekli: pip install influxdb-client"
            ) from exc

        self._client = InfluxDBClient(
            url=self.config.url,
            token=self.config.token,
            org=self.config.org,
            timeout=self.config.timeout_s * 1_000,
        )
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
        self._connected = True
        logger.info("InfluxDB bağlandı: %s / %s", self.config.url, self.config.bucket)

    def write(self, df: pd.DataFrame) -> int:
        """DataFrame'i batch'ler halinde InfluxDB'ye yaz."""
        if not self._connected:
            raise RuntimeError("connect() çağrılmadan write() kullanılamaz.")

        from influxdb_client import Point  # type: ignore

        numeric_cols = [
            c for c in df.columns
            if c not in self._SKIP_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
        ]

        total_written = 0
        batch: List[Any] = []

        for _, row in df.iterrows():
            ts = pd.Timestamp(row["DateTime"]).to_pydatetime()
            point = (
                Point(row.get("hostname", "unknown"))
                .tag("hostname", row.get("hostname", "unknown"))
                .tag("cpu", str(row.get("CPU", "all")))
                .time(ts)
            )
            for col in numeric_cols:
                val = row[col]
                if pd.notna(val):
                    point = point.field(col, float(val))

            batch.append(point)

            if len(batch) >= self.config.batch_size:
                self._write_api.write(bucket=self.config.bucket, record=batch)
                total_written += len(batch)
                logger.debug("InfluxDB: %d point yazıldı", total_written)
                batch = []

        if batch:
            self._write_api.write(bucket=self.config.bucket, record=batch)
            total_written += len(batch)

        logger.info("InfluxDB: toplam %d satır yazıldı.", total_written)
        return total_written

    def close(self) -> None:
        """Bağlantıyı kapat."""
        if self._client:
            self._client.close()
            self._connected = False
            logger.info("InfluxDB bağlantısı kapatıldı.")


# ---------------------------------------------------------------------------
# PostgreSQL Writer
# ---------------------------------------------------------------------------

class PostgreSQLWriter(BaseWriter):
    """
    PostgreSQL writer — psycopg2 connection pool ile batch insert.

    Gereksinim: pip install psycopg2-binary
    Tablo yoksa otomatik oluşturur (dynamic schema).
    """

    def __init__(self, config: PostgreSQLConfig):
        super().__init__(config)
        self._pool = None

    def connect(self) -> None:
        """Bağlantı havuzu oluştur."""
        try:
            from psycopg2 import pool as pg_pool  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "psycopg2 paketi gerekli: pip install psycopg2-binary"
            ) from exc

        self._pool = pg_pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=self.config.pool_size,
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.database,
            user=self.config.user,
            password=self.config.password,
            connect_timeout=self.config.timeout_s,
        )
        self._connected = True
        logger.info(
            "PostgreSQL pool oluşturuldu: %s:%d/%s",
            self.config.host, self.config.port, self.config.database,
        )

    def _ensure_table(self, conn: Any, df: pd.DataFrame) -> None:
        """Tablo yoksa dinamik olarak oluştur."""
        type_map = {
            "int64": "BIGINT",
            "float64": "DOUBLE PRECISION",
            "object": "TEXT",
            "bool": "BOOLEAN",
        }
        col_defs = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            pg_type = type_map.get(dtype, "TEXT")
            safe_col = f'"{col}"'
            col_defs.append(f"{safe_col} {pg_type}")

        ddl = (
            f"CREATE TABLE IF NOT EXISTS {self.config.table} "
            f"(id BIGSERIAL PRIMARY KEY, {', '.join(col_defs)})"
        )
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()

    def write(self, df: pd.DataFrame) -> int:
        """DataFrame'i PostgreSQL tablosuna COPY ile yükle."""
        import io
        from psycopg2.extras import execute_values  # type: ignore

        conn = self._pool.getconn()
        try:
            self._ensure_table(conn, df)
            cols = [f'"{c}"' for c in df.columns]
            col_list = ", ".join(cols)
            insert_sql = (
                f"INSERT INTO {self.config.table} ({col_list}) VALUES %s"
            )
            rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
            with conn.cursor() as cur:
                execute_values(cur, insert_sql, rows, page_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 1000)
            conn.commit()
            logger.info("PostgreSQL: %d satır yazıldı.", len(df))
            return len(df)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        """Bağlantı havuzunu kapat."""
        if self._pool:
            self._pool.closeall()
            self._connected = False
            logger.info("PostgreSQL pool kapatıldı.")


# ---------------------------------------------------------------------------
# Prometheus Exporter
# ---------------------------------------------------------------------------

class PrometheusExporter(BaseWriter):
    """
    Prometheus metrics HTTP exporter.

    Gereksinim: pip install prometheus-client
    Son DataFrame snapshot'ını gauge olarak expose eder.
    Endpoint: http://localhost:{port}/metrics
    """

    def __init__(self, config: PrometheusConfig):
        super().__init__(config)
        self._gauges: Dict[str, Any] = {}
        self._server = None

    def connect(self) -> None:
        """Prometheus HTTP server'ı başlat."""
        try:
            from prometheus_client import start_http_server, Gauge  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "prometheus-client paketi gerekli: pip install prometheus-client"
            ) from exc

        from prometheus_client import Gauge  # type: ignore
        self._Gauge = Gauge
        start_http_server(self.config.port)
        self._connected = True
        logger.info("Prometheus exporter başladı: http://0.0.0.0:%d/metrics", self.config.port)

    def _gauge_name(self, col: str) -> str:
        """SAR kolon adını Prometheus-uyumlu metrik adına dönüştür."""
        safe = col.lower().replace("/", "_per_").replace("%", "pct_").replace("-", "_")
        return f"{self.config.prefix}_{safe}"

    def write(self, df: pd.DataFrame) -> int:
        """DataFrame'in son satırını Prometheus gauge olarak yayınla."""
        if df.empty:
            return 0

        expose_cols = self.config.metrics or [
            c for c in df.columns
            if c not in {"DateTime", "hostname", "CPU"}
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        # Her hostname için ayrı gauge label
        for hostname, node_df in df.groupby("hostname"):
            last_row = node_df.iloc[-1]
            for col in expose_cols:
                if col not in node_df.columns:
                    continue
                gauge_key = f"{self._gauge_name(col)}_{hostname}"
                if gauge_key not in self._gauges:
                    self._gauges[gauge_key] = self._Gauge(
                        self._gauge_name(col),
                        f"SAR metric {col}",
                        ["hostname"],
                    )
                val = last_row[col]
                if pd.notna(val):
                    self._gauges[gauge_key].labels(hostname=hostname).set(float(val))

        logger.debug("Prometheus: %d gauge güncellendi.", len(self._gauges))
        return len(df)

    def close(self) -> None:
        """Prometheus server durdurulamaz (prometheus-client API kısıtı); no-op."""
        logger.info("Prometheus exporter: kapatma isteği alındı (server çalışmaya devam eder).")


# ---------------------------------------------------------------------------
# Database Pipeline Orchestrator
# ---------------------------------------------------------------------------

class DatabasePipeline:
    """
    Aktif tüm database writer'larını yönetir.
    Mevcut CSV/JSON OutputManager'ı ile paralel çalışır.

    Örnek kullanım::

        pipeline = DatabasePipeline.from_config(db_config)
        pipeline.connect_all()
        pipeline.write_all(df)
        pipeline.close_all()
    """

    def __init__(self, writers: List[BaseWriter]):
        self._writers = writers

    @classmethod
    def from_config(cls, config: DatabaseConfig) -> "DatabasePipeline":
        """
        DatabaseConfig'den aktif writer'ları oluştur.

        Yalnızca ``enabled=True`` olan adapter'lar dahil edilir.
        """
        writers: List[BaseWriter] = []
        if config.influxdb is not None and getattr(config.influxdb, "enabled", False):
            writers.append(InfluxDBWriter(config.influxdb))
        if config.postgresql is not None and getattr(config.postgresql, "enabled", False):
            writers.append(PostgreSQLWriter(config.postgresql))
        if config.prometheus is not None and getattr(config.prometheus, "enabled", False):
            writers.append(PrometheusExporter(config.prometheus))
        return cls(writers)

    @property
    def active_writers(self) -> List[BaseWriter]:
        """Aktif writer listesi."""
        return self._writers

    def connect_all(self) -> None:
        """Tüm writer'ları bağla."""
        for w in self._writers:
            try:
                w.connect()
            except Exception as exc:
                logger.error("%s bağlanırken hata: %s", type(w).__name__, exc)
                raise

    def write_all(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        DataFrame'i tüm aktif writer'lara yaz.

        Returns:
            {writer_name: yazılan_satır_sayısı}
        """
        results: Dict[str, int] = {}
        for w in self._writers:
            name = type(w).__name__
            try:
                n = w.write_with_retry(df)
                results[name] = n
            except Exception as exc:
                logger.error("%s yazma hatası: %s", name, exc)
                results[name] = -1
        return results

    def close_all(self) -> None:
        """Tüm writer bağlantılarını kapat."""
        for w in self._writers:
            try:
                w.close()
            except Exception as exc:
                logger.warning("%s kapatılırken hata: %s", type(w).__name__, exc)

    def __enter__(self) -> "DatabasePipeline":
        self.connect_all()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close_all()
