from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import socket
import sys
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# Config

class StreamMode(str, Enum):
    STDOUT    = "stdout"
    SOCKET    = "socket"
    WEBSOCKET = "websocket"


class StreamingConfig(BaseModel):
    """Streaming output yapılandırması."""
    model_config = ConfigDict(use_enum_values=False)

    enabled:              bool       = False
    mode:                 StreamMode = StreamMode.STDOUT
    host:                 str        = "0.0.0.0"
    port:                 int        = Field(default=9000, ge=1, le=65535)
    record_format:        str        = "json"       # "json" | "csv"
    flush_interval_rows:  int        = Field(default=100, ge=1)
    max_clients:          int        = Field(default=10, ge=1)
    # WebSocket ping intervali (saniye)
    ws_ping_interval:     int        = Field(default=20, ge=1)



# Record Serializers

def _row_to_json(row: pd.Series) -> str:
    """Tek DataFrame satırını JSON string'e dönüştür."""
    d: Dict[str, Any] = {}
    for k, v in row.items():
        if hasattr(v, "item"):          # numpy scalar
            v = v.item()
        d[k] = v
    return json.dumps(d, default=str)


def _df_to_csv_lines(df: pd.DataFrame, include_header: bool = False) -> List[str]:
    """DataFrame chunk'ını CSV satırlarına dönüştür."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=include_header)
    return buf.getvalue().splitlines()


# Base Streamer

class BaseStreamer:
    """Tüm streamer implementasyonlarının ortak arayüzü."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self._header_sent = False
        self._lock = threading.Lock()

    def send_chunk(self, df: pd.DataFrame) -> None:
        """Bir DataFrame chunk'ını yayınla."""
        raise NotImplementedError

    def close(self) -> None:
        """Bağlantıları kapat."""

    def _serialize(self, df: pd.DataFrame) -> List[str]:
        """Config'e göre DataFrame'i string satırlara dönüştür."""
        if self.config.record_format == "json":
            return [_row_to_json(row) for _, row in df.iterrows()]
        else:  # csv
            include_header = not self._header_sent
            lines = _df_to_csv_lines(df, include_header=include_header)
            self._header_sent = True
            return lines


# Stdout Streamer

class StdoutStreamer(BaseStreamer):
    """SAR verilerini stdout'a satır satır yazar."""

    def send_chunk(self, df: pd.DataFrame) -> None:
        """DataFrame chunk'ını stdout'a flush et."""
        lines = self._serialize(df)
        with self._lock:
            for line in lines:
                sys.stdout.write(line + "\n")
            sys.stdout.flush()
        logger.debug("Stdout: %d satır yayınlandı.", len(lines))

    def close(self) -> None:
        sys.stdout.flush()


# Socket Streamer

class SocketStreamer(BaseStreamer):
    """
    TCP socket üzerinden SAR verisi yayını (server mode).
    Bağlanan istemcilere aynı veriyi broadcast eder.

    Test: nc localhost 9000
    """

    def __init__(self, config: StreamingConfig):
        super().__init__(config)
        self._clients: List[socket.socket] = []
        self._server_sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._running = False

    def start_server(self) -> None:
        """Socket server'ı arka plan thread'de başlat."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.config.host, self.config.port))
        self._server_sock.listen(self.config.max_clients)
        self._running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="SocketStreamer-accept"
        )
        self._accept_thread.start()
        logger.info(
            "SocketStreamer başladı: %s:%d", self.config.host, self.config.port
        )

    def _accept_loop(self) -> None:
        self._server_sock.settimeout(1.0)
        while self._running:
            try:
                conn, addr = self._server_sock.accept()
                with self._lock:
                    self._clients.append(conn)
                logger.info("Yeni client: %s", addr)
            except socket.timeout:
                continue
            except OSError:
                break

    def send_chunk(self, df: pd.DataFrame) -> None:
        """Tüm bağlı client'lara chunk yayınla, kopuk olanları temizle."""
        lines = self._serialize(df)
        payload = ("\n".join(lines) + "\n").encode()

        dead: List[socket.socket] = []
        with self._lock:
            for client in self._clients:
                try:
                    client.sendall(payload)
                except (BrokenPipeError, OSError):
                    dead.append(client)
            for d in dead:
                self._clients.remove(d)
                d.close()

        logger.debug("Socket: %d client'a %d satır gönderildi.", len(self._clients) - len(dead), len(lines))

    def close(self) -> None:
        """Server ve client bağlantılarını kapat."""
        self._running = False
        if self._server_sock:
            self._server_sock.close()
        with self._lock:
            for c in self._clients:
                try:
                    c.close()
                except OSError:
                    pass
            self._clients.clear()
        logger.info("SocketStreamer kapatıldı.")


# WebSocket Streamer

class WebSocketStreamer(BaseStreamer):
    """
    WebSocket üzerinden SAR verisi yayını (asyncio tabanlı).
    """

    def __init__(self, config: StreamingConfig):
        super().__init__(config)
        self._clients: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server = None
        self._queue: asyncio.Queue = None  # type: ignore

    def start_server(self) -> None:
        """WebSocket server'ı ayrı bir thread+event loop'ta başlat."""
        try:
            import websockets  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "websockets paketi gerekli: pip install websockets"
            ) from exc

        self._server_thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name="WebSocketStreamer"
        )
        self._server_thread.start()
        # Event loop hazır olana kadar bekle
        time.sleep(0.5)
        logger.info(
            "WebSocketStreamer başladı: ws://%s:%d", self.config.host, self.config.port
        )

    def _run_event_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue()
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        import websockets  # type: ignore

        async def handler(ws: Any) -> None:
            self._clients.add(ws)
            logger.info("WS client bağlandı. Toplam: %d", len(self._clients))
            try:
                await ws.wait_closed()
            finally:
                self._clients.discard(ws)

        async with websockets.serve(
            handler,
            self.config.host,
            self.config.port,
            ping_interval=self.config.ws_ping_interval,
        ) as server:
            self._server = server
            # Broadcast kuyruğunu tüket
            while True:
                message = await self._queue.get()
                if message is None:
                    break
                if self._clients:
                    await asyncio.gather(
                        *[c.send(message) for c in self._clients],
                        return_exceptions=True,
                    )
                self._queue.task_done()

    def send_chunk(self, df: pd.DataFrame) -> None:
        """Chunk'ı WebSocket broadcast kuyruğuna ekle (non-blocking)."""
        if self._loop is None or self._queue is None:
            return
        lines = self._serialize(df)
        payload = "\n".join(lines)
        asyncio.run_coroutine_threadsafe(self._queue.put(payload), self._loop)
        logger.debug("WS: %d satır kuyruğa eklendi.", len(lines))

    def close(self) -> None:
        """Event loop'u durdur."""
        if self._loop and self._queue:
            asyncio.run_coroutine_threadsafe(self._queue.put(None), self._loop)
        logger.info("WebSocketStreamer kapatıldı.")


# SARStreamer

class SARStreamer:
    """
    Tek giriş noktası: config'e göre doğru streamer'ı oluşturur ve yönetir.
    """

    def __init__(self, config: StreamingConfig):
        self.config = config
        self._impl: Optional[BaseStreamer] = None

    def start(self) -> None:
        """Streaming backend'i başlat."""
        if not self.config.enabled:
            logger.info("Streaming devre dışı, başlatılmıyor.")
            return

        mode = self.config.mode
        if mode == StreamMode.STDOUT:
            self._impl = StdoutStreamer(self.config)
        elif mode == StreamMode.SOCKET:
            impl = SocketStreamer(self.config)
            impl.start_server()
            self._impl = impl
        elif mode == StreamMode.WEBSOCKET:
            impl = WebSocketStreamer(self.config)
            impl.start_server()
            self._impl = impl
        else:
            raise ValueError(f"Bilinmeyen stream modu: {mode}")

        logger.info("SARStreamer başlatıldı: mode=%s", mode.value if hasattr(mode, 'value') else mode)

    def send_chunk(self, df: pd.DataFrame) -> None:
        """Bir chunk'ı aktif streamer'a ilet."""
        if self._impl is None:
            return
        self._impl.send_chunk(df)

    def stop(self) -> None:
        """Streamer'ı durdur ve kaynakları serbest bırak."""
        if self._impl:
            self._impl.close()
            self._impl = None

    def __enter__(self) -> "SARStreamer":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()
