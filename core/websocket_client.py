"""
Background WebSocket client for live IMU streaming.

Runs an asyncio event loop in a daemon thread.
Parsed JSON dicts are pushed into a thread-safe queue.Queue.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Optional

import streamlit as st

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False


class WebSocketClient:
    def __init__(self, url: str, data_queue: queue.Queue) -> None:
        self.url = url
        self.queue = data_queue
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()
        self.connected = False
        self.error: Optional[str] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not _WS_AVAILABLE:
            self.error = "websockets library not installed"
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception as exc:
            self.error = str(exc)
            self.connected = False
        finally:
            self._loop.close()

    async def _connect(self) -> None:
        try:
            async with websockets.connect(self.url) as ws:
                self.connected = True
                self.error = None
                while not self._stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(raw)
                        # Normalise key names: accept both t/timestamp and ax/ay/az
                        parsed = {
                            "t": data.get("t") or data.get("timestamp"),
                            "ax": float(data.get("ax", 0)),
                            "ay": float(data.get("ay", 0)),
                            "az": float(data.get("az", 0)),
                            "gx": float(data.get("gx", 0)),
                            "gy": float(data.get("gy", 0)),
                            "gz": float(data.get("gz", 0)),
                        }
                        self.queue.put_nowait(parsed)
                    except asyncio.TimeoutError:
                        continue
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            self.error = f"Connection failed: {exc}"
            self.connected = False


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def get_client(session_state) -> Optional[WebSocketClient]:
    return session_state.get("ws_client")


def start_client(session_state, url: str) -> WebSocketClient:
    """Create, start, and store a WebSocketClient in session_state."""
    q = session_state.get("ws_queue")
    if q is None:
        q = queue.Queue()
        session_state["ws_queue"] = q

    client = WebSocketClient(url, q)
    client.start()
    session_state["ws_client"] = client
    return client


def stop_client(session_state) -> None:
    client: Optional[WebSocketClient] = session_state.get("ws_client")
    if client:
        client.stop()
        session_state["ws_client"] = None


def drain_queue(session_state) -> list[dict]:
    """Pop all pending samples from the queue, return as list."""
    q: Optional[queue.Queue] = session_state.get("ws_queue")
    if q is None:
        return []
    samples = []
    while True:
        try:
            samples.append(q.get_nowait())
        except Exception:
            break
    return samples
