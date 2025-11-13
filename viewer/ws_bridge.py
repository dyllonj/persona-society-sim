"""Threaded WebSocket broadcast server for live 3D viewer.

This module provides a `ViewerServer` that runs an asyncio-based
websocket server in a background thread and exposes a thread-safe
`broadcast(event: dict)` method to publish simulation events.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Set

import websockets


@dataclass
class ViewerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    path: str = "/ws"


class ViewerServer:
    def __init__(self, config: ViewerConfig | None = None) -> None:
        self.config = config or ViewerConfig()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._clients: Set[websockets.WebSocketServerProtocol] = set()
        self._queue: asyncio.Queue[str] | None = None
        self._server: websockets.server.Serve | None = None

    async def _handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        self._clients.add(websocket)
        try:
            # Send a hello message when connected
            await websocket.send(json.dumps({"type": "hello", "msg": "viewer connected"}))
            # Keep the connection open and ignore inbound messages for now
            async for _ in websocket:
                pass
        finally:
            self._clients.discard(websocket)

    async def _broadcaster(self) -> None:
        assert self._queue is not None
        while True:
            msg = await self._queue.get()
            # Broadcast to all connected clients; drop broken connections silently
            if not self._clients:
                continue
            stale: Set[websockets.WebSocketServerProtocol] = set()
            for ws in list(self._clients):
                try:
                    await ws.send(msg)
                except Exception:
                    stale.add(ws)
            for ws in stale:
                self._clients.discard(ws)

    async def _run_async(self) -> None:
        self._queue = asyncio.Queue()
        async def handler_with_path(ws: websockets.WebSocketServerProtocol, path: str):
            if path != self.config.path:
                await ws.close()
                return
            await self._handler(ws)
        self._server = await websockets.serve(handler_with_path, self.config.host, self.config.port)
        # Start broadcaster task
        asyncio.create_task(self._broadcaster())
        # Keep loop alive
        await asyncio.Future()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._loop = asyncio.new_event_loop()
        def run_loop() -> None:
            assert self._loop is not None
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_async())
        self._thread = threading.Thread(target=run_loop, name="viewer-ws", daemon=True)
        self._thread.start()

    def broadcast(self, event: Dict[str, Any]) -> None:
        if not self._loop or not self._queue:
            return
        msg = json.dumps(event)
        # Thread-safe put into async queue
        def _put() -> None:
            assert self._queue is not None
            self._queue.put_nowait(msg)
        self._loop.call_soon_threadsafe(_put)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        if not self._loop:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=1.0)
        self._loop = None
        self._thread = None
