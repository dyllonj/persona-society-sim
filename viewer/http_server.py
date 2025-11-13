"""Simple threaded HTTP server to serve static viewer assets.

Serves files from `viewer/static` on http://127.0.0.1:8000/ by default.
"""

from __future__ import annotations

import http.server
import os
import socketserver
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StaticServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    directory: Path = Path(__file__).parent / "static"


class StaticServer:
    def __init__(self, config: StaticServerConfig | None = None) -> None:
        self.config = config or StaticServerConfig()
        self._thread: threading.Thread | None = None
        self._httpd: socketserver.TCPServer | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        handler = http.server.SimpleHTTPRequestHandler
        # Ensure directory exists
        os.makedirs(self.config.directory, exist_ok=True)
        # Change working directory only for this server thread
        def run_server() -> None:
            cwd = os.getcwd()
            try:
                os.chdir(self.config.directory)
                with socketserver.TCPServer((self.config.host, self.config.port), handler) as httpd:
                    self._httpd = httpd
                    httpd.serve_forever()
            finally:
                os.chdir(cwd)
        self._thread = threading.Thread(target=run_server, name="viewer-http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None
        self._httpd = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

