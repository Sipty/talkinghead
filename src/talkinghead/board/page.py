"""Serve an editable browser preview for tablet-annotated board videos."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def _probe(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _proxy(source: Path, destination: Path, *, tablet: bool) -> Path:
    source = source.expanduser().resolve()
    signature = hashlib.sha256(json.dumps([str(source), _probe(source), tablet]).encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    memo = destination.with_suffix(destination.suffix + ".json")
    try:
        if destination.is_file() and json.loads(memo.read_text()).get("signature") == signature:
            return destination
    except (OSError, json.JSONDecodeError):
        pass
    temp = destination.with_name(destination.stem + ".tmp.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-map", "0:v:0"]
    if tablet:
        command += ["-vf", "scale=1180:-2,fps=15", "-an"]
    else:
        command += ["-vf", "scale=960:-2", "-map", "0:a:0?", "-c:a", "aac", "-b:a", "96k"]
    command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        os.replace(temp, destination)
        memo.write_text(json.dumps({"signature": signature}))
    except (OSError, subprocess.CalledProcessError) as exc:
        temp.unlink(missing_ok=True)
        detail = exc.stderr[-4000:] if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise RuntimeError(f"Could not build {'tablet' if tablet else 'phone'} proxy: {detail}") from exc
    return destination



def serve_page(run_dir: Path, spec_path: Path, port: int = 8765, open_browser: bool = True) -> None:
    """Build source proxies and serve the local board alignment/editor page."""
    run_dir, spec_path = Path(run_dir).resolve(), Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    run = json.loads((run_dir / "run.json").read_text())
    phone = Path(run["source"]).expanduser().resolve()
    tablet = Path(spec["tablet"]).expanduser().resolve()
    proxies = run_dir / "board" / "proxy"
    phone_proxy = _proxy(phone, proxies / "phone.mp4", tablet=False)
    tablet_proxy = _proxy(tablet, proxies / "tablet.mp4", tablet=True)
    page_template = Path(__file__).with_name("page.html").read_text()
    sentences_path = run_dir / "board" / "sentences.json"
    sentences = json.loads(sentences_path.read_text()) if sentences_path.is_file() else []
    html = page_template.replace("__INITIAL_SPEC__", json.dumps(spec).replace("</", "<\\/"))
    html = html.replace("__SENTENCES__", json.dumps(sentences).replace("</", "<\\/"))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def _reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._reply(200, html.encode(), "text/html; charset=utf-8")
                return
            target = {"/phone.mp4": phone_proxy, "/tablet.mp4": tablet_proxy}.get(path)
            if target is None:
                self._reply(404, b"Not found", "text/plain; charset=utf-8")
                return
            size = target.stat().st_size
            start, end, status = 0, size - 1, 200
            header = self.headers.get("Range")
            if header:
                try:
                    unit, value = header.split("=", 1)
                    if unit != "bytes":
                        raise ValueError
                    first, last = value.split("-", 1)
                    if first:
                        start = int(first)
                        end = min(size - 1, int(last)) if last else size - 1
                    else:
                        length = int(last)
                        start = max(0, size - length)
                    if start > end or start >= size:
                        raise ValueError
                    status = 206
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with target.open("rb") as stream:
                stream.seek(start)
                remaining = length
                while remaining:
                    block = stream.read(min(1 << 16, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save":
                self._reply(404, b"Not found", "text/plain; charset=utf-8")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length))
                from talkinghead.board.spec import save_spec, validate
                validated = validate(data)
                save_spec(validated, spec_path)
                self._reply(200, b'{"ok":true}', "application/json")
            except (ValueError, OSError) as exc:
                self._reply(400, json.dumps({"error": str(exc)}).encode(), "application/json; charset=utf-8")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    if open_browser:
        threading.Timer(0.35, webbrowser.open, args=(url,)).start()
    print(f"Board editor: {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
