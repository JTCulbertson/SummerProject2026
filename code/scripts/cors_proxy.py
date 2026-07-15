#!/usr/bin/env python3
"""
Local CORS proxy for the Model Baseline Analysis browser tool.

The tool (site/model-baseline-analysis.html) runs in the browser and calls the
MindRouter OpenAI-compatible API. Browsers enforce CORS, and MindRouter does not
send an Access-Control-Allow-Origin header, so a direct fetch() from the page is
blocked and shows up as a "NetworkError". Server-side scripts (run_multimodel.py,
run_reconstruction.py) never hit this because CORS is a browser-only rule.

This tiny stdlib-only proxy sits in the middle: it forwards every request to
MindRouter, streams the response straight back (so long SSE generations keep
flowing), and adds the CORS headers the browser needs.

Usage:
    uv run code/scripts/cors_proxy.py                 # listens on 127.0.0.1:8811
    uv run code/scripts/cors_proxy.py --port 9000

Then in the tool set  Base URL  to:   http://localhost:8811/v1

The proxy injects the MindRouter key from keys.txt when the browser doesn't send
one, so you can leave the tool's API-key field blank.
"""
import argparse
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

UPSTREAM = "https://mindrouter.uidaho.edu"


def load_api_key() -> str | None:
    # keys.txt lives in code/ (this file is code/scripts/cors_proxy.py → parents[1] == code/)
    keys = Path(__file__).resolve().parents[1] / "keys.txt"
    if not keys.exists():
        return None
    for line in keys.read_text().splitlines():
        line = line.strip()
        if line.startswith("Mindrouter_api"):
            return line.partition("=")[2].strip().strip('"').strip("'")
    return None


API_KEY = load_api_key()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, Accept")

    def _send_bytes(self, code: int, data: bytes, ctype: str) -> None:
        self.close_connection = True
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self) -> None:  # CORS preflight
        self.close_connection = True
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:
        self._forward("GET")

    def do_POST(self) -> None:
        self._forward("POST")

    def _forward(self, method: str) -> None:
        body = None
        if method == "POST":
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""

        headers: dict[str, str] = {}
        for h in ("Content-Type", "Accept"):
            if self.headers.get(h):
                headers[h] = self.headers[h]
        auth = self.headers.get("Authorization")
        if not auth and API_KEY:
            auth = "Bearer " + API_KEY
        if auth:
            headers["Authorization"] = auth

        req = Request(UPSTREAM + self.path, data=body, headers=headers, method=method)
        try:
            resp = urlopen(req, timeout=1200)
        except HTTPError as e:  # forward MindRouter's status + error body verbatim
            data = e.read()
            self._send_bytes(e.code, data, e.headers.get("Content-Type", "application/json"))
            return
        except URLError as e:
            msg = f'{{"error":"upstream unreachable: {e.reason}"}}'.encode()
            self._send_bytes(502, msg, "application/json")
            return

        # Stream the upstream response back. Use Connection: close (no Content-Length)
        # so SSE flows incrementally and the browser reads until EOF.
        self.close_connection = True
        self.send_response(resp.status)
        self._cors()
        self.send_header("Content-Type", resp.headers.get("Content-Type", "application/octet-stream"))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    ap = argparse.ArgumentParser(description="Local CORS proxy for the Model Baseline Analysis tool")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8811)
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"CORS proxy  →  {UPSTREAM}")
    print(f"Listening on   http://{args.host}:{args.port}")
    print(f"Tool Base URL: http://localhost:{args.port}/v1")
    print(f"keys.txt key:  {'loaded (leave the tool key field blank)' if API_KEY else 'NOT found — paste your key in the tool instead'}")
    print("Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        srv.shutdown()


if __name__ == "__main__":
    main()
