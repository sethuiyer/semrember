#!/usr/bin/env python3
"""Exercise the compiled Kemal server through its standard-library client."""
from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "clients/python"))
from semrember_client import Semrember, SemremberError


@contextmanager
def service(executable: Path, *args: str):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryFile(mode="w+") as log:
        proc = subprocess.Popen(
            [str(executable), "--port", str(port), *args], cwd=ROOT,
            env={**os.environ, "KEMAL_ENV": "production"}, stdout=log, stderr=log,
        )
        try:
            kb = Semrember(f"http://127.0.0.1:{port}", timeout=2)
            for _ in range(100):
                if proc.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(f"server failed to start:\n{log.read()}")
                try:
                    if kb.ping():
                        break
                except SemremberError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("server startup timed out")
            yield kb
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise AssertionError("Kemal did not shut down gracefully")
        assert proc.returncode == 0, f"server exited with {proc.returncode}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=ROOT / "build/semrember-server")
    parser.add_argument("--cli", type=Path, default=ROOT / "build/semrember")
    args = parser.parse_args()
    server, cli = args.server.resolve(), args.cli.resolve()

    with service(server) as kb:
        assert kb.set("disk full", "free_disk") is None
        assert kb.escalate("disk full")["action"] == "free_disk"
        assert kb.set("disk full", "clear_cache") == "free_disk"
        assert kb.escalate("disk full")["action"] == "clear_cache"
        for key in ("a/exists", "b/ttl", "c/expire", "a+b", "東京", "a%2Fexists", ""):
            kb.set(key, "value")
            assert kb.get(key) == "value" and kb.exists(key), key
        kb.set("review_suffix", "x")
        assert kb.scan_all(match="review") == []
        kb.set("Disk", "upper")
        kb.set("disk", "lower")
        kb.delete("Disk")
        assert any(c["key"] == "disk" for c in kb.similar("disk", 100))
        assert kb.expire("disk full", 1)
        time.sleep(1.1)
        assert not kb.exists("disk full") and kb.get("disk full") is None
        assert all(c["key"] != "disk full" for c in kb.similar("disk full", 100))
        for method, path, body, status in (
            ("GET", "/missing", None, 404),
            ("POST", "/v1/ping", None, 405),
            ("PUT", "/v1/kv/disk", {}, 400),
        ):
            try:
                kb._request(method, path, body)
            except SemremberError as exc:
                assert exc.status == status, (path, exc)
            else:
                raise AssertionError(f"expected HTTP {status} from {path}")
        with urllib.request.urlopen(urllib.request.Request(kb.url + "/v1/ping", method="HEAD")) as response:
            assert response.status == 200 and response.read() == b""
        print("PASS: Kemal routes, encoded keys, JSON errors, HEAD, updates and expiry")
    print("PASS: graceful SIGTERM shutdown")

    corpus = ROOT / "examples/incidents.csv"
    with service(server, "--train", str(corpus)) as kb:
        with corpus.open() as source:
            for row in csv.DictReader(source):
                decision = kb.escalate(row["message"])
                assert decision["handled"] and decision["action"] == row["action"], decision
        print("PASS: all 34 bundled incident routes")

    with tempfile.TemporaryDirectory() as temporary:
        model = str(Path(temporary) / "memory.amem")
        subprocess.run([str(cli), "train", str(corpus), "message", "action", model], check=True, capture_output=True)
        with service(server, "--model", model) as kb:
            kb.set("disk", "x")
            assert kb.get("disk") == "x"
            assert kb.info()["semantic_updates"] is False
            try:
                kb.similar("disk")
            except SemremberError as exc:
                assert exc.status == 409
            else:
                raise AssertionError("loaded reader must report unavailable semantic state")
    print("PASS: loaded-reader HTTP contract")


if __name__ == "__main__":
    main()
