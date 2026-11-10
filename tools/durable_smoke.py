#!/usr/bin/env python3
"""Check acknowledged writes survive SIGKILL using the real HTTP executable."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="build/semrember-server")
    binary = str(Path(parser.parse_args().server).resolve())
    with tempfile.TemporaryDirectory(prefix="semrember-durable-") as directory:
        root = Path(directory)
        key = root / "secret.key"
        key.write_bytes(os.urandom(32))
        key.chmod(0o600)
        data = root / "state.enc"
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        args = [binary, "--port", str(port), "--data", str(data), "--encryption-key-file", str(key)]

        def request(method, path, body=None):
            payload = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/{path}", data=payload, method=method,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.load(response)

        def start():
            process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            for _ in range(100):
                if process.poll() is not None:
                    raise AssertionError(process.stderr.read().decode())
                try:
                    request("GET", "ping")
                    return process
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.05)
            process.kill()
            process.wait()
            raise AssertionError("server did not start")

        process = start()
        try:
            request("PUT", "kv/disk", {"value": "free_disk"})
            request("PUT", "kv/deleted", {"value": "gone"})
            request("DELETE", "kv/deleted")
            request("PUT", "kv/offline-expiry", {"value": "gone"})
            request("POST", "kv/offline-expiry/expire", {"seconds": 1})
            request("POST", "kv/disk/expire", {"seconds": 3600})
        finally:
            process.kill()  # Deliberately bypass all graceful shutdown hooks.
            process.wait(timeout=5)
            process.stderr.close()
        time.sleep(1.1)
        process = start()
        try:
            assert request("GET", "kv/disk")["value"] == "free_disk"
            assert request("GET", "kv/disk/ttl")["ttl"] > 3500
            assert not request("GET", "kv/deleted/exists")["exists"]
            assert not request("GET", "kv/offline-expiry/exists")["exists"]
            assert request("POST", "similar", {"text": "disk"})["candidates"][0]["value"] == "free_disk"
            assert request("GET", "info")["persistence"] == "encrypted-snapshot"
            request("PUT", "kv/disk", {"value": "clear_cache"})
            assert request("GET", "kv/disk/ttl")["ttl"] is None
        finally:
            process.kill()
            process.wait(timeout=5)
            process.stderr.close()
        encrypted = data.read_bytes()
        assert b"clear_cache" not in encrypted
        key.write_bytes(os.urandom(32))
        wrong_key = subprocess.run(args, capture_output=True, timeout=5)
        assert wrong_key.returncode != 0
        assert data.read_bytes() == encrypted
        print("PASS: SIGKILL recovery, records, deletes, TTLs, semantic rebuild, encrypted bytes, wrong-key startup")


if __name__ == "__main__":
    main()
