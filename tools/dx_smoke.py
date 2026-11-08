#!/usr/bin/env python3
"""Verify installation, documented examples, and AMEM/restart behavior.

Requires the built executables, an installed Python client, Crystal, curl,
and Node.js. Uses temporary ports/files and no optional embedding models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from http_smoke import ROOT, service
from semrember_client import SemremberError


def run(command: list[str], **kwargs) -> str:
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"command failed: {command!r}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def executable(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"Required executable not found: {name}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=ROOT / "build/semrember-server")
    parser.add_argument("--cli", type=Path, default=ROOT / "build/semrember")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--crystal", default="crystal")
    args = parser.parse_args()
    server, cli = args.server.resolve(), args.cli.resolve()
    python, crystal = executable(args.python), executable(args.crystal)
    node, curl = executable("node"), executable("curl")
    corpus = ROOT / "examples/incidents.csv"
    corpus_hash = hashlib.sha256(corpus.read_bytes()).hexdigest()
    key = "certificate has expired"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)  # Exercise the installed package, not a source-path fallback.

    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        run([python, "-c", "from semrember_client import Semrember"], cwd=temp, env=env)
        prefix = temp / "installed tools"
        # Build artifacts were already checked by `make all`. Test the real
        # install recipe without recompiling unchanged executables.
        run(["make", "-o", "all", "install", f"PREFIX={prefix}"], cwd=ROOT)
        for binary in ("semrember", "semrember-server"):
            run([str(prefix / "bin" / binary), "--help"], cwd=temp)
        print("PASS: installed Python package and custom-prefix executable installation")

        with service(server, "--train", str(corpus)) as memory:
            assert json.loads(run([curl, "-fsS", memory.url + "/v1/ping"]))["ok"]
            run([python, str(ROOT / "examples/quickstart.py"), "--url", memory.url], cwd=temp, env=env)
            run([node, str(ROOT / "examples/quickstart.mjs")], cwd=temp,
                env={**env, "SEMREMBER_URL": memory.url})
            path = memory.url + "/v1/kv/certificate%20has%20expired"
            result = json.loads(run([curl, "-fsS", "-X", "PUT", path, "-H", "Content-Type: application/json",
                                     "-d", '{"value":"inspect_certificate"}']))
            assert result["previous"] == "renew_certificate" and result["semantic_updated"]
            assert json.loads(run([curl, "-fsS", path]))["value"] == "inspect_certificate"
            memory.set("quickstart/transient", "review")
            memory.expire(key, 60)
        print("PASS: curl walkthrough, installed Python example, and JavaScript example")

        with service(server, "--train", str(corpus)) as memory:
            assert memory.get(key) == "renew_certificate"
            assert memory.get("quickstart/transient") is None
            assert memory.ttl(key) is None
        assert hashlib.sha256(corpus.read_bytes()).hexdigest() == corpus_hash
        print("PASS: restart rebuilds source records; live writes and TTLs do not survive")

        run([crystal, "run", "examples/quickstart.cr"], cwd=ROOT)
        print("PASS: Crystal engine, replacement, save/load, and reader-only updates")
        model = temp / "brain.amem"
        run([str(cli), "train", str(corpus), "message", "action", str(model)], cwd=ROOT)
        result = json.loads(run([str(cli), "query", str(model), key, "3"], cwd=temp))
        assert result["hits"][0]["label"] == "renew_certificate"
        json.loads(run([str(cli), "info", str(model)], cwd=temp))

        with service(server, "--model", str(model)) as memory:
            assert memory.info()["records"] == 0
            assert memory.get(key) is None
            write = memory._request("PUT", "/v1/kv/temporary", {"value": "review"})
            assert write["semantic_updated"] is False
            assert memory.get("temporary") == "review"
            for operation in (memory.similar, memory.escalate, memory.novelty):
                try:
                    operation(key)
                except SemremberError as error:
                    assert error.status == 409
                else:
                    raise AssertionError("loaded AMEM must not expose observation-dependent reads")
        with service(server, "--model", str(model)) as memory:
            assert memory.get("temporary") is None
        print("PASS: CLI classification reader, missing original keys, HTTP 409, and reader restart")


if __name__ == "__main__":
    main()
