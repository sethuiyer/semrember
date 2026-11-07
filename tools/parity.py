#!/usr/bin/env python3
"""Cross-implementation parity harness — a MAINTAINER tool.

    python3 tools/parity.py --python-root ../ [--static models/potion-base-8M.bin]

Checks this implementation against an independent one on three axes that
must agree for the format to be worth anything:

  1. AMEM bytes   — a model written by either is byte-identical.
  2. Scores       — the same model and query give the same numbers.
  3. Token ids    — the static encoder segments text identically.

This is not part of the shipped test suite, and a public checkout cannot run
it: it imports a reference implementation that is not in this repository.
`crystal spec` is the suite that ships.  The harness exits 0 with a clear
message when the reference is missing, so `make parity` is harmless either
way — it just proves nothing without one.

Why keep it here: the conformance claims in README.md are only meaningful if
the check is reproducible, and this is the check.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILURES = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def crystal(*args: str) -> str:
    exe = ROOT / "build" / "semrember"
    if not exe.exists():
        raise SystemExit("build the CLI first: crystal build src/cli.cr -o build/semrember")
    r = subprocess.run([str(exe), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"crystal {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python-root", default=str(ROOT.parent),
                    help="directory containing a reference implementation to compare against")
    ap.add_argument("--static", default=None, help="static encoder for the semantic half")
    ap.add_argument("--corpus", default=str(ROOT.parent / "examples/incidents.csv"))
    args = ap.parse_args()

    sys.path.insert(0, args.python_root)
    try:
        from semrember import AssociativeMemory
    except ImportError:
        print("Python reference not found at", args.python_root)
        print("Pass --python-root.  Nothing to compare against.")
        return 0

    print("Building the Crystal CLI...")
    (ROOT / "build").mkdir(exist_ok=True)
    subprocess.run(["crystal", "build", "--release", "src/cli.cr",
                    "-o", "build/semrember"], cwd=ROOT, check=True)

    # ── corpus ──────────────────────────────────────────────────────────
    import csv
    rows = []
    with open(args.corpus, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["message"].strip(), r["action"].strip()))
    data = dict(rows)
    queries = [t for t, _ in rows[:8]] + [t for t, _ in rows[-4:]]

    static = args.static
    py_static = StaticEncoder = None
    if static:
        from semrember.encoders import StaticEncoder
        py_static = StaticEncoder(static)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        py_amem = d / "py.amem"
        cr_amem = d / "cr.amem"

        # ── 1. write parity ──────────────────────────────────────────────
        print("\n1. AMEM write parity")
        mem = AssociativeMemory(static_path=static, solver="dual").fit(data)
        mem.save(py_amem)
        crystal(*(["train", args.corpus, "message", "action", str(cr_amem), "1", "dual"] + (["--static", static] if static else [])))

        pb = py_amem.read_bytes()
        cb = cr_amem.read_bytes()
        if len(pb) != len(cb):
            check("AMEM byte length", False, f"python {len(pb)} vs crystal {len(cb)}")
        else:
            diff = next((i for i in range(len(pb)) if pb[i] != cb[i]), None)
            if diff is None:
                check("AMEM bytes identical", True)
            else:
                # report where and how much the weights differ
                n_diff = sum(1 for a, b in zip(pb, cb) if a != b)
                check("AMEM bytes identical", False,
                      f"first difference at byte {diff}, {n_diff} bytes differ")

        # ── 2. score parity ──────────────────────────────────────────────
        print("\n2. Score parity")
        # Compare against the *loaded* (float16) Python model, not the live
        # float32 one: the file is what both implementations actually read.
        ref = AssociativeMemory.load(str(py_amem), static_path=static)
        worst = 0.0
        worst_q = ""
        for q in queries:
            out = json.loads(crystal(*(["scores", str(py_amem), q] + (["--static", static] if static else []))))
            py_scores = ref._scores(q)
            for label, sc in zip(out["labels"], out["scores"]):
                i = mem._labels.index(label)
                d = abs(sc - float(py_scores[i]))
                if d > worst:
                    worst, worst_q = d, q
        tol = 1e-5
        check(f"scores agree (worst delta {worst:.3e})", worst <= tol,
              f"worst delta {worst:.3e} on {worst_q!r}")

        # ── 3. cross-load ────────────────────────────────────────────────
        print("\n3. Cross-load")
        # Crystal writes, Python reads.
        loaded = AssociativeMemory.load(str(cr_amem), static_path=static)
        ok = all(loaded.query(q, topk=1) == mem.query(q, topk=1) for q in queries)
        check("python reads a Crystal-written model", ok)
        # Python writes, Crystal reads.
        out = json.loads(crystal(*(["scores", str(py_amem), queries[0]] + (["--static", static] if static else []))))
        check("crystal reads a Python-written model", len(out["scores"]) == len(mem._labels))

        # ── 4. tokenizer parity ──────────────────────────────────────────
        if static:
            print("\n4. Tokenizer parity")
            enc = StaticEncoder(static)
            cr_enc = crystal_token_ids(static)
            bad = []
            for q in queries + ["hello—world", "ΟΣ", "a\u0315\u0300", "app won't open"]:
                want = enc._tok.encode(q)
                got = cr_enc(q)
                if want != got:
                    bad.append((q, want, got))
            check(f"token ids agree on {len(queries) + 4} strings", not bad,
                  f"first divergence: {bad[0][0]!r} python={bad[0][1]} crystal={bad[0][2]}"
                  if bad else "")

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("\nFailures:")
        for f in FAILURES:
            print("  -", f)
        return 1
    return 0


def crystal_token_ids(static: str):
    """Ask the Crystal binary for the token ids of a string."""
    def ids(text: str):
        out = subprocess.run([str(ROOT / "build" / "semrember"), "tokens", static, text],
                             capture_output=True, text=True, check=True)
        return json.loads(out.stdout)
    return ids


if __name__ == "__main__":
    raise SystemExit(main())
