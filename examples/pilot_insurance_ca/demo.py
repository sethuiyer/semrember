#!/usr/bin/env python3
"""Canadian insurance pilot demo: semrember memory + Jev System One judgment.

    python examples/pilot_insurance_ca/demo.py
    python examples/pilot_insurance_ca/demo.py --live
    python examples/pilot_insurance_ca/demo.py --live --html-out report.html

The default is offline: semrember runs locally and the policy uses only the
local decision plus simple keyword guards. --live adds the TypeSafe System
One request and the calibrated question bank.

Synthetic data only. This is a routing demonstration, not a production claims
decision system. No automated denial or payment is performed.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "clients" / "python"))
sys.path.insert(0, str(HERE))

from semrember_client import Semrember, SemremberError  # noqa: E402

from jev_gate import JevClient, JevError, combine, judge, load_key  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Service:
    def __init__(self, binary: Path, csv: Path, key_col: str, value_col: str):
        self.port = free_port()
        self.log = tempfile.TemporaryFile(mode="w+")
        env = {**os.environ, "KEMAL_ENV": "production"}
        self.process = subprocess.Popen(
            [
                str(binary),
                "--train",
                str(csv),
                "--key",
                key_col,
                "--value",
                value_col,
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            cwd=ROOT,
            env=env,
            stdout=self.log,
            stderr=self.log,
        )
        self.client = Semrember(f"http://127.0.0.1:{self.port}", timeout=10)
        for _ in range(200):
            if self.process.poll() is not None:
                self.log.seek(0)
                raise RuntimeError(f"semrember failed to start:\n{self.log.read()}")
            try:
                if self.client.ping():
                    return
            except SemremberError:
                time.sleep(0.05)
        raise RuntimeError("semrember did not start in time")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.log.close()


def shorten(text: str, width: int = 120) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def print_scenario(audit: dict[str, Any], index: int) -> None:
    scenario = audit["scenario"]
    decision = audit["decision"]
    print()
    print("=" * 92)
    print(f"SCENARIO {index}: {scenario['title']}  [{scenario['province']} / {scenario['product']}]")
    print("=" * 92)
    print(f"Claim: {scenario['note']}")
    print()
    print("Local memory — similar claims:")
    if audit["similar_claims"]:
        for item in audit["similar_claims"][:4]:
            print(f"  {item['score']:+.3f}  {item['value']:<26}  {shorten(item['key'], 58)}")
    else:
        print("  (none)")
    local = audit["local"]
    print(
        "Local escalation: "
        f"handled={local.get('handled')} action={local.get('action')!r} "
        f"margin={local.get('margin'):.3f} novelty={local.get('novelty'):.3f}"
    )
    print()
    print("Experience memory — scars / fairness / attacks:")
    if audit["experiences"]:
        for item in audit["experiences"][:4]:
            print(f"  {item['score']:+.3f}  {shorten(item['value'], 78)}")
    else:
        print("  (none)")
    print()
    if audit.get("jev"):
        answers = audit["jev"]["answers"]
        signals = decision["signals"]
        route = answers.get("route", {})
        print("System One / Jev judgment:")
        print(
            f"  route: {route.get('choice')!r} "
            f"p={decision.get('probability')} conf={decision.get('confidence')}"
        )
        for key in (
            "actionable",
            "injury",
            "fraud_review",
            "complaint",
            "human_request",
            "injection",
            "routing_blocked",
            "experience_warning",
            "retrieval_fit",
        ):
            value = signals.get(key)
            if value is not None:
                print(f"  {key:<20} {value:.3f}")
        print(f"  severity             {decision.get('severity')}")
        usage = audit["jev"].get("usage", {})
        print(
            f"  model={audit['jev'].get('model')} "
            f"tokens={usage.get('input_tokens')}/{usage.get('output_tokens')}"
        )
    else:
        print("System One / Jev: offline (local-only fallback)")
    print()
    expected = scenario.get("expected_route")
    matched = decision["queue"] == expected or (expected == "human_review" and decision["queue"] == "human_review")
    print(
        f"DECISION: {decision['decision'].upper()} → {decision['queue']} "
        f"(human_required={decision['human_required']}, source={decision['source']})"
    )
    print(f"Reasons: {', '.join(decision['reasons'])}")
    if expected:
        print(f"Expected route: {expected}  {'✓' if matched else '✗'}")
    print()


def write_html(audit: list[dict[str, Any]], path: Path) -> None:
    cards = []
    for row in audit:
        scenario = row["scenario"]
        decision = row["decision"]
        local = row["local"]
        similar = "".join(
            f'<li><b>{html.escape(str(item["value"]))}</b> '
            f'<span class="score">{item["score"]:+.3f}</span><br>'
            f'{html.escape(str(item["key"]))}</li>'
            for item in row["similar_claims"][:4]
        ) or "<li>(none)</li>"
        experiences = "".join(
            f'<li><span class="score">{item["score"]:+.3f}</span> '
            f'{html.escape(str(item["value"]))}</li>'
            for item in row["experiences"][:4]
        ) or "<li>(none)</li>"
        signals = decision["signals"]
        signal_rows = "".join(
            f"<tr><td>{html.escape(key.replace('_', ' '))}</td><td>{value:.3f}</td></tr>"
            for key, value in signals.items()
            if isinstance(value, (int, float))
        )
        jev_block = ""
        if row.get("jev"):
            route = row["jev"]["answers"].get("route", {})
            jev_block = f"""
            <div class="panel">
              <h4>System One / Jev</h4>
              <p>Route: <b>{html.escape(str(route.get('choice')))}</b>
                 · p={decision.get('probability')}
                 · confidence={decision.get('confidence')}</p>
              <table>{signal_rows}</table>
              <p class="muted">model={html.escape(str(row['jev'].get('model')))}</p>
            </div>"""
        else:
            jev_block = '<div class="panel"><h4>System One / Jev</h4><p class="muted">Offline: local-only fallback.</p></div>'
        badge = "review" if decision["decision"] == "human_review" else "route"
        cards.append(
            f"""
        <section class="card">
          <div class="card-head">
            <div>
              <span class="eyebrow">{html.escape(scenario['id'])}</span>
              <h3>{html.escape(scenario['title'])}</h3>
              <p class="muted">{html.escape(scenario['province'])} / {html.escape(scenario['product'])}</p>
            </div>
            <span class="badge {badge}">{html.escape(decision['decision'])} → {html.escape(decision['queue'])}</span>
          </div>
          <blockquote>{html.escape(scenario['note'])}</blockquote>
          <div class="grid">
            <div class="panel"><h4>Similar claims</h4><ul>{similar}</ul></div>
            <div class="panel"><h4>Scars / attacks</h4><ul>{experiences}</ul></div>
            {jev_block}
          </div>
          <p><b>Why:</b> {html.escape(', '.join(decision['reasons']))}</p>
          <p class="muted">Local: handled={local.get('handled')} action={html.escape(str(local.get('action')))}
             margin={local.get('margin')} novelty={local.get('novelty')}</p>
        </section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Maple Shield Insurance — FNOL Triage Pilot</title>
<style>
  :root {{ --ink:#14231b; --muted:#63736a; --line:#d9e5dd; --paper:#f6faf7; --accent:#1f6f4a; --warn:#9a5b00; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--paper); }}
  header {{ padding:36px 28px 20px; max-width:1180px; margin:auto; }}
  h1 {{ margin:0 0 6px; font-size:30px; letter-spacing:-.5px; }}
  h2 {{ margin:0 0 8px; font-size:20px; }}
  .sub {{ color:var(--muted); max-width:800px; }}
  main {{ max-width:1180px; margin:auto; padding:0 28px 60px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:22px; margin:18px 0; box-shadow:0 1px 0 rgba(0,0,0,.02); }}
  .card-head {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-start; }}
  h3 {{ margin:2px 0; font-size:21px; }}
  h4 {{ margin:0 0 8px; font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }}
  .eyebrow {{ font:12px ui-monospace,monospace; color:var(--muted); }}
  .muted {{ color:var(--muted); }}
  blockquote {{ margin:14px 0; padding:12px 14px; background:#f4f8f5; border-left:3px solid var(--accent); border-radius:8px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; }}
  @media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  .panel {{ border:1px solid var(--line); border-radius:12px; padding:14px; }}
  ul {{ margin:0; padding-left:18px; }} li {{ margin:0 0 8px; }}
  .score {{ font:12px ui-monospace,monospace; color:var(--accent); }}
  .badge {{ display:inline-block; padding:7px 11px; border-radius:999px; font-weight:650; white-space:nowrap; }}
  .badge.route {{ background:#e7f5ec; color:#145c39; }}
  .badge.review {{ background:#fff0d6; color:var(--warn); }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  td {{ padding:3px 0; border-bottom:1px solid #eef3ef; }} td:last-child {{ text-align:right; font-family:ui-monospace,monospace; }}
  footer {{ color:var(--muted); font-size:13px; padding:0 28px 40px; max-width:1180px; margin:auto; }}
</style>
</head>
<body>
<header>
  <span class="eyebrow">PILOT DEMO · SYNTHETIC DATA</span>
  <h1>Maple Shield Insurance — FNOL Triage</h1>
  <p class="sub">Semrember retrieves similar claims and past operational scars. Jev / System One
  supplies calibrated, typed judgment. Code routes to a human-owned queue. No automated denial or payment.</p>
</header>
<main>{''.join(cards)}</main>
<footer>
  This is a demonstration, not a production claims decision system. All claims are synthetic.
  Canadian operating guardrails: human review for adverse decisions, privacy minimization,
  fairness testing, audit logging, and no reliance on protected grounds or proxies.
</footer>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=ROOT / "build" / "semrember-server")
    parser.add_argument("--claims", type=Path, default=HERE / "data" / "claims.csv")
    parser.add_argument("--experiences", type=Path, default=HERE / "data" / "experiences.csv")
    parser.add_argument("--actions", type=Path, default=HERE / "data" / "actions.json")
    parser.add_argument("--scenarios", type=Path, default=HERE / "data" / "scenarios.json")
    parser.add_argument("--live", action="store_true", help="call TypeSafe System One / Jev")
    parser.add_argument("--key-file", default=None, help="path to the TypeSafe API key file")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N scenarios")
    parser.add_argument("--only", default=None, help="comma-separated scenario ids")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--html-out", type=Path, default=None)
    args = parser.parse_args()

    if not args.server.is_file():
        raise SystemExit(f"missing {args.server}; run `make all` first")

    actions = load_json(args.actions)
    scenarios = load_json(args.scenarios)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        scenarios = [item for item in scenarios if item["id"] in wanted]
    if args.limit:
        scenarios = scenarios[: args.limit]

    jev_client = None
    if args.live:
        try:
            jev_client = JevClient(load_key(args.key_file), model=args.model)
        except JevError as exc:
            raise SystemExit(str(exc)) from exc

    claims_service = Service(args.server, args.claims, "claim_note", "action")
    experiences_service = Service(args.server, args.experiences, "experience", "lesson")
    audit: list[dict[str, Any]] = []
    try:
        for index, scenario in enumerate(scenarios, 1):
            try:
                similar = claims_service.client.similar(scenario["note"], top_k=6)
                local = claims_service.client.escalate(scenario["note"])
            except SemremberError as exc:
                raise SystemExit(f"semrember claims service failed: {exc}") from exc
            try:
                experiences = experiences_service.client.similar(scenario["note"], top_k=5)
            except SemremberError as exc:
                raise SystemExit(f"semrember experience service failed: {exc}") from exc

            jev_response = None
            if jev_client:
                try:
                    jev_response = judge(
                        jev_client, scenario, actions, similar, experiences
                    )
                except JevError as exc:
                    print(f"warning: Jev unavailable for {scenario['id']}: {exc}", file=sys.stderr)
                    jev_response = None

            decision = combine(
                scenario, local, similar, experiences, actions, jev_response
            )
            row = {
                "scenario": scenario,
                "similar_claims": similar,
                "local": local,
                "experiences": experiences,
                "jev": jev_response,
                "decision": decision,
            }
            audit.append(row)
            print_scenario(row, index)
    finally:
        claims_service.close()
        experiences_service.close()

    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)
    print(f"{'scenario':<30} {'expected':<28} {'queue':<28} result")
    matches = 0
    for row in audit:
        expected = row["scenario"].get("expected_route", "")
        queue = row["decision"]["queue"]
        ok = queue == expected
        matches += int(ok)
        print(f"{row['scenario']['id']:<30} {expected:<28} {queue:<28} {'✓' if ok else '✗'}")
    print(f"\n{matches}/{len(audit)} routed to the expected queue.")
    print("Synthetic demo only. No automated denial or payment. Human review remains authoritative.")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.json_out}")
    if args.html_out:
        write_html(audit, args.html_out)
        print(f"wrote {args.html_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
