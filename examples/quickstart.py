"""Run against a fresh --train examples/incidents.csv service.

Install the client first: python -m pip install ./clients/python
Run: python examples/quickstart.py [--url http://127.0.0.1:7379]
"""
import argparse
import json
from uuid import uuid4

from semrember_client import Semrember


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7379")
    args = parser.parse_args()
    memory = Semrember(args.url)
    info = memory.info()
    if not info["semantic_reads"] or not info["semantic_updates"]:
        raise SystemExit("Start the service with --train examples/incidents.csv for this example.")

    key = "certificate has expired"
    assert memory.get(key) == "renew_certificate"
    print("Exact value:", memory.get(key))
    print("Related records:", json.dumps(memory.similar(key, top_k=2)))

    decision = memory.escalate(key)
    assert decision["handled"] and decision["action"] == "renew_certificate"
    print("Decision:", json.dumps(decision))
    # Your application decides whether to execute a suggested action or
    # forward an escalated case to a human or another system.

    demo_key = "quickstart/python/" + uuid4().hex
    try:
        assert memory.set(demo_key, "review") is None
        assert memory.get(demo_key) == "review"
        assert memory.set(demo_key, "review_again") == "review"
        assert memory.exists(demo_key)
        assert demo_key in memory.scan_all(match="quickstart/python/*")
        assert memory.expire(demo_key, 60)
        assert memory.ttl(demo_key) is not None
        print("Exact write, replacement, scan and TTL: OK")
    finally:
        memory.delete(demo_key)
    assert memory.get(demo_key) is None
    print("Deleted demo key; bundled training records are unchanged.")


if __name__ == "__main__":
    main()
