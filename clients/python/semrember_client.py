"""
semrember — Python client for the semrember service.

    from semrember_client import Semrember

    kb = Semrember("http://127.0.0.1:7379")
    kb.set("No space left on device", "free_disk")
    kb.get("No space left on device")        # "free_disk"
    kb.similar("disk full")                  # [{key, value, score}, ...]
    kb.escalate("ImagePullBackOff")          # {decision, reason, margin, novelty, ...}

The API is Redis-shaped: exact reads return dictionary values, writes
maintain the semantic index, and the semantic operations
(SIMILAR, ESCALATE, NOVELTY) always return candidates and confidence
rather than a bare answer.

Standard library only — no requests, no redis-py, no numpy.  The service
itself is a compiled binary; static linking is an optional build mode.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

__all__ = ["Semrember", "SemremberError"]

DEFAULT_URL = "http://127.0.0.1:7379"


class SemremberError(RuntimeError):
    """Raised when the service returns an error response."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class Semrember:
    """A client for the semrember HTTP/JSON service."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 5.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    # ── transport ────────────────────────────────────────────────────────
    def _request(self, method: str, path: str, body: Any = None) -> Any:
        url = f"{self.url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read().decode("utf-8")
                status = resp.status
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", "replace")
            status = exc.code
        except urllib.error.URLError as exc:
            raise SemremberError(f"cannot reach {self.url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise SemremberError(f"request to {self.url} timed out") from exc
        try:
            parsed = json.loads(payload) if payload else {}
        except json.JSONDecodeError as exc:
            raise SemremberError(f"malformed response: {payload[:120]!r}") from exc
        if not isinstance(parsed, dict):
            raise SemremberError("malformed response: expected a JSON object", status)
        if status >= 400:
            raise SemremberError(parsed.get("error", f"HTTP {status}"), status)
        return parsed

    @staticmethod
    def _key_path(key: str) -> str:
        return "/v1/kv/" + urllib.parse.quote(key, safe="")

    # ── server ───────────────────────────────────────────────────────────
    def ping(self) -> bool:
        return bool(self._request("GET", "/v1/ping").get("ok"))

    def info(self) -> dict:
        return self._request("GET", "/v1/info")

    # ── exact operations ────────────────────────────────────────────────
    def get(self, key: str) -> str | None:
        """The stored value, or None when the key is absent."""
        try:
            return self._request("GET", self._key_path(key)).get("value")
        except SemremberError as exc:
            if exc.status == 404:
                return None
            raise

    def set(self, key: str, value: str) -> str | None:
        """Store a value; returns the previous value, or None."""
        out = self._request("PUT", self._key_path(key), {"value": value})
        return out.get("previous")

    def delete(self, key: str) -> bool:
        return bool(self._request("DELETE", self._key_path(key)).get("deleted"))

    def exists(self, key: str) -> bool:
        return bool(self._request("GET", self._key_path(key) + "/exists").get("exists"))

    def scan(self, cursor: str = "0", match: str | None = None,
             count: int = 100) -> tuple[str, list[str]]:
        """Deterministic key scan, Redis-style."""
        q = {"cursor": cursor, "count": str(count)}
        if match:
            q["match"] = match
        out = self._request("GET", "/v1/kv?" + urllib.parse.urlencode(q))
        return out.get("cursor", "0"), list(out.get("keys", []))

    def scan_all(self, match: str | None = None, count: int = 100) -> list[str]:
        keys: list[str] = []
        cursor = "0"
        while True:
            cursor, page = self.scan(cursor, match, count)
            keys.extend(page)
            if cursor == "0":
                break
        return keys

    def expire(self, key: str, seconds: int) -> bool:
        return bool(self._request("POST", self._key_path(key) + "/expire",
                                  {"seconds": seconds}).get("ok"))

    def ttl(self, key: str) -> int | None:
        return self._request("GET", self._key_path(key) + "/ttl").get("ttl")

    # ── semantic operations (always return candidates and confidence) ─────
    def similar(self, text: str, top_k: int = 5,
                min_score: float | None = None) -> list[dict]:
        """Records most similar to `text`, with scores."""
        body: dict = {"text": text, "top_k": top_k}
        if min_score is not None:
            body["min_score"] = min_score
        out = self._request("POST", "/v1/similar", body)
        return list(out.get("candidates", []))

    def escalate(self, text: str, min_margin: float = 0.35,
                 min_novelty: float = 0.55) -> dict:
        """Decide whether to act, and say why.

        Returns a dict with `decision` ("handle" or "escalate"), `reason`
        ("ambiguous", "out_of_domain" or "both"), the two signals, the
        ranked candidates and — when the answer was escalated — nothing
        hidden: the caller gets the evidence, not just a refusal.
        """
        return self._request("POST", "/v1/escalate", {
            "text": text,
            "min_margin": min_margin,
            "min_novelty": min_novelty,
        })

    def novelty(self, text: str) -> float:
        """Out-of-distribution score in [0, 1]; 1 means nothing stored resembles it."""
        return float(self._request("GET", "/v1/novelty?" +
                                   urllib.parse.urlencode({"text": text}))["novelty"])


def _demo() -> None:
    kb = Semrember()
    print("ping:", kb.ping())
    info = kb.info()
    print(f"service: {info.get('records')} records, {info.get('labels')} labels, "
          f"solver={info.get('solver')}")

    print("\nexact operations:")
    prev = kb.set("No space left on device", "free_disk")
    print(f"  set    -> previous={prev!r}")
    print(f"  get    -> {kb.get('No space left on device')!r}")
    print(f"  exists -> {kb.exists('No space left on device')}")
    print(f"  delete -> {kb.delete('No space left on device')}")
    print(f"  get    -> {kb.get('No space left on device')!r}")

    print("\nsemantic operations:")
    for cand in kb.similar("disk full", top_k=3):
        print(f"  similar {cand['key']!r} -> {cand['value']!r} ({cand['score']:.4f})")
    for text in ("No space left on device", "ImagePullBackOff"):
        d = kb.escalate(text)
        print(f"  escalate {text!r}: {d['decision']} ({d['reason']}) "
              f"margin={d['margin']:+.3f} novelty={d['novelty']:.3f}")


if __name__ == "__main__":
    _demo()
