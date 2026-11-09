// Node.js >= 18. Start the bundled incident service first.
// SEMREMBER_URL=http://127.0.0.1:7379 node examples/quickstart.mjs
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

const base = (process.env.SEMREMBER_URL || "http://127.0.0.1:7379").replace(/\/$/, "");

async function request(method, path, body) {
  const response = await fetch(base + path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(5000),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${result.error}`);
  return result;
}

const info = await request("GET", "/v1/info");
assert(info.semantic_reads && info.semantic_updates, "Start with --train examples/incidents.csv");
const decision = await request("POST", "/v1/escalate", { text: "certificate has expired" });
assert.equal(decision.handled, true);
assert.equal(decision.action, "renew_certificate");
console.log("Decision:", decision);

const key = `quickstart/javascript/${randomUUID()}`;
// Encode the whole key once, including every slash.
const path = "/v1/kv/" + encodeURIComponent(key);
try {
  const write = await request("PUT", path, { value: "review" });
  assert.equal(write.previous, null);
  assert.equal(write.semantic_updated, true);
  assert.equal((await request("GET", path)).value, "review");
  const hits = await request("POST", "/v1/similar", { text: "certificate has expired", top_k: 2 });
  console.log("Related records:", hits.candidates);
} finally {
  await request("DELETE", path);
}
console.log("Encoded-key write, read and cleanup: OK");
