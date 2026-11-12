# Maple Shield Insurance — FNOL triage pilot

A first pilot demo for a fictional Canadian property and casualty insurer.

**The product thesis:** Semrember is experiential memory for System One AI. It
remembers exact facts, similar claims, past operational scars, fairness lessons,
and manipulation attempts. Jev / System One supplies calibrated, typed judgment.
Code owns the routing policy, and a licensed human remains authoritative.

> Synthetic data only. This is a routing demonstration, not a production claims
> decision system. It does not deny, approve, or pay a claim.

For the long-form technical write-up, read [`BLOG.md`](BLOG.md).

## The customer problem

A Canadian P&C insurer receives first notice of loss (FNOL) as free text:

- call-centre notes,
- broker emails,
- web forms,
- French-language notes from Quebec,
- documents and photos with their own text.

An adjuster has to decide quickly:

1. Which queue does this belong to?
2. Is there an injury, fraud indicator, or regulatory complaint?
3. Have we seen this pattern before — and did it go wrong?
4. Is the note trying to manipulate the triage system?
5. Does a human need to review before anything happens?

The demo shows a memory-and-judgment layer for that decision.

## What the demo shows

Ten synthetic scenarios:

| Scenario | What it tests |
|---|---|
| Ontario DCPD rear-end | ordinary auto collision |
| Simple glass claim | low-risk fast track |
| Condo water damage | property routing |
| Staged accident indicators | SIU referral |
| FSRA complaint threat | regulatory complaint |
| Prompt injection in a claim note | adversarial input |
| French-language Quebec claim | language parity |
| Calgary hail | catastrophe routing |
| Injury indicators | injury override |
| Incomplete FNOL | request more information |

For each scenario the demo prints:

- **local memory** — similar claims and the local semrember decision,
- **experience memory** — scars, fairness lessons, and attack patterns,
- **System One / Jev** — typed probabilities for route, injury, fraud, complaint,
  injection, routing blockage, severity, and evidence fit,
- **final decision** — queue, human-required flag, reasons, and expected route.

It can also write a self-contained HTML report for a customer walkthrough.

## Architecture

```text
FNOL note
   │
   ├─ semrember claims memory ──────► similar claims + local route
   │
   ├─ semrember experience memory ──► scars / fairness / attacks
   │
   └─ Jev / System One ─────────────► typed judgment
          actionable
          route (Choice over queues + none/insufficient)
          injury
          fraud_review
          complaint
          human_request
          injection
          routing_blocked
          experience_warning
          retrieval_fit
          severity (Score)
   │
   └─ code policy ──────────────────► route to human-owned queue
                                      or human_review
                                      with audit reasons
```

The local engine is fast and private. Jev is called only for the semantic
judgment layer in this pilot. A production integration would call Jev only on
uncertain cases and cache the results.

## Run it

Build the semrember server first:

```bash
make all
```

Offline demo, no external API:

```bash
python3 examples/pilot_insurance_ca/demo.py
```

Live Jev demo:

```bash
python3 examples/pilot_insurance_ca/demo.py --live
```

The API key is read from `TYPESAFE_API_KEY`, or from `--key-file`. If neither is
set, the demo looks for `~/.typesafe_api` or `~/TYPESAFE_API.md`. The key is
never printed or written to the audit output.

Write an HTML walkthrough and a machine-readable audit:

```bash
python3 examples/pilot_insurance_ca/demo.py --live \
  --html-out examples/pilot_insurance_ca/report.html \
  --json-out examples/pilot_insurance_ca/audit.json
```

Run a subset:

```bash
python3 examples/pilot_insurance_ca/demo.py --live --only injection_attack,fraud_pattern
```

Run the offline policy tests:

```bash
make pilot-test
```

## Files

```text
BLOG.md                 long-form technical write-up
demo.py                 starts two semrember services and runs the scenarios
jev_gate.py             System One client + question bank + routing policy
test_policy.py          offline tests for the routing policy
data/claims.csv         synthetic labelled FNOL claims
data/experiences.csv    synthetic scars, fairness lessons, attack patterns
data/actions.json       routing queues and descriptions used by Jev
data/scenarios.json     ten synthetic demo scenarios
data/scenarios_new.json one mixed complaint/water/injury prompt
data/scenarios_10.json  ten additional stress-test prompts
```

`report.html` and `audit.json` in this directory are generated examples from a
live run. Re-running the demo overwrites them.

`claims.csv` trains the claims memory. `experiences.csv` trains a second
semrember memory for experiential lessons. The two memories are intentionally
separate: a routing label is not the same thing as a lesson about a past
failure.

## Canadian pilot guardrails

This demo is designed around a conservative Canadian operating model:

- **No automated denial or payment.** Jev and semrember may route and prioritize;
  a licensed adjuster decides coverage, payment, and adverse outcomes.
- **Human review for high-risk queues.** Injury, total loss, liability, fire,
  catastrophe, SIU, and complaints route to human-owned queues.
- **Privacy minimization.** The demo uses synthetic claims. A production pilot
  must minimize personal information sent to any external model, complete a
  privacy impact assessment, and address PIPEDA and applicable provincial
  privacy law (for example Quebec Law 25, Alberta PIPA, BC PIPA).
- **Cross-border data transfer.** Jev is an external service. The pilot must
  document where data is processed and obtain the insurer's approval before any
  real claimant data is sent.
- **Fairness.** Do not use protected grounds or proxies (for example postal code
  alone) as fraud signals. Test routing and fraud flags across customer groups.
  Every override needs a reason and reviewer ID.
- **Language.** French-language notes must route as accurately as English notes.
  The demo includes a Quebec scenario; a pilot needs a French evaluation set.
- **Complaints.** FSRA, AMF, BCFSA, Ombudsman, lawyer, MPP, and supervisor
  requests go to the complaint team.
- **Audit.** Every decision should retain the local evidence, Jev answers,
  thresholds, model version, reviewer, and outcome.
- **Model risk.** A production rollout needs model risk management, version
  pinning, evaluation, monitoring, drift checks, and change control. For
  federally regulated insurers, align with OSFI E-23 expectations; provincial
  regulators may add their own requirements.
- **No claim decision in the model.** Semrember stores associations and
  experiences, not a coverage determination.

## Pilot success metrics

Measure on a held-out set of real, de-identified claims before expanding:

| Metric | Target direction |
|---|---|
| Correct queue top-1 | above the current manual baseline |
| Time to correct queue | lower median and p95 |
| False fast-track rate | near zero; reviewed weekly |
| Human override rate | decreasing, with reasons captured |
| Complaint routing | 100% to complaint team |
| Injury override recall | 100% on the evaluation set |
| French/English parity | no material gap |
| Injection detection | high recall; false positives reviewed |
| Cost per claim | measured from Jev usage |
| Latency p50/p95 | local path fast; Jev only on uncertainty |

## What is real and what is simulated

Real:

- the semrember engine, HTTP service, exact records, similarity, novelty,
  escalation, and durable-store code in this repository;
- the local memory retrieval and routing evidence;
- the System One request shape and the Jev client in `jev_gate.py`;
- the audit output and HTML report.

Simulated for the pilot:

- the insurer, claims, people, policy numbers, and outcomes;
- the action catalogue and thresholds;
- the policy-admin and claims-system integration;
- the production security, privacy, and model-risk controls.

## Next engineering steps

1. Move the Jev gate into the Crystal server as an optional verifier.
2. Add an episode schema for outcomes, corrections, scars, and provenance.
3. Add a write gate so Jev can reject, redact, or merge a new memory.
4. Add a French evaluation set and a fairness evaluation set.
5. Add an audit log with model version, thresholds, reviewer, and outcome.
6. Add a policy-admin mock so the demo shows the queue assignment end to end.
7. Run the pilot on synthetic claims first, then de-identified historical claims
   with the insurer's privacy and model-risk approval.
