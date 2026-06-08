# Study {{NN}} — {{Title}} {{optional single emoji}}

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

<!--
  ───────────────────────────────────────────────────────────────────────────
  THE DESK FRONT-CARD — the README is a SYNTHETIC DASHBOARD, not a document.
  It answers, in ~30 lines: what's the verdict, what did we test, where's the
  proof. The full demonstration — the 7 narrative beats, every number, every
  figure — lives ENTIRELY in the two notebooks, so we never tell the same
  story twice. If a paragraph here restates what a notebook shows, cut it.

  Keep the four sections below. The full rationale is in ../../METHODOLOGY.md.
  Delete this comment in real studies.
  ───────────────────────────────────────────────────────────────────────────
-->

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | {{🟢 `REAL` · 🟡 `WEAK` · 🔴 `NONE`}} | {{one line — the decisive number}} |
| **Tradability** — does it survive costs, capacity, scale? | {{🟢 `INVESTABLE` · 🟡 `FRAGILE` · 🔴 `MIRAGE`}} | {{one line}} |
| {{*optional 3rd axis specific to the study, e.g. "Manipulation?"*}} | ⚪ `{{STAMP}}` | {{one line}} |

> **In one sentence:** {{the whole study compressed to a single honest line}}.

## What we tested

{{2–4 sentences, no more. State the claim the way its *believers* state it — at full
strength, steelmanned, with a link to whoever's making it (paper / book / viral
thread). Then the one-line setup: the data and the apparatus. NO findings, NO
teardown, NO numbers-dump — that's the notebooks' job.}}

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes in plain language, the tricks that inflate the headline |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full machinery: inference, confounds, alpha-vs-beta, capacity |

{{ONE optional sentence linking any OTHER deliverables that actually exist —
paper/, RESPONSE.md, docs/results*.md (the fingerprinted real run), docs/extensions.md
(beat-7 forks), examples/, _data/PROVENANCE.md. Omit the sentence entirely if none.}}

---

*{{if docs/references.md exists}}Sources & literature map: [docs/references.md](docs/references.md). {{/if}}Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*

<!--
  STAMP COLOURS: Signal REAL→🟢 WEAK→🟡 NONE→🔴 (mixed→🟡).
  Tradability INVESTABLE→🟢 FRAGILE→🟡 MIRAGE→🔴.
  Optional 3rd "myth-check" axis (BUSTED / MISATTRIBUTED / NOT SUPPORTED /
  CONFIRMED …)→ ⚪ (it's a qualitative verdict, off the green/red scale).
-->
