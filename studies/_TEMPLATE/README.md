# Study {{NN}} — {{Title}}

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the shared engine
> (`quantlab/`), start at the [root](../../README.md).*

<!--
  ───────────────────────────────────────────────────────────────────────────
  THE DESK TEMPLATE — every study's front page follows these 7 fixed beats so
  a reader always lands in the same place. Keep the headers; swap the content.
  The full rationale is in ../../METHODOLOGY.md — read it once.

  TWO REGISTERS, ONE PAGE:
    • Plain prose = for the curious. Must read top-to-bottom with zero maths.
    • `> 🔬 **For the quants** — …` = the rigorous layer, inline & skippable.
    • Heavy passages go in a <details> block so the page stays light.

  Delete this comment in real studies.
  ───────────────────────────────────────────────────────────────────────────
-->

## Verdict — read this first

| Axis | Stamp | Why (one line) |
|---|---|---|
| **Signal** — is the effect statistically real? | `REAL` · `WEAK` · `NONE` | {{one line}} |
| **Tradability** — does it survive costs, capacity, scale? | `INVESTABLE` · `FRAGILE` · `MIRAGE` | {{one line}} |
| {{*optional 3rd axis specific to the study, e.g. "Manipulation?"*}} | `{{STAMP}}` | {{one line}} |

> **In one sentence:** {{the whole study compressed to a single honest line}}.

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

---

## 1 · The Claim

> *What people say, in one line — the folk belief, the famous anomaly, the thing
> traders swear by.*

{{State the claim as its believers would, at full strength. Steelman it. Cite who
says it and link the source paper/post.}}

> 🔬 **For the quants** — {{the claim restated as a testable hypothesis: a sign,
> a magnitude, a null. "H₁: mean overnight return > 0 with t > 2 after HAC."}}

## 2 · So What?

> *If it were true, what would it be worth — and what would it imply?*

{{The stakes. Why a reader should care. If the claim holds: how much money, what
it says about how markets work, who's right/wrong. Keep it concrete.}}

> 🔬 **For the quants** — {{the quantitative stakes: gross edge in bps, a back-of-
> envelope Sharpe, the capital it would absorb if real. What breaks downstream if
> true.}}

## 3 · How We'd Know

> *The falsifiable test — announced before we run it, so we can't move the goalposts.*

{{In plain language: the experiment that would confirm or kill the claim, the data
we'll use, and — crucially — what result would make us say "mirage". Name the traps
we're watching for.}}

> 🔬 **For the quants** — the protocol (shared desk rubric, see
> [root](../../README.md)):
> 1. **Decompose / measure** the raw effect (exact identities, no fitting).
> 2. **Robust inference** — Newey-West (HAC) / Lo (2002) SEs, bootstrap CIs,
>    White (2000) Reality Check. *Is it real?*
> 3. **Critique the magnitude** — compounding, units, artefacts, selection.
> 4. **Alpha vs beta** — how much is just risk premium?
> 5. **Execution & capacity** — costs, market impact, the scale where it dies.
> 6. **Verdict** — the stamps, with the numbers behind them.
>
> Engine used: {{list the `quantlab/` modules this study leans on}}.

## 4 · The Teardown

> *We run it. Here's what the data actually says.*

{{The findings, in narrative order, each as a short claim a curious reader can
follow. One bullet = one result = one number. This is the body of the work; let
the figures carry it.}}

- **{{Finding 1, plain words}}.** {{the number, the picture}}.
- **{{Finding 2}}.** …
- **{{Finding 3}}.** …

> 🔬 **For the quants** — {{the rigour behind each bullet: test statistics, CIs,
> the confound you ruled out and how. Point to the figure script and the notebook
> cell that reproduces it.}}

<details>
<summary>🔬 The maths, in full</summary>

{{Optional: derivations, the decomposition identity, the impact model, anything
too heavy for the inline layer. A curious reader never has to open this.}}

</details>

## 5 · The Verdict

> *The two stamps, and the numbers that earned them.*

{{Restate each stamp from the box above, now justified. Be honest about what's
real and what isn't — the dead-ends are the point. If it's a mirage, say exactly
where it dies.}}

> 🔬 **For the quants** — {{the decisive statistics in one place: HAC t, residual
> alpha vs break-even cost, decay slope, capacity figure, posterior — whatever
> closed the case.}}

## 6 · Could You Trade It?

> *The honest money question: if you wanted to actually get paid, what would it take?*

{{The part most repos skip. Walk a reader from "interesting print" to "live P&L":
the venue you'd execute on, the costs you'd really pay, the size before impact eats
the edge, the sizing/risk you'd run — and the candid bottom line on whether it's
worth a desk's time. If the answer is "no", show the line where it flips.}}

> 🔬 **For the quants** — {{break-even cost vs edge, square-root impact capacity in
> \$, turnover × spread × 252, decay-adjusted Sharpe net of costs, the capital curve.
> The conditions under which it *would* pay, stated precisely.}}

## 7 · Going Further

> *Open threads — and how you, the reader, could push this.*

{{What we didn't settle, the next experiment, the variant worth testing, the data
that would sharpen the verdict. An explicit invitation: what to fork, what to
challenge. Link related studies in the queue.}}

- {{Open question / next test}}
- {{A variant or alternative explanation still on the table}}
- {{What a contributor could PR}}

---

## How this study is laid out

| Path | What's inside |
|---|---|
| [`notebooks/01_for_the_curious.ipynb`](notebooks/) | the story + the stakes, plain language |
| [`notebooks/02_for_the_quants.ipynb`](notebooks/) | the full teardown: inference, confounds, capacity |
| [`paper/`](paper/) | {{working paper, if any}} |
| [`docs/references.md`](docs/) | sources + literature map |

The engine that produced every number lives at [`../../quantlab/`](../../quantlab/).
