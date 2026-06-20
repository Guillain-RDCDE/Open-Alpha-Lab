# Study 345 — Survivorship-Bias 🪦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the strategy have a real edge? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On the honest FULL tape (firms that delist *included*) the buy-losers rule has **no edge**: HAC *t* = **+1.38** at a realistic 30% deletion rate. The "significant" read only appears once the dead names are deleted. |
| **Tradability** — is there anything to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The edge a survivors-only backtest advertises evaporates the instant you put back the firms that actually went to zero. |
| **Does survivorship bias manufacture an edge?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Toggle the dead names off and the *same rule on the same data* jumps from *t* = 1.4 (no edge) to *t* = 2.8 (a "discovery"). Monotone in the death rate; correctly vanishes at zero. |

> **In one sentence:** run one identical strategy on a tape that keeps the companies that died and on the survivors-only tape you get from a current-membership universe, and the deletion alone manufactures a statistically "significant" edge out of nothing — survivorship bias isn't a footnote, at a realistic deletion rate it's the whole result.

## What we tested

The textbook warning: a backtest built on *today's* index membership projected backwards never contained the firms that delisted, merged, or went bankrupt — so the losers were deleted before the test ever saw them, biasing every long-biased read upward (Brown–Goetzmann–Ibbotson–Ross 1992; Shumway 1997 on the CRSP delisting bias). Believers wave this away as a small adjustment. We make it the *subject*: one strategy — a contrarian "buy the past-12-month losers" book — run on the **same** panel in two states, **FULL** (every firm, including the ones that take a −80% delisting loss and leave the tape) versus **SURVIVORS** (the dead names removed). The deterministic synthetic core lets us dial the death rate from 0 to 45%; a real large-cap tape (itself survivors-only, via the opt-in guard) is the illustration.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why deleting the dead invents a winning strategy, in plain language — and the control that proves the harness isn't cheating |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the death-rate sweep, HAC *t* and block-bootstrap CI on FULL vs SURVIVORS, the zero-death control, the delisting-loss mechanism, and the biased real tape |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`survivorship_bias/`](survivorship_bias/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
