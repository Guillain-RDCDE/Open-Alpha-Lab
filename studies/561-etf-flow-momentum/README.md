# Study 561 — ETF-Flow-Momentum 💸

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do the biggest-inflow ETFs keep winning? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The claim has real academic support (Ben-Rephael-Kandel-Wohl flow persistence) but competes head-on with an equally documented **reversal / crowding** effect (Frazzini-Lamont "dumb money"; Ben-David-Franzoni-Moussawi ETF mean-reversion) — the *sign* is genuinely disputed. And **no free real tape can measure ETF creation-unit flows cleanly** (yfinance exposes one stale shares-outstanding scalar, not a daily history), so this is **synthetic-only** and can never earn `REAL`. On the synthetic **null** the flow spread is a coin: annualised **+2.9%**, one-sample *t* **+0.78**, placebo *p* **0.42**, hit **51.7%**. Data-availability limit named on this axis. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | You cannot cheaply *measure* the signal at all — honest ETF flow needs a paid daily shares-outstanding feed. Even the planted-edge illustration only shows ETF frictions are mild (gross **+16.0%** → net **+14.2%** after 3 bps/leg + 40 bps borrow); that is a *planted* world, not evidence. Nothing measurable to trade on a retail stack. |

> **In one sentence:** flow momentum — buy the sector ETFs pulling in the most new money — is a real-ish idea in the literature, but it fights a well-documented crowding *reversal*, and because no free feed can reconstruct honest ETF creation-unit flows this study is synthetic-only: the engine provably catches a planted edge of *either* sign (control mean-*t* +0.25 at the null, up to ±6 as the effect grows), but on the honest null the spread is indistinguishable from a coin (*t* +0.78, placebo *p* 0.42) and there is nothing cheap to measure or trade.

## What we tested

The **flow-momentum** claim: sector/asset ETFs with the largest recent net *creation-unit inflows*
go on to outperform their peers — versus the **reversal/trap** view that heavy inflows mark
over-extended, crowded sectors that subsequently lag. We build a **deterministic synthetic ETF-flow
panel** (16 ETFs × 120 months, seed 561) whose single knob `flow_alpha` plants either sign of the
flow→forward-return relation (positive = momentum, negative = the trap, zero = null). The engine
sorts each month into an inflow leg (long) and an outflow leg (short), forms a monthly long-short
spread, and reports a **one-sample *t*** on that series, a **within-month label-shuffle placebo**
null, a Fama-MacBeth-style pooled slope, monthly-rebalance costs + a short borrow, a leg-fraction
robustness sweep, and a **seed-robust (25-seed) synthetic control** proving the detector catches a
planted edge of either sign and stays flat at the null. **This study is synthetic-only** — the free
real data to measure ETF flows honestly does not exist (yfinance's `sharesOutstanding` is a single
stale scalar), so per the desk's rubric it is capped at `WEAK`/`NONE` and the limitation is stated
on the Signal axis, like the [lego-returns](../../273-lego-returns/),
[whisky-cask](../../275-whisky-cask/) and [sneaker-resale](../../276-sneaker-resale/) studies.
*Distinct from the ETF price-vs-NAV study [378 ETF-NAV-Premium](../../378-etf-nav-premium/) and the
return-based momentum studies [507](../../507-cross-sectional-momentum/) /
[518](../../518-time-series-momentum/): the predictor here is **flow**, not price gaps or lagged
returns.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what ETF flows are, why chasing them might work (or be a trap), why we can't measure them cheaply, the honest coin-flip null, and the synthetic control |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the flow sort, the one-sample *t* on the monthly spread, the label-shuffle placebo, the Fama-MacBeth pooled slope, the leg-fraction sweep, costs & borrow, and the seed-robust both-signs synthetic control |

The reproducible synthetic run (null panel fp `9cd505b99a53`, planted-edge panel fp `dca0e2e0ba52`,
as-of 2026-06-30) is in [docs/results.md](docs/results.md); the deterministic offline generator is
[`etf_flow_momentum/data.py`](etf_flow_momentum/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`etf_flow_momentum/`](etf_flow_momentum/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
