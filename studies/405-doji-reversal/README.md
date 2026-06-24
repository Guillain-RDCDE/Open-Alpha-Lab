# Study 405 — Doji Reversal 🕯️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the doji precede a reversal? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The *absolute* 5-day reversal clears the bar (**+5.98 bps**, HAC *t* = **+2.31**) — but the **doji-minus-base-rate** delta is **negative at every horizon** (−2.60 bps at 5d), a label-shuffle placebo beats the dojis **84%** of the time (*p* = 0.84), and the **SPY-only** doji is *negative* (−7.99 bps). The number is the baseline, not the candle. **Survivorship** is named here: a fixed surviving-names basket (ambiguous direction). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No doji-specific gross edge to pay costs from; net of a 2 bps round-trip + short borrow the short horizons go negative and the rest is just base-rate reversion minus friction. |
| **A reversal signal?** | ![Misattributed](https://img.shields.io/badge/Reversal_signal%3F-Misattributed-8b949e?style=flat-square) | The reversal is real but the *attribution* is wrong: it belongs to generic short-horizon mean-reversion (present on every bar, and the trend-filter myth-check confirms it) — not to the candle of indecision. |

> **In one sentence:** the doji's forward "reversal" is real-looking only because short-horizon prices mildly mean-revert in general — pinned against the unconditional against-the-move base rate the doji *underperforms* at every horizon (delta −2.6 bps at 5d, label-shuffle *p* = 0.84, SPY-only negative), so the candle of indecision adds nothing and the absolute +5.98 bps / *t* = 2.31 is just the baseline wearing a costume.

## What we tested

We detect every **doji** (real body ≤ 10% of the day's high-low range) across a fixed basket of **SPY + 28 long-listed US large-caps** (yfinance daily, 2001–2025, cache-first), tag the 2-day move into each one, and take the textbook **reversal** bet *against* that move at the next session's open — reading the forward **1 / 3 / 5 / 10-day** return. The Signal axis tests the doji return against zero with a one-sample HAC *t* **and against the unconditional base rate** (the same bet on every bar), plus a 2,000-draw label-shuffle placebo and a detector-cut robustness sweep; Tradability charges a 2 bps round-trip + short borrow. A myth-check asks whether a trend/volume filter rescues it (it doesn't — it relabels the base-rate reversion). A deterministic synthetic panel with a *planted* doji-specific reversal confirms the harness can detect a real effect when one exists. Survivorship (the basket is names still trading in 2026) is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a doji is, why "indecision" *looks* like a turn, why a random day beats it, and why net of costs there's nothing there — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the OHLC detector, forward 1/3/5/10-day reversal vs the base rate, one-sample HAC *t* + label-shuffle placebo, the detector-cut artefact, the trend/volume filter myth-check, and a synthetic faithful-harness control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`doji_reversal/`](doji_reversal/). Doji = real body ≤ 10% of range; reversal bet is against the 2-day prior move, entered at the next open. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
