# Study 315 — Sovereign-Downgrade 🏛️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The 2011 reaction was real and brutal (−6.8% on the first tradeable bar) — but it is **one event**. Across all three downgrades the CAR is indistinguishable from random non-event windows (permutation *p* = 0.11 at one day, ≈0.45 over a month); the naive HAC *t* < −2 is an artefact of n = 3. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | "Sell the downgrade" **loses money gross and net** at every horizon: the one big move was an untradeable post-close gap, and shorting after it bleeds the market's +10.8%/yr drift plus borrow. The only profitable version needs look-ahead. |
| **Sell the downgrade?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A single Monday in August 2011 is the entire myth — by the time anyone could act on the news the drop was already in the tape, and the next two downgrades barely registered. |

> **In one sentence:** stripped of the 2011 anecdote, a US sovereign downgrade is a non-event for stocks — and even in 2011 the crash was an un-tradeable overnight gap, so "sell the downgrade" is a mirage you can only win with hindsight.

## What we tested

Every debt-ceiling fight revives the same warning: *if the rating agencies cut the US, stocks
will crash — so sell the downgrade.* The reference point is always the ~6.7% S&P 500 drop on
the Monday after **S&P's August 2011 cut to AA+**. We take the claim literally as a directional
**abnormal-return event study** on SPY total return: line up all **three** times a major agency
cut the US sovereign rating one notch from the top (S&P 2011, Fitch 2023, Moody's 2025 — a
hardcoded, press-release-sourced table), measure the cumulative abnormal return over the windows
*into* and *out of* each announcement, race them against a **synthetic control** of 3,000 random
non-event windows, and score the naive short trade net of costs and borrow. This is the
directional-equity sibling of the volatility-angle [Study 312 — Debt-Ceiling](../../312-debt-ceiling/).
The offline core and tests run on a deterministic synthetic tape that plants (or withholds) a
downgrade dip.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the 2011 anecdote, the other two downgrades that did nothing, and why you couldn't have traded it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-event CARs, naive *t* vs permutation *p*, the synthetic control, the short-trade timing trap, the positive control |

The fingerprinted real run lives in [docs/results.md](docs/results.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/) + [`sovereign_downgrade/`](sovereign_downgrade/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
