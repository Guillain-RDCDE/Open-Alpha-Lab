# Study 671 — Special K 🔀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the crossover flag major cyclic turns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | 51 bull/51 bear crossovers on 33 years of SPY: the only individually significant result (126-day post-bull-crossover window, Newey-West *t* = **−2.37**) is **wrong-signed** — "buy" crossovers precede *below*-average returns — confirmed by a random-timing placebo (**p = 0.88**, worse than random 88% of the time). On 64 years of ^GSPC (price-only, 2× the crossovers, every post-war bear market) the effect vanishes in both directions (\|*t*\| < 0.6). No real tape supports the claim. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The long/flat timer underperforms buy-and-hold on **three independent tapes** — SPY daily **−7.48%/yr** (HAC *t* = −3.15), ^GSPC **−4.47%/yr** (*t* = −2.81), SPY weekly **−7.80%/yr** (*t* = −3.51) — even at **zero cost**, and no parameter scale (0.7×–1.3× Pring's periods) closes the gap; it only approaches buy-and-hold by trading almost never. |
| **"Flags major cyclic turns"?** | ![Busted](https://img.shields.io/badge/Flags_major_cyclic_turns%3F-Busted-8b949e?style=flat-square) | Neither the event study nor the timer supports it on any of three real tapes, and a synthetic planted-regime-cycle control proves the harness *can* detect a genuine multi-year cycle (spread flips from *t* = −1.47 to *t* = +2.57) — a true negative, not a broken test. One honest concession: Special K trades **4.4× less often** than its own sibling KST for a marginally better Sharpe — the "reduced whipsaw" engineering claim holds, it just doesn't turn a losing rule into a winning one. |

> **In one sentence:** Martin Pring's twelve-ROC "reduced-whipsaw KST" really does whipsaw less than plain KST (4.4× fewer trades, marginally better Sharpe) — but its crossovers carry no forward-return information on three independent real tapes (the one significant number is wrong-signed, and a random-timing placebo and a 64-year price-only cross-check both say so), and the long/flat timer it implies loses to buy-and-hold and to a plain 200-day moving average on every tape, at every cost, at every parameter scale tested — a clean **None x Mirage**.

## What we tested

We build Pring's daily **Special K** — twelve SMA-smoothed rate-of-change series (10-to-530-day lookbacks, weighted 1-2-3-4 across four bands) crossed against a 100-day signal SMA (StockCharts ChartSchool canonical parameters) — and test the "major cyclic turn" claim three ways: a **post-crossover event study** (Newey-West *t*, lags = horizon, plus a Coppock-style random-timing placebo) on SPY total-return and on 64 years of ^GSPC price-only (2× the crossovers, every post-war bear market), a **long/flat timer** raced NET-of-cost against buy-and-hold and a one-line 200-day SMA (with a sign-flip permutation, a cost sweep and a SPY-weekly cross-check), and a **parameter-robustness sweep** across a common period-scaling factor. A deterministic synthetic tape with a **planted multi-year bull/bear regime cycle** — matched to Special K's own 530-day lookback — confirms the engine *can* detect a real cycle when one exists, so the real-tape miss is a true negative. **Dedup:** [426-know-sure-thing](../426-know-sure-thing/) (Special K's direct parent, KST, raced head-to-head on the identical tape), [105-coppock-curve](../105-coppock-curve/) (the random-timing event-study template, one direction only), [425-detrended-price-oscillator](../425-detrended-price-oscillator/) and [427-rate-of-change](../427-rate-of-change/) (single-scale cousins with the same *None x Mirage* shape). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Special K is, why summing twelve ROCs is supposed to catch "the big turns", the wrong-signed crossover chart, and why fewer trades than KST still isn't a free lunch — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Newey-West event-window design, the Coppock-style random-timing placebo, the three-tape (SPY/^GSPC/weekly) timer race, the cost and parameter sweeps, the head-to-head vs KST, and a regime-cycle synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`special_k/`](special_k/). SPY closes are **total-return** (`auto_adjust=True`); ^GSPC closes are **price-only** (no dividends), named everywhere it appears. No survivorship — both are broad indices/index-tracking ETFs. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
