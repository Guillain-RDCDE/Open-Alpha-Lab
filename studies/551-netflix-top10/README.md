# Study 551 — Netflix-Top10 📺

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does engagement momentum predict returns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | **No real tape exists** (no free point-in-time Top-10 hours feed) so this is synthetic-only, capped at `NONE`/`WEAK`. On the synthetic *null* the seed-551 slope prints **OLS *t* −2.03** — a **false positive**: overlapping 4-week forward windows inflate the naive SE. The overlap-robust **Newey-West *t* is −1.48**, and across 20 seeds the mean NW *t* is **+0.11 with 0/20** clearing \|*t*\| ≥ 2. |
| **Tradability** — does timing on it pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Timing on the (noise) signal *loses* to buy-and-hold: long/flat **−3.6%/yr** (net **−4.4%**) vs buy-and-hold **+21.8%**; long/short **−25.7%/yr** after a borrow. And the whole trade rests on data you cannot actually buy. |
| **"Alt-data edge?"** | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The XLY spillover leg is flat (*t* −0.47); the "edge" is a single-seed overlap artifact, and any real engagement report lags the viewing week and is widely covered — priced in before a retail reader could act. |

> **In one sentence:** streaming-engagement momentum is a seductive alt-data story with **no free real tape to test it on** (so it can never be `REAL` here), and even the synthetic *null* prints a single-seed "significant" slope that **evaporates** under an overlap-robust standard error and seed-averaging (mean NW *t* +0.11, 0/20 seeds clear the bar) — a textbook overlapping-window false positive, not a signal.

## What we tested

The claim: **streaming-engagement momentum** — how fast the world's hours-watched on Netflix's
Top-10 are accelerating — *predicts* NFLX and consumer-discretionary (XLY) forward returns. Because
there is **no free, machine-readable, point-in-time Top-10 hours feed** for a no-key retail stack
(the series is short, 2021+, and methodology-revised), we build a **deterministic synthetic weekly
world** ([`netflix_top10/data.py`](netflix_top10/data.py)) with a single knob (`beta`) that plants
or removes the engagement→return link, and we state that data-availability limitation openly on the
SIGNAL axis. The engine runs a **predictive regression** (slope + OLS *t* + an overlap-robust
Newey-West *t*), a **label-shuffle placebo**, a **long/flat and long/short timing backtest** with
one-way costs and a short borrow, a **forward-horizon robustness sweep**, and a **seed-robust
synthetic positive control** (25 seeds) that proves the engine catches a planted link and stays flat
at the null. The central lesson: on **overlapping** forward windows the naive *t* is untrustworthy —
the honest, seed-averaged view is a flat zero. *Distinct from the sentiment alt-data studies
([257 AAII](../257-aaii-sentiment/), [335 Buzz](../335-buzz-sentiment-etf/),
[392 Glassdoor](../392-glassdoor-sentiment/)) — here the signal is streaming engagement and the
lesson is the overlapping-window false positive.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what streaming-engagement momentum is, why there's no real data to test it, and why the one "significant" result is a mirage |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the predictive regression, the OLS-vs-Newey-West gap, the placebo null, the seed-robust flat, the horizon-overlap artifact, costs + borrow, and the synthetic positive control |

The fingerprinted synthetic headline run (null world, seed 551, 197 weekly rows, world fp
`a31af2c4d9d7`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); there is no real tape by
design (the caveat is executable in [`netflix_top10/data.py`](netflix_top10/data.py)).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`netflix_top10/`](netflix_top10/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
