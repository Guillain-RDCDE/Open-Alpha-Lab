# Study 667 — Negative-Volume-Index 📉📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does NVI>EMA forecast beyond the base rate? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Fosback's own annual framing (NVI state at year-end predicting the following year) gives P(year up \| NVI>EMA) = **70.3%** (n=37) vs an **unconditional base rate of 73.0%** (n=74, 1952–2025) — *worse*, not better; a 20,000-draw label-shuffle placebo gives **p = 0.790**. A higher-power daily cross-check (21/63/252-day forward returns) never clears the desk's *t* ≥ 2 bar once the overlapping-return trap is corrected: Newey-West *t* tops out at **+1.18** (the naive Welch *t*, up to +6.83, is exactly the illusion the HAC bar exists to catch). |
| **Tradability** — could you time it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A long/flat SPY timer (long when NVI>EMA) **loses to buy-and-hold**: active spread **−0.84 bps/day at HAC *t* = −2.44** (net, 5 bps costs; gross barely different at *t* = −2.38 — costs aren't the story). A circular-shift placebo shows 98.5% of random re-timings of the same position path beat the real rule. |
| **"NVI>EMA = ~96% odds of a bull market"?** | ![Busted](https://img.shields.io/badge/96%25_bull_odds%3F-Busted-8b949e?style=flat-square) | Measured exactly the way Fosback measured it, conditioning on NVI's regime adds **nothing** over just knowing stocks drift up most years — the famous number does not appear on 74 years of S&P 500 data, and the honest replication sits *below* the base rate it should be beating. |

> **In one sentence:** Fosback's Negative Volume Index — "smart money" accumulates
> quietly on low-volume days, so NVI above its 1-year EMA means a bull market ~96% of
> the time — collapses on replication: the 74-year annual test lands at 70.3% against
> a 73.0% unconditional base rate (placebo *p* = 0.79), the higher-power daily
> cross-check never clears *t* = 2 once overlap is corrected, and a costed SPY timer
> built on the rule actually **loses** to buy-and-hold at *t* = −2.44.

## What we tested

We build **NVI** exactly as Fosback defined it — cumulate the index's return only on
days volume falls versus the prior day, base 1000 — against its own **255-session
(1-year) EMA**, on **^GSPC** daily OHLCV back to 1950-01-03 (yfinance, the first
session with reported volume) and, for the tradable third axis, **SPY** total-return
OHLCV since 1993. The headline test replicates Fosback's own annual bull/bear framing
on 74 complete calendar years, benchmarked against the **unconditional base rate** the
folklore never reports, plus a label-shuffle placebo; a higher-power daily cross-check
(21/63/252-day forward returns) adds a Newey-West HAC *t* specifically to expose the
overlapping-return trap that the raw, frequently-quoted Welch *t* falls into. The third
axis is a costed long/flat timer on SPY (one execution lag, one-way costs × NAV),
raced against buy-and-hold with a circular-shift placebo. A deterministic synthetic
world with a tunable "quiet days precede drift" knob proves the detector is unbiased
(null silent across 20 seeds, planted effect lights up at *t* = 2.90). **Dedup:**
siblings [492-up-down-volume](../492-up-down-volume/) (cross-market breadth, not a
single-instrument indicator), [109-obv-divergence](../109-obv-divergence/) (cumulates
volume, not return), [511-volume-momentum](../511-volume-momentum/) (cross-sectional
double-sort, not single-instrument timing) and [116-power-hour](../116-power-hour/)
(an unrelated intraday claim) — none of them test Fosback's specific NVI construction.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "smart money" volume folklore claims, why a 96% number should make you suspicious before you even open the data, the base-rate reveal, and why the timer actually loses money |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the annual replication + Wilson intervals + label-shuffle placebo, the overlapping-return trap (naive vs HAC *t*), the costed timer with its circular-shift placebo, the sample-half split, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`negative_volume_index/`](negative_volume_index/). ^GSPC is a price index
(no dividends, no survivorship); SPY is total-return (`auto_adjust=True`). NVI is built
on each series' own reported volume — a named proxy for the NYSE composite tape
Fosback used in 1976. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
