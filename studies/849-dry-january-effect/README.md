# Study 849 — Dry January / Veganuary 🍸🥦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do the January "Dry January" / "Veganuary" waves move the stocks? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The folklore's fingerprint is **directionally perfect** — the alcohol basket (`BUD STZ TAP DEO SAM`) is the year's *most-negative* calendar month (−1.20% abnormal vs SPY, rank **1/12**) and Veganuary's `BYND` the *most-positive* (rank 1/12), sign-consistent across both eras — but **nothing clears |t| ≥ 2**: the well-sampled alcohol drag is only *t* = −1.28 (27 Januaries), the +14.5% plant-minus-alcohol spread rides **7** noisy `BYND` years (*t* = +1.33), and the February "hangover" is absent. A 20-seed synthetic control fires on a genuine +5% seasonal (90%) but 0/20 on the null, so this is a real-but-sub-threshold *hint*, not an effect. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The long-plant / short-alcohol January timer nets **+14.2%/yr** at 5 bps — but on **7 observations at *t* = +1.31**, driven by one small-cap's early tape, once a year, capacity-trivial and inside a multi-cut search. Nothing bankable. |

> **In one sentence:** Dry January and Veganuary leave *exactly the right footprint* — alcohol
> is the year's weakest month and plant-based its strongest — yet at *t* ≈ −1.3 / +1.2 on 27
> and just 7 Januaries the seasonal never clears the bar, a charming directional hint with no
> tradable paycheck.

## What we tested

The two January cultural campaigns — **Dry January** (abstain from alcohol) and **Veganuary**
(go plant-based) — should, if they shift demand hard and predictably enough, show up as a
January seasonal in the **abnormal** return (group − `SPY`) of the relevant names: alcohol
(`BUD STZ TAP DEO SAM`) *down*, plant-based (`BYND`) *up*, staples (`XLP`) flat. We take
**yfinance daily total-return closes for 8 tickers, 1999-01-04 → 2026-06-30**, resample to
monthly, and read the January (and February) abnormal return across the independent yearly
observations — with a one-sample *t*, a Newey-West January-dummy *t*, a Wilson hit-rate, a
twelve-month calendar placebo, a two-era cut, a costed once-a-year timer, and a 20-seed
synthetic positive control. The calendar is fixed and known decades ahead, so the test is
**zero look-ahead by construction**. **Dedup:** [55-summer-lull](../55-summer-lull/) is the
market-wide *summer* lull; [95-holiday-cheer](../95-holiday-cheer/) the December Santa-rally;
[641-sell-in-may](../641-sell-in-may/) the market-wide Halloween/Sell-in-May seasonal;
[723-guacamole-bowl](../723-guacamole-bowl/) a single-date Super-Bowl demand pulse; and
[775-halloween-candy](../775-halloween-candy/) a fixed-date single-name Hershey run-up — none
tests the **January alcohol-vs-plant consumer-demand** theme that is this study's own axis.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the Dry-January / Veganuary calendar *could* move the names — and why the footprint is there directionally but too faint to trade |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the January/February abnormal returns, the Newey-West dummy *t*, the twelve-month placebo, the two-era cut, the costed timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dry_january/`](dry_january/). Daily total-return closes via yfinance, cached under
this study's own `_cache/`. Abnormal returns are `group − SPY`; the plant / spread legs rest
on `BYND`'s **7** Januaries and are flagged as thin-sample throughout.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
