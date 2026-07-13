# Study 733 — Kentucky-Derby-Effect 🐎

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | `CHDN`, the stock that literally runs the Derby, is flat at every horizon (run-up **+0.36%**, *t* = **+0.55**; 1-month **−0.20%**, *t* = **−0.17**; placebo *p* ≥ 0.41) with *full* tape coverage. The market's only whisper is a week-after dip (**−0.62%**, *t* = **−1.43**) that misses the bar raw and fails its drift-neutral placebo (*p* = **0.11**). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The only cut that crosses \|*t*\| ≥ 2 (market/1-week net, **−2.11**) is an arithmetic mirage — costs charged against an *already-negative* return make its *t* *more* extreme — and it dies under its own placebo (*p* = **0.11**), needing you to short the S&P 25 times for a dip inside the noise. |
| **Sell in May?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The Derby-week market dip is real in *sign* but sits inside the random-window luck cloud (*p* = 0.11–0.37) and swings across the −2 line if you drop a single year (jackknife [−2.42, −1.00]). |

> **In one sentence:** the most Derby-exposed stock on the market does nothing around its
> own race, the broad-market "first Saturday in May" dip is a placebo-failing whisper, and
> the one *t*-stat that looks tradable only got there because subtracting costs made a
> negative number look worse.

## What we tested

Every May, retail-forum and almanac folklore floats two versions of a "Kentucky Derby
effect": that the first-Saturday-in-May race is a broad-market seasonal (it sits right on
the "Sell in May" boundary), and — more sharply — that **Churchill Downs Inc. (`CHDN`)**,
the company that owns and operates the Derby, gets a pop around its marquee event. We
hardcode all 26 Derbys 2000→2025 (2020's COVID-postponed September running flagged as a
named quirk), and run an event study on `SPY` (the market seasonal, drift-removed) and on
`CHDN`'s abnormal return vs `SPY`, with a documented Saturday-race execution lag, a
calendar-known run-up window, a costed capture, a multi-seed random-window placebo, and a
synthetic positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the two folklore claims, why the *directly exposed* stock is the cleanest possible test, and the costed trade that only *looks* like it works |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-*t* battery on both legs, the drift-neutral placebo, the jackknife, the event anatomy, the costs-inflate-the-*t* trap, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`kentucky_derby_effect/`](kentucky_derby_effect/). The Derby calendar is
hardcoded from Wikipedia; **no survivorship funnel** (CHDN has traded since the 1990s) —
the caveat named on the Signal axis is **exposure dilution**: CHDN today is a diversified
gaming company for which the Derby is a shrinking slice of revenue. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
