# Study 331 — Fifty-Two-Week-Range

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The range-position Q5−Q1 spread clears the bar only at the 1-day horizon (HAC *t* = **+2.00**, bootstrap CI barely positive, bid-ask-bounce-tainted) and is noise at 5/21/65 days (*t* = +0.67 / +0.32 / +0.45). Long-only it ties the basket (+12.0% vs +11.3%/yr, *t* = +0.26). No robust standalone edge. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The largest gross spread (+12.8 bps, *t* = +0.32) is already insignificant and turns **negative** by ~15 bps round-trip. There is nothing to charge costs against. |
| **Does the low add anything over the high alone?** | ![Confirmed, but hollow](https://img.shields.io/badge/Better_than_the_high%3F-Confirmed,_but_hollow-8b949e?style=flat-square) | Range-position *does* beat the high ratio (paired-difference HAC *t* up to **+4.04**; spanning-regression incremental *t* = +2.16) — but only by dodging the high ratio's *inverted* drag on this large-cap survivor sample. Better than a losing signal is not a winning signal. |

> **In one sentence:** adding the 52-week *low* as a second anchor genuinely beats distance-from-the-high alone — yet it wins by sidestepping a losing signal, not by earning one, so on this sample range position has no standalone edge and dies on the first basis point of cost.

## What we tested

George & Hwang (2004) showed that *nearness to the 52-week high* (`close / high_52w`) predicts returns; practitioners go further and read **where in its 52-week range** a stock sits — `(close − low_52w) / (high_52w − low_52w)` — as a richer, "more confirmed" momentum signal because it anchors on both endpoints at once. We put that generalisation on the stand with the one test that makes it a new study rather than a re-run of [236 (high)](../../236-fifty-two-week-high/) or [202 (low)](../../202-fifty-two-week-low/): a **head-to-head horse race** — a paired-difference HAC test and a Fama-MacBeth spanning regression — to isolate the *incremental* value of the second anchor on a 20-name S&P 500 large-cap basket (2013-2026, **survivorship-biased**), with a deterministic synthetic control that plants the discriminating edge separately from a shared confound.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, why "where in the range" *feels* smarter than "off the high", the spread that isn't there, why beating a losing signal isn't winning, the cost wall |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* per horizon, the paired spread-vs-spread difference, the Fama-MacBeth spanning regression, block-bootstrap CIs, the cost sweep, the two-knob synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fifty_two_week_range/`](fifty_two_week_range/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
