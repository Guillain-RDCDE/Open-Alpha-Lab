# Study 653 — Dividend-Cut-Drift 📉✂️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a stock that cuts or omits its dividend keep underperforming? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Decades-old academic support (Michaely-Thaler-Womack 1995: **−9.5%/yr** post-omission on 1964-88 data) does not replicate on the modern real tape. 172 real cut/omission events, 1997-2025: post-event CAR[+1..+120] = **−0.31%**, *t* = **−0.14**; Newey-West *t* = **−1.23**; random-date placebo *p* = **0.68**; hit rate **47.7%** — a coin flip. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Neither short-the-cutter (excess-of-matched-exposure *t* = **−1.86**) nor buy-the-cutter (*t* = **+1.73**) clears the *t* ≥ 2 bar net of costs. The dramatic-looking gross numbers (−15%/+15% per event) are six months of ordinary market beta, not a cutter-specific edge — and the short leg carries a documented **−467%** single-event squeeze. |
| **"Never catch a falling dividend"?** | ![Busted](https://img.shields.io/badge/Falling_dividend%3F-Busted-8b949e?style=flat-square) | The point estimate is a near-zero wash, and the two event subtypes even point in *opposite*, individually-uncertified directions: cuts mildly negative (*t* = −1.25), full omissions mildly positive (*t* = +1.55). |

> **In one sentence:** the classic post-dividend-cut drift is a genuine, well-documented 1960s-90s
> finding (Michaely-Thaler-Womack) that simply doesn't show up on 172 real cut/omission events in
> a modern, survivor-biased large-cap basket (1997-2025) — the six-month post-event drift is
> statistically nothing (*t* = −0.14, placebo *p* = 0.68), and neither shorting nor buying the
> cutter clears the bar after costs.

## What we tested

We hardcode a named **101-ticker** basket of mature US dividend payers — deliberately a mix of
long-run stable names and names that visibly cut across four distinct eras (2008-09 banks,
2015-16 energy, 2020 pandemic travel/retail, 2022-24 industrials/staples) — and scan each
ticker's split-adjusted yfinance dividend history for a **cut** (a scheduled payment ≤ 70% of the
prior payment) or an **omission** (a gap ≥ 1.8× the ticker's own typical interval), stripping
one-off special-dividend/stub artifacts that would otherwise manufacture false events. For every
one of the 172 events with a full window on the tape we build a [−20, +120]-trading-day
cumulative abnormal return vs SPY (total-return, so the mechanical ex-date drop never
contaminates the math), test it cross-sectionally and via a Newey-West calendar-time portfolio,
and back-test two honest trades — short-the-cutter and buy-the-cutter — one execution lag (enter
the close the session **after** the event), matched-exposure excess over SPY, real costs and
borrow. Survivorship is named explicitly: the universe is current survivors only, so the true
population effect (including names that later went to zero) is plausibly *more* negative than
what we measure. **Dedup:** siblings [240-dividend-initiation](../240-dividend-initiation/) (the
START side, `NONE` — too few events/year), [143-dividend-capture](../143-dividend-capture/) (the
routine ex-date drop on *uncut* payments), [201-dividend-growth](../201-dividend-growth/)
(consecutive raises, the opposite tail) and [233-shareholder-yield](../233-shareholder-yield/)
(a level-sort composite factor) — none of them tests what happens to a stock in the six months
**after it specifically cuts or omits its dividend.** As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "never catch a falling dividend" sounds so plausible, what a real cut looks like on the tape, and why the classic finding quietly stopped showing up |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the split/special-dividend cleaning, the cross-sectional and Newey-West splits, the cut-vs-omission divergence, the placebo, the short/long capture tests with costs and tails, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dividend_cut_drift/`](dividend_cut_drift/). The universe is a hardcoded named basket
of current survivors (named on the Signal axis); prices and dividends are yfinance total-return
series. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
