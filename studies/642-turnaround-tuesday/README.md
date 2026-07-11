# Study 642 — Turnaround Tuesday 🔄

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a down Monday predict a Tuesday bounce? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Down-Monday Tuesdays average **+21.34 bps** vs +7.09 bps unconditionally and −2.21 bps after an *up* Monday — Welch *t* = **+2.38** (vs unconditional) / **+3.85** (vs up-Monday), Newey-West *t* = **+3.19 to +3.80**, random-pair placebo **p = 0.00005** over 20,000 draws, robust to dropping the 2008-09 and 2020-03 crisis windows (*t* = +3.14), and — the honest check — genuinely **Monday-specific**: pooled across the other four weekday reversal pairs the effect is nothing (*t* = +0.66). No decay pre/post-2000 (diff *t* = +0.15). |
| **Tradability** — can you get paid for it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Buying the down-Monday close and selling the Tuesday close nets **+11.3 bps/event** at 5 bps one-way cost (*t* = +2.14, Sharpe 0.37, ≈ +2.3%/yr) — barely certified — and is **statistically dead** at 10 bps (*t* = +0.25, Sharpe 0.04, ≈ +0.3%/yr). Only ≈ 8% of trading days carry a position, and a single worst event (−4.6% to −4.7% net) erases years of edge. |
| **Is this really Monday, or generic "buy the dip"?** | ![Confirmed](https://img.shields.io/badge/Monday--specific%3F-Confirmed-8b949e?style=flat-square) | The same down-day → next-day split on the other four weekday pairs (Tue→Wed, Wed→Thu, Thu→Fri, Fri→Mon), pooled, is statistically nothing (*t* = +0.66) — while Mon→Tue alone clears *t* = +3.85. It really is Monday's down day that carries the signal. |

> **In one sentence:** the market really does bounce back on Tuesday after a down
> Monday — **+21.34 bps** vs a flat unconditional Tuesday, Welch/HAC *t* up to +3.9,
> placebo *p* = 0.00005, and the pattern is *specifically* about Monday (not generic
> "buy the dip") — but the net edge only barely clears the bar at retail costs and
> is dead at slightly higher ones, so the honest read is **real signal, fragile
> paycheck**.

## What we tested

The trading-desk staple, stated the way it's told: *"if Monday closes down, buy the
close — Tuesday tends to bounce back."* We take it literally on **total-return SPY
(1993-01-29 → 2026-06-30)**: E[Tuesday close-to-close return | prior Monday's own
close-to-close return < 0], contrasted against the unconditional Tuesday mean, against
Tuesdays that follow an *up* Monday, and against all trading days — Welch *t* for each
contrast plus a Newey-West dummy-regression cross-check, a Wilson-bounded hit rate and
a 20-seed × 1,000-draw random-pair placebo. A **Monday-specificity check** runs the
identical split on the other four weekday pairs to rule out generic short-horizon
reversal wearing a calendar costume. A timer buys SPY at the down-Monday's own close
(zero look-ahead — the flag *is* that close) and sells at the Tuesday close, net of
2 × one-way cost × NAV. A 20-seed synthetic null plus a planted-bounce world proves the
machinery. **Dedup:** siblings [224-monday-effect](../224-monday-effect/) (the
*unconditional* Monday level — Monday is positive on this tape, `NONE`) and
[90-weekend](../90-weekend/) (the day-of-week *level* table, which names "turnaround
Tuesday" but never conditions on the prior Monday's sign) test day-of-week **means**;
[116-power-hour](../116-power-hour/) tests an **intraday** continuation/reversal on a
different clock. None of them test the **conditional** claim — this study does. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy the dip on Monday, sell Tuesday" is a real pattern, why it's specifically about Monday, and why the paycheck is thinner than the pattern |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/HAC splits, the random-pair placebo, the five-weekday-pair specificity test, the crisis-window robustness cut, the era contrast, the cost sweep, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`turnaround_tuesday/`](turnaround_tuesday/). SPY total-return is a single
continuous index series (no survivorship). **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
