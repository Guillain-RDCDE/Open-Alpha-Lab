# Study 802 — Stock-Split-Modern ✂️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the classic *positive* post-split drift survive in the post-2020 mega-cap era? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Market-adjusted (SPY-hedged, total-return) abnormal CARs: **all 44 forward splits since 2010 average −1.83% at 3m (HAC *t* = −0.82)** — no positive drift. Even the pre-2020 slice, where the sign is *right* (+3.37%), can't certify on this basket (*t* = +1.48). Post-2020 the sign **flips**: −5.78% at 3m (*t* = −2.21). No cohort shows the claimed positive drift on the real tape. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | ~1.6 events/yr. The modern "edge" is negative-or-nothing; the **seven headline mega-cap splits** (TSLA/NVDA/AMZN/GOOGL/AAPL) average **−4.95% at 3m with *t* = −0.77** — statistically nothing — and a single event (Tesla, Aug-2022, −34%) dominates the mean. No borrowable, scalable, certifiable edge in either direction. |
| **Is the near-term dip a real inverse signal?** | ![Busted](https://img.shields.io/badge/Real_inverse_drift%3F-Busted-8b949e?style=flat-square) | The post-2020 −5.78% at 3m *does* clear \|*t*\| = 2 (and a random-date placebo agrees, *p* = 0.999), but it is **fragile**: it evaporates by 6m (*t* = −0.76), turns **+5.44% positive by 12m** (*t* = +0.97), and vanishes on the seven actual mega-caps (*t* = −0.77). A horizon-fragile "sell-the-news" blip on a survivorship-selected basket, not a robust inverse drift. |

> **In one sentence:** the famous post-split drift is a pre-2020 relic that never even
> certified on this basket, and in the post-2020 mega-cap era the market-adjusted return
> turns mildly *negative* for a month or two after the ex-date — but with ~1.6 events a
> year, one Tesla-2022 crash driving the mean, the seven headline mega-caps averaging a
> statistically-nothing −5%, and the effect flipping positive by twelve months, it is
> **tiny-N idiosyncratic noise wearing a memorable narrative.**

## What we tested

Ikenberry, Rankine & Stice (1996) found stocks drift up ~+8% in the year after a split
*announcement* — a costly-signal story — and the retail version of it got a spectacular
poster child when **Tesla, Nvidia, Amazon and Alphabet** all split and all soared
2020-2024. We steelman the modern claim: using `yfinance` split events (the "Stock
Splits" action series) on a **30-name large-cap basket**, we take **44 forward splits
(ratio ≥ 1.5) since 2010** and measure **market-adjusted abnormal returns** — the stock's
total return minus SPY's over the identical window — around the ex-date ([−1, +1]) and at
1/3/6/12-month horizons after it. Market-adjustment is the whole point: it strips out the
2020-24 bull run so "the stock went up after its split" can't masquerade as a signal when
it is really just beta. A Newey-West HAC *t* on the date-ordered CARs is the decisive
statistic; a random-non-split-date placebo, an era split (pre/post-2020) and a costed
timer round it out, and a seeded synthetic positive control proves the machinery.
**Two honesties are stated loudly:** yfinance gives the *effective* (ex) date, not the
*announcement* (a strictly weaker window — same caveat as our sibling), and the basket is
a **current-membership** list of today's winners, a survivorship/selection bias that
points *for* the drift — yet the drift still isn't there. **Dedup:**
[142-split-drift](../142-split-drift/) tests the same anomaly on a 2000-2025 basket and
also finds `NONE` (post-*effective*); [250-reverse-split](../250-reverse-split/) tests the
opposite corporate action (the "kiss of death"). This study is the **modern post-2020
mega-cap re-test**, with an explicit tiny-N caveat and an SPY-hedged abnormal-return lens.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the mega-cap splits *looked* magical, what "market-adjusted" removes, the enormous spread across just seven events, why the near-term dip isn't a signal you can trade |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the cohort table, the HAC horizon sweep, the pre/post-2020 era difference, the random-date placebo, the costed long/short timer, survivorship accounting, and the seeded synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`stock_split_modern/`](stock_split_modern/). Market-adjusted total-return
abnormal CARs; **survivorship/selection named on the Signal axis** (current-membership
basket). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
