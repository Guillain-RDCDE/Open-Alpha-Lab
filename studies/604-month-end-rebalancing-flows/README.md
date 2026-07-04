# Study 604 — Month-End Rebalancing Flows 🔄

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do rebalancers sell the winner into month-end, then let it bounce? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the sell leg · Weak on the reversal leg.* After months when stocks trounce bonds (top gap quintile), the last-3-day equity-bond spread is **−46.5 bps** (one-sample *t* = **−2.88**), monotone in the gap, permutation placebo **p ≈ 0.022** — but the purely conditional top-vs-bottom differential misses on the pre-registered 3-day window (Welch *t* = −1.60). The first-days "reversal" (+33.5 bps, *t* = +3.28) turns out **unconditional** — no gap dose-response (Welch *t* = +0.95, placebo p ≈ 0.13). No survivorship (broad index vehicles). |
| **Tradability** — can you trade the flow? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | An expanding-threshold flow trade (short SPY/long AGG last 3 days, flip after) nets **+1.72%/yr** at 2 bps/leg (HAC *t* = **+2.52**, 5% of days at risk) — but it is a **1993–2015 phenomenon** (+2.90%/yr, *t* = 3.42) that has earned **−0.07%/yr since 2016** and slips below the bar at 5 bps/leg. Real once, decayed. |
| **"Is turn-of-the-month just this in disguise?"** | ![Busted](https://img.shields.io/badge/TOM_is_flows_in_disguise%3F-Busted-8b949e?style=flat-square) | [Study 89](../89-turn-of-the-month/)'s first-days drift survives at **full strength** (+36.4 bps, *t* = 3.45) exactly in the months where the prior equity-bond gap was near zero and rebalancers had nothing to do; extremes-vs-middle Welch *t* = −0.23. TOM is a calendar drift, not a rebalancing flow. |

> **In one sentence:** the month-end rebalancing story is *half* real — after equity-rich months
> stocks genuinely lag bonds into the close of the month (−46.5 bps, *t* = −2.88, placebo
> p ≈ 0.02) — but the fabled first-days reversal is just the unconditional turn-of-the-month
> drift wearing a rebalancing costume, the tradable version died around 2016, and the TOM effect
> itself is **not** flows in disguise — so **Mixed, Fragile, and the disguise theory Busted**.

## What we tested

The pension-desk classic: *"month-to-date equities trounced bonds, so rebalancers must sell
$X bn of stocks into month-end — fade it, then ride the bounce."* (Etula-Rinne-Suominen-
Vaittinen's *dash for cash*; Parker-Schoar-Sun's target-date-fund flows.) On SPY vs a spliced
Aggregate-bond leg (VBMFX → AGG at 2003-09-30, return-space splice — documented; 401 complete
months, 1993-02 → 2026-06) we condition the **last-3-day** and **first-3-day** equity-minus-bond
spreads on the month-to-date gap (top/bottom quintiles, Welch *t*), with the conditioning gap
always measured strictly before the window it predicts — one execution lag everywhere. A
**permutation placebo** (gap shuffled across months, 20 seeds × 2,000 draws), era splits, window
robustness (k = 2/3/4), and an **expanding-quintile** flow trade (8 one-way tickets per event,
borrow on the short leg, HAC *t*) complete the gauntlet. The dedup guard vs
[89-turn-of-the-month](../89-turn-of-the-month/) — the *unconditional* calendar drift — is the
third axis: we test explicitly whether TOM concentrates in big-gap months (it does not). A
deterministic synthetic control with a planted, tunable flow reversal proves the machinery (null
quiet across 20 seeds; planted edge lights up at Welch *t* = −9.3). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "pension rebalancing" actually is, why the sell-the-winner half shows up on the tape, why the bounce half is a costume — and why the trade stopped paying in 2016 |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | quintile splits + Welch *t*, the seed-averaged permutation placebo, dose-response and era/window robustness, the expanding-threshold trade with HAC *t*, and the planted-reversal synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`month_end_rebalancing_flows/`](month_end_rebalancing_flows/). The signal is the
month-to-date equity-bond gap, always measured before the window it predicts; the myth-check is
whether study 89's TOM drift is this flow in disguise. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
