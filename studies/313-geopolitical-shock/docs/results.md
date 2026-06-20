# Results — Study 313 (Geopolitical-Shock) on the real SPY tape

*Event study of SPY around a curated table of 28 major geopolitical shocks (1990–2025).
The Caldara-Iacoviello GPR index would be the data-driven way to pick events, but it is
network-blocked here, so the shocks are a hand-built table of the dates a believer in
"geopolitics moves markets" would point at (see [`data.py`](../geopolitical_shock/data.py)).
Abnormal return = excess over the full-sample mean (constant-mean market model); CAR =
cumulative abnormal return; the synthetic control is a **placebo distribution** of 3,000
sets of random non-event dates. As-of **2026-06-17**; match the fingerprint to confirm the
tape.*

## Data stamp

| Ticker | Window | Sessions | Fingerprint |
|---|---|--:|---|
| SPY | 1993-01-29 → 2026-06-17 | 8,403 | `b891c5df8fc1` |

Of the 28 curated shocks, **26** fall inside the SPY window with a full ±(5, +10) session
window; two pre-date the SPY tape's 1993 start (1990 Kuwait invasion, 1991 Gulf War) and
are dropped from the CAR. The buy-the-dip ledger keeps all 28 with a hold that fits.

## The headline — does the market drift after a shock? (constant-mean model)

Mean cumulative abnormal return over the post-event window, with the cross-event *t* and
the percentile of that CAR inside the random-date placebo distribution:

| Post window | mean CAR | cross-event *t* | placebo percentile |
|---|--:|--:|--:|
| +1 session | +0.46% | +1.88 | 98th |
| +3 sessions | +0.20% | +0.57 | 72nd |
| +5 sessions | +0.27% | +0.62 | 74th |
| **+10 sessions** | **+0.04%** | **+0.07** | **51st** |
| +21 sessions | +0.48% | +0.75 | 68th |

- The **event-day abnormal return is −0.20%** — a small, real dip on the day the news
  becomes tradable. That is the entire visible footprint.
- Beyond one session there is **nothing**: the +10-day CAR is **+0.04%** with *t* = +0.07,
  sitting at the **51st percentile** of the placebo distribution — the median of pure
  chance. The market has fully shrugged the shock off.
- The only window with a hint of life is **+1 session** (a +0.46% relief bounce, 98th
  placebo percentile) — but at *t* = +1.88 it falls **short of the inference bar (|t| ≥ 2)**
  and is the most multiple-testing-prone window we looked at. It does not earn a `REAL`.

## The synthetic control — placebo distribution of random dates

The +10-session placebo distribution (3,000 draws of 26 random non-event dates) has mean
**+0.01%** and standard deviation **0.62%**. The real shock CAR (+0.04%) is one-fifteenth
of a placebo standard deviation from the placebo mean. A 95% **block bootstrap** CI on the
mean post-event CAR (resampling events) is **[−1.13%, +1.05%]** — straddling zero with
room to spare.

> The synthetic control is a *machinery / falsification* check: it shows what a CAR of this
> size looks like by chance. It can refute a signal (this one) but can never, on its own,
> certify one — see METHODOLOGY → the inference bar.

## The tradable overlay — "buy the geopolitical dip" (SPY, long-only)

Enter at the close of the trade date (the shock is known), hold *h* sessions, one execution
lag applied once, one-way costs charged twice (entry + exit) against NAV:

| Hold | n | win-rate | mean (bps) | *t* (gross) |
|---|--:|--:|--:|--:|
| +1 session | 28 | 60.7% | +52.6 | +2.29 |
| +3 sessions | 28 | 67.9% | +45.8 | +1.36 |
| +5 sessions | 28 | 64.3% | +62.8 | +1.53 |
| +10 sessions | 28 | 64.3% | +56.5 | +1.05 |
| +21 sessions | 28 | 82.1% | +145.8 | +2.38 |

These are **raw** (not abnormal) returns, so they include the equity's up-drift. The
+21-session *t* = +2.38 looks tempting — but +145.8 bps over 21 sessions is almost exactly
SPY's ~10.8%/yr drift prorated; it is **beta you were always paid for**, not a shock edge
(the constant-mean CAR over the same window is an insignificant +0.48%, *t* = +0.75). The
+1-session *t* = +2.29 is the relief bounce again, gross of cost.

### Cost sweep (hold = 10, net)

| round-trip cost | n | mean (bps) | *t* |
|---|--:|--:|--:|
| 0 bps (gross) | 28 | +56.5 | +1.05 |
| 2 bps | 28 | +52.5 | +0.97 |
| 5 bps | 28 | +46.5 | +0.86 |
| 10 bps | 28 | +36.5 | +0.68 |

The overlay never clears *t* = 2 net, fires only **~0.8 times a year**, and what return it
shows is the market drift it would have earned anyway. There is no harvestable shock edge.

## Verdict

- **Signal — NONE.** The post-shock abnormal-return path is statistically indistinguishable
  from a placebo of random dates at every horizon past one session (+10-day CAR +0.04%,
  *t* = +0.07, 51st placebo percentile; bootstrap CI [−1.13%, +1.05%]). The lone +1-day
  relief bounce (*t* = +1.88) falls short of the bar.
- **Tradability — MIRAGE.** "Buy the dip" never clears *t* = 2 net, trades < 1×/yr, and its
  only positive numbers are the equity risk premium prorated — not a geopolitical edge.
- **Do markets shrug off shocks within days? — CONFIRMED.** The folk wisdom is *right*: a
  small same-day dip, then full recovery within a session or two. The market is efficient
  to geopolitics in days; that very efficiency is why there is nothing to trade.
