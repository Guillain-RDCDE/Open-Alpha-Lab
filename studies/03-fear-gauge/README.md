# Study 03 — Fear-Gauge 🌡️ — does buying the VIX spike actually pay?

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the house style,
> see the [methodology](../../METHODOLOGY.md). This page follows the desk's
> standard seven beats. Companion study: [02 — Falling-Knife](../02-falling-knife/)
> — this is its twin in **volatility space**.*

## Verdict — read this first

*Numbers below are measured on `^GSPC` + `^VIX`, daily closes, 1990–2026 (9,174
common sessions), reproducible via [`examples/verify_real.py`](examples/verify_real.py)
and the [notebooks](notebooks/).*

| Axis | Stamp | Why (one line) |
|---|---|---|
| **Signal** — is the effect statistically real? | `REAL` for the **level**, `NONE` for the **spike** | VIX≥30 beats a random day by **+1.0% at 1wk** (p≈0.00) and **+1.3% at 1mo** (p≈0.01); but the famous **+30% spike** earns **−0.02% at 1mo** (p≈0.51) — its whole edge is the 2016–2026 window. |
| **Tradability** — does it survive costs, capacity, scale? | `MIRAGE` | The level's excess **does not significantly beat just buying a −3% price day** (gap p≈0.13–0.20), is borderline once clustering is respected (bootstrap p≈0.05, CI touches 0), and traded it sits in cash ~88% of the time and **underperforms buy-and-hold**. |
| **The "double down at 50"?** — is the martingale survivable? | `RUIN-PRONE` | Held a quarter, the worst episode draws down **−33%** (−40% over six months); the 2016–2026 window that sells the rule caps the worst *terminal* loss at −3.6% and hides all of it. |

> **In one sentence:** the fear gauge genuinely carries information — a high VIX
> really is followed by a real rebound — but it's the **variance risk premium**
> (you're paid to hold the tail), it barely beats the price drop we already studied
> in [02](../02-falling-knife/), and the "double-down" martingale is a risk-of-ruin
> generator the cherry-picked chart window conveniently hides.

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

---

## 1 · The Claim

There are **two** claims in the screenshot that started this study, and they are
routinely conflated — which is half the reason it's worth a teardown.

**The "VIX rule" (a *level*).** *"Buy stocks when VIX hits 30. Double down when VIX
hits 50."* A cran of panic as a buy signal, with a martingale bolted on top.
([source — @TheProf, quoting](https://twitter.com))

**The Altucher chart (a *spike*).** *"S&P 500 after every VIX 30%+ single-day
spike, June 2016 – June 2026: avg +0.42% next day, +2.66% next month, 21/23
positive."* ([source — @jaltucher](https://twitter.com/jaltucher))

A level and a one-day jump are **not the same trigger**, and the famous chart
already mixes regimes wildly: a +46% spike that lands VIX at **15.59** (May 2017)
sits in the same column as a +43% spike that lands it at **82.69** (Mar 2020).
So we test the whole **family**, not one lucky member:

| ID | Definition | Reads as |
|---|---|---|
| **V1** | VIX **closes** ≥ K (K ∈ {30, 40, 50}) | the Prof's *level* rule |
| **V2** | VIX **1-day change** ≥ +30% | Altucher's *spike* |
| **V3** | a spike, split by **base level** (low-base vs high-base) | "+30% from 13" ≠ "+30% from 60" |
| **V4** | level ≥ 30 **then add** at ≥ 50 | the *martingale* — tested as a sizing rule |

Each feeds the *same* event study, benchmark and backtest as Study 02, so the two
are apples-to-apples.

> 🔬 **For the quants** — H₁: the forward S&P return conditional on a fire exceeds
> the unconditional (random-day) mean at the same horizon, with HAC *t* > 2 after
> clustering-aware inference — at horizons +1d / +1w / +1m to match the chart
> exactly. Null H₀: excess ≈ 0 (the move is just drift + beta).

## 2 · So What?

If the gauge pays, it's the simplest fear-buy a retail trader could run, and it
would mean a number anyone can read off a screen front-runs the market's recovery.
If it *doesn't* — or if it pays only as compensation for catastrophe — then the
millions who "buy the VIX" are loading **negatively-skewed risk** and calling it
edge.

The deeper lesson is **the difference between alpha and a risk premium**. Study 02
found the dip-bounce was mostly *nothing*. This one will likely find a bounce
that's genuinely *real* — and that's the trap: a real, repeatable, positive-expectancy
effect can still be uninvestable, because you're being paid to hold the tail.
*Lesson: "statistically real" and "worth trading" are different questions — a true
signal can be pure insurance premium in disguise.*

> 🔬 **For the quants** — back-of-envelope: the chart's "+2.66% / 21-23 positive at
> 1 month" must be read against the S&P's unconditional ~+0.8–1%/month drift and
> the ~70% base rate of any 1-month window being positive. The interesting quantity
> is the **excess**, and its **skew** — selling a thin positive mean against a fat
> left tail.

## 3 · How We'd Know

The market drifts up anyway, so a green curve after the spike proves nothing. As in
Study 02, the question isn't *"did it rise?"* but *"did it rise **more** than on a
normal day?"* — plus a question unique to this study: **does the VIX add anything
over the price drop already studied in 02?** (VIX-high and price-low are nearly the
same event in two coordinates.) That drives our commitments:

- **Excess over a random-day null**, by permutation — never absolute return.
- **Excess over Study 02's price trigger** — the cross-study control: is the vol
  coordinate informative *given* the drawdown?
- **Block bootstrap**, because spikes cluster (4 of Altucher's 23 are Feb–Mar 2020):
  23 raw events ≈ ~8 independent episodes.
- **Full-history rerun.** The chart's 2016–2026 window **excludes 2008**. We rerun
  on `^VIX` since 1990 and watch the excess move.
- **Alpha vs beta** — regress the forward return on the market and a short-vol
  factor: is there residual alpha, or is it all VRP?
- **Risk-of-ruin** for V4 — the martingale is evaluated as position sizing, not a
  point estimate.

We run it on **two faces** of the same market, and always report both:

| Symbol | History | Role |
|---|---|---|
| `^GSPC` + `^VIX` | since 1990 | spot index + spot vol — deep sample (1998, 2008, 2020), great stats, **not tradeable** |
| `SPY` (+ a real VIX instrument) | since 1993 / later | what you could actually trade — real prints, **shorter** sample |

> 🔬 **For the quants** — the shared desk protocol, powered by
> [`quantlab/`](../../quantlab/): (1) decompose the conditional vs unconditional
> mean by exact identity; (2) Newey-West / Lo (2002) SEs, block bootstrap CIs,
> White (2000) Reality Check over the (trigger × horizon) grid; (3) magnitude
> critique — skew, drift, the 2016–2026 selection; (4) alpha-vs-beta on a
> market + short-vol regression; (5) cost/impact sweep and the martingale
> risk-of-ruin sim; (6) verdict. Engine: `decompose`, `analytics`, `stats`,
> `bayes`, `backtest`, `simulate`, `diagnostics`.

## 4 · The Teardown

> *We run it. Here's what the data actually says.* (^GSPC + ^VIX, 1990–2026;
> 54 fresh VIX≥30 events, 35 fresh +30% spikes, 21-day cooldown.)

- **The level rebound is real — at a week and a month.** After a VIX≥30 cross the
  S&P returns **+1.22%** over the next 5 days and **+2.13%** over 21, versus a
  random-day **+0.19%** / **+0.81%** — an excess of **+1.0%** (p≈0.00) and
  **+1.3%** (p≈0.01). The gauge is not noise.
- **The famous +30% *spike* is the weaker trigger.** Altucher's actual signal earns
  a small bump at a week (+0.31% excess, p≈0.21) and **nothing at a month**
  (−0.02% excess, p≈0.51). The headline chart leans on the *level* intuition while
  showing the *spike* data.
- **It barely beats just buying the −3% day.** Pitted against Study 02's price
  trigger on the same forward returns, VIX≥30's gap is **+0.9%/+1.3%** at 1wk/1mo
  but **not significant** (p≈0.13 / 0.20); the spike's gap is ≈0. Most of the "VIX
  edge" is the price drop in a volatility costume.
- **Clustering eats the rest.** A month-block bootstrap puts the level's 21-day
  excess at **+1.3%, 95% CI [−0.2%, +2.7%]**, p(excess≤0)≈**0.05** — it touches
  zero. The spike's CI straddles zero firmly (p≈0.46).
- **The spike's edge *is* the window.** The 21-day excess is **+0.5%** on the viral
  2016–2026 window, **−0.02%** on the full history, **−0.6%** pre-2016. Put the
  crashes back and the effect inverts — textbook selection.

> 🔬 **For the quants** — permutation null (2,000 random baskets) and a circular
> month-block bootstrap (`benchmark.py`, `robustness.py`); horizons fixed at
> +1d/+1w/+1m to match the chart, announced before running. The excess t-stats are
> uncorrected for the family scan — the deflated-Sharpe pass in beat 6 handles
> selection. Reproduce: `02_for_the_quants.ipynb`, beats 3–4.

<details>
<summary>🔬 The maths, in full</summary>

Excess = E[r_{t→t+h} | event] − E[r_{t→t+h}]; significance by permuting the event
mask over the pool of valid forward returns. The cross-study control permutes the
label over the union of the VIX and the −3% event sets, so its null is "the two
triggers draw forward returns from the same distribution". The block bootstrap
resamples contiguous 21-day blocks of (forward-return, is-event) pairs to preserve
the clustering of crises; the window test simply re-evaluates the same excess on
sub-samples.

</details>

## 5 · The Verdict

> *The two stamps, and the numbers that earned them.*

- **Signal — `REAL` for the level, `NONE` for the spike.** VIX≥30 clears the
  random-day null at 1wk (p≈0.00) and 1mo (p≈0.01): vol mean-reverts, and that
  shows up as a genuine equity rebound. But the +30% spike — the chart's real
  trigger — has no monthly excess (p≈0.51) once you leave its decade. Overall the
  signal is real but **fragile**: borderline under clustering (p≈0.05) and not a
  clean win over the price drop.
- **Tradability — `MIRAGE`.** The excess is the variance risk premium, not alpha;
  it doesn't significantly beat buying a −3% day; and as a strategy it's in cash
  ~88% of the time (Sharpe ≈0.8, CAGR ≈2.3% vs the index's ~7–8%) — market-timing
  that **underperforms buy-and-hold**.

> 🔬 **For the quants** — decisive numbers in one place: random-day p≈0.00 (5d) /
> 0.01 (21d) for VIX≥30; cross-study gap p≈0.13–0.20; block-bootstrap p(excess≤0)
> ≈0.05; window excess +0.5% → −0.02% → −0.6%; family-scan top Sharpe ≈14 on 5
> trades → **deflated Sharpe ≈0** (pure data-mining).

## 6 · Could You Trade It?

> *The honest money question.*

You can't buy spot VIX — the trade is SPY (or a vol product carrying its own roll
and decay). Charge realistic costs, including extra entry slippage for buying into
the spike, and the level rule posts **Sharpe ≈0.8, max drawdown −29%**, sitting in
cash seven days in eight: you took on fear and timing to earn *less* than holding
the index. That's the `MIRAGE`.

Then the rule's second half — **"double down at 50"** — is a martingale: you add
capital exactly as the tail fattens. Held a quarter, the worst historical episode
draws down **−33%** (−40% over six months) and ends **−21%** in the red; a tenth of
episodes double down at all, and that's where the damage concentrates. The viral
chart's 2016–2026 window caps the worst *terminal* loss at **−3.6%** — it literally
cannot see the path (2008) that breaks the rule. The martingale "works" right up
until the one regime the selling window omits.

> 🔬 **For the quants** — `backtest.run` with `CostModel(panic_slippage_bps=5)`;
> `robustness.martingale_ruin` holds a fixed 63-day window and tracks the deepest
> mark-to-market drawdown (the honest version of "wait for the bounce" — a buyer who
> closes the instant they're green never experiences the tail). Capacity is a side
> issue: ~1.5 events/year, dominated by a handful of clustered crises.

## 7 · Going Further

- **Is the rebound just the VRP?** Replace the raw S&P return with a VRP-hedged
  return and see whether any excess survives. If not, the case is closed: pure premium.
- **Term structure.** Does VIX *backwardation* (front > back) at the spike predict
  the rebound better than the level/jump alone?
- **Cross-asset.** The same "buy the fear gauge" claim is made on the MOVE index
  (bonds) and crypto's DVOL — does the result generalise or is it equity-specific?
- **What a contributor could PR:** a fourth trigger (realised-vol spike vs implied),
  or a cleaner short-vol factor than the proxy we ship.

---

## How this study is laid out

| Path | What's inside |
|---|---|
| [`notebooks/01_for_the_curious.ipynb`](notebooks/) | the story + the stakes, plain language |
| [`notebooks/02_for_the_quants.ipynb`](notebooks/) | the full teardown: inference, confounds, capacity |
| [`docs/references.md`](docs/) | sources + literature map (VRP, vol mean-reversion) |
| [`fear_gauge/`](fear_gauge/) | the study package: `data` · `triggers` · `eventstudy` · `benchmark` · `exits` · `backtest` · `robustness` |
| [`examples/`](examples/) | [`run_synthetic_demo.py`](examples/run_synthetic_demo.py) (offline) · [`verify_real.py`](examples/verify_real.py) (the headline tables) |

Every number is produced by [`fear_gauge/`](fear_gauge/), in the house style of the
shared [`../../quantlab/`](../../quantlab/) engine; `pytest` covers it in CI.
