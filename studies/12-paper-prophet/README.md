# Study 12 — Paper-Prophet 🧥 — does an ARIMA+GARCH stack really "win every single trade" on the SPY, or is it vol-targeting in a trenchcoat?

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the house style, see the
> [methodology](../../METHODOLOGY.md). This page follows the desk's standard seven beats.
> Where [Study 10 — Markov-Mint](../10-markov-mint/) took a viral "win every single trade"
> quant pipeline to a **prediction market**, this one stays on the **equity tape** and tests the
> same author's companion thread: the **time-series** stack — ARIMA for direction, GARCH for
> sizing, walk-forward for honesty. The twist we test: the author himself admits the direction is
> a coin flip and "the GARCH sizing is doing more work." So the whole study is one question — once
> you subtract the volatility-targeting, **is there any forecasting alpha left at all?***

## Verdict — read this first

*Measured on the article's `TimeSeriesTradingSystem` ported **verbatim** and run **cold** (refit
every day, as its code does) walk-forward over **8,091** graded days of daily SPY — the full cached
tape **1993–2026**, split-only, not the article's 5-year slice. The forecast (ARIMA direction) is
separated from the sizing (GARCH 1/σ̂) by a constant-long control that reuses the **identical** σ̂
path. As-of 2026-06-01, sample fingerprint `0c5568d20239`; every number in
[`docs/results.md`](docs/results.md).*

| Axis | Stamp | Why (one line) |
|---|---|---|
| **Signal** — does ARIMA forecast the *direction* of SPY returns better than a coin? | `NONE` | Walk-forward directional hit-rate **51.98%** (HAC *t* = **+0.85**) — inside the noise band, and even that tilt is just the up-drift, not skill. The author's own "52–55%" is a coin flip once you account for the drift. |
| **Tradability** — does the live stack's edge survive once you remove the vol-targeting and charge costs? | `MIRAGE` | The stack's Sharpe is **+0.17**; the **same GARCH sizing on a constant-long signal** scores **+0.53** — so the forecast doesn't just add nothing, it **subtracts 0.35 Sharpe** (boot CI [−0.74, +0.07], 95% negative). Against the vol-managed-SPY factor the stack's alpha is **+0.0006%/day** ≈ 0. It *is* vol-targeting; the forecast is a tax on it. |
| **"Win every single trade"?** | `BUSTED` | Per-day win rate **51.98%** — a coin flip; the headline is mathematically impossible and empirically ~50%. |

> **In one sentence:** the "complete time-series trading stack" is **volatility targeting wearing an
> ARIMA forecast as a trenchcoat** — the GARCH 1/σ̂ sizing earns a real +0.53 Sharpe (documented
> managed-beta, not alpha), and bolting the coin-flip forecast on top drags it down to +0.17, so the
> whole stack is *worse than its own risk layer run naked*.

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

---

## 1 · The Claim

> *"Check for stationarity, model returns not prices, fit ARIMA to capture the directional signal
> in return autocorrelations, layer GARCH on top to scale position size by volatility, validate
> with a rolling walk-forward — and you have a complete systematic stack that generates consistent
> edge on any liquid market."*

The claim comes from a viral X/Twitter article by **Roan (@RohOnChain)**, *How To Build A Time
Series Model To Win Every Single Trade (Quant Framework)* (19 May 2026; ~792 K views — see
[`docs/references.md`](docs/references.md)). Like its Markov-chain companion it is unusually
concrete: a runnable `TimeSeriesTradingSystem` class on SPY that

1. pulls 5y of daily SPY, converts to **log-returns** (correctly — prices have a unit root, returns
   don't; the ADF discipline in the article is sound and we keep it);
2. fits **ARIMA(1,0,1)** on a rolling 252-day window and forecasts one step ahead for the **sign**;
3. fits **GARCH(1,1)** on the ARIMA residuals and forecasts one-step volatility;
4. sets `position_size = min(1, 1/σ̂)` and trades `sign(forecast) × position_size`, walk-forward.

We take the steelman seriously: the stationarity discipline is correct, the walk-forward structure
is correct, and **the author is honest** — he writes, in his own Part 5, that *"the GARCH-based
position sizing is doing more work than the ARIMA forecast direction"* and that directional accuracy
is *"52 to 55 percent."* That candour is exactly the hypothesis we operationalise. The thread sells
the **stack**; we test whether the **forecast** inside it does anything a constant `sign = +1`
(long) with the same vol-scaling wouldn't.

> 🔬 **For the quants** — H₁ (the part worth testing): the ARIMA sign carries realized directional
> edge on SPY daily returns, `E[sign(f̂_{t+1})·r_{t+1}] > 0` with HAC *t* > 2, **and** the
> walk-forward stack's Sharpe is materially higher than the *same GARCH sizing applied to a
> constant-long signal* (pure vol-targeting). The sharp null is the random-walk-in-returns / weak
> EMH: daily index returns have negligible, non-tradable serial dependence (Fama 1970), so
> `sign(f̂)` is ≈ a coin and the entire Sharpe is the vol-targeting term (Moreira–Muir 2017).

## 2 · So What?

If a one-line `ARIMA(1,0,1)` on five years of daily closes could call tomorrow's *direction* on the
single most-traded, most-arbitraged index on Earth, it would be both a money printer and a
falsification of forty years of market-efficiency evidence — the SPY's daily return would have to
carry exploitable memory that thousands of funds have somehow left lying in `statsmodels`. The far
likelier story, and the one with real stakes for a retail reader, is subtler and more useful: the
stack probably *does* post a respectable backtest Sharpe — but for a reason that has **nothing to do
with forecasting**. Scaling exposure down when volatility is high (and up when it's low) is
**volatility targeting**, a documented and genuinely positive strategy — but it is a *risk-premium
harvest*, essentially a smarter way to hold beta, not an alpha the model "found." The reader who
can't tell those apart will believe ARIMA is predicting the market when the GARCH layer is quietly
doing the only thing that works — and will then mis-apply "the forecast" to instruments where the
vol-targeting tailwind isn't there. Separating the two is the whole product of this desk.

> 🔬 **For the quants** — vol-targeting raises Sharpe on equities because realized vol is
> persistent *and* negatively related to forward returns (the leverage effect / Moreira–Muir
> "Volatility-Managed Portfolios", 2017). That lifts the Sharpe of *any* long-biased signal,
> including a constant one. So a high stack-Sharpe is consistent with **zero** forecasting skill.
> The decisive quantity is the *increment* the ARIMA sign adds over constant-long-with-same-sizing,
> net of the turnover the forecast induces.

## 3 · How We'd Know

The claim is a *method* claim on public data, so the honest test is to run the author's own code,
unaltered, on real SPY — then **decompose** the result into "the forecast" and "the sizing", and
pre-register what kills the forecasting story:

- **Signal `NONE`** if the walk-forward ARIMA directional hit-rate is statistically
  indistinguishable from 50% (HAC *t* < 2 on the realized directional edge) — i.e. the sign is a
  coin flip, as the author concedes.
- **Tradability `MIRAGE`** if the stack's Sharpe is **not materially above** the same GARCH sizing
  driven by a **constant-long** signal (the pure vol-targeting control), and/or the ARIMA increment
  goes ≤ 0 once a realistic round-trip cost is charged on the turnover the forecast creates.
- **"Win every trade" `BUSTED`** if the realized per-trade win rate is a coin flip (any
  sub-100%-accuracy model makes the headline impossible by construction).

Traps we defang up front: **look-ahead** — strict walk-forward, the window at *t* sees only data
< *t*, forecast graded on *t+1*; **the in-sample mirage the author himself warns against** — we
never fit-then-grade on the same data ("a curve fit with a label"); **a silent benchmark swap** —
the vol-targeting control reuses the *exact* GARCH `σ̂` path, only replacing `sign(f̂)` with `+1`,
so the comparison isolates the forecast and nothing else; and **costs charged on the alpha** —
turnover from sign-flips × spread × ~252, per house rule, not a gross figure.

> 🔬 **For the quants** — protocol (shared desk rubric):
> 1. **Measure** the realized directional edge `sign(f̂_{t+1})·r_{t+1}`, pooled walk-forward, and
>    HAC-*t* it (Newey–West). Hit-rate vs 50%.
> 2. **Robust inference / is-it-noise** — bootstrap CI on the stack Sharpe; ADF on price vs returns
>    reproduced (sanity, and to keep the article's one correct lesson).
> 3. **Critique the mechanism** — the decomposition: Sharpe(stack) vs Sharpe(constant-long × same
>    σ̂) vs Sharpe(ARIMA-sign, flat sizing). Attribute the Sharpe to *direction* vs *sizing*.
> 4. **Alpha vs beta** — regress strategy returns on SPY (and on a vol-targeted-SPY factor); how
>    much "edge" is just managed beta (Moreira–Muir)?
> 5. **Execution & capacity** — cost sweep on the forecast's turnover; break-even spread for the
>    ARIMA increment; net-of-cost Sharpe.
> 6. **Verdict** — the three stamps.
>
> Engine: `quantlab.analytics.mean_tstat_hac`, `quantlab.stats.annualized_sharpe`,
> `quantlab.repro`; the increment's CI is a joint day-resampling bootstrap of the *difference of
> Sharpes*. Study code in [`paper_prophet/`](paper_prophet/).

## 4 · The Teardown

> *We run it. Here's what the data actually says.*

- **ARIMA's direction is a coin flip.** Across **8,091** walk-forward days the one-step sign is
  right **51.98%** of the time; the realized directional edge is **+1.04 bp/day** but with HAC
  *t* = **+0.85** — statistically zero, and the small positive tilt is the market's up-drift (more
  up days than down), not forecasting skill. The author's conceded "52–55%" is, precisely, a coin.
- **The Sharpe is the sizing — and the forecast actively *subtracts*.** Full stack Sharpe **+0.17**.
  Replace the forecast with a constant **+1** while keeping the **identical** GARCH σ̂ path — pure
  vol-targeting — and the Sharpe *rises* to **+0.53**. So the constant-long control reproduces
  **302%** of the stack, and the ARIMA increment is **−0.35** (joint-bootstrap CI **[−0.74, +0.07]**,
  **95%** of resamples negative). Even **plain buy-and-hold (+0.45)** beats the celebrated stack:
  the forecast's job is to forfeit the drift roughly half the time.
- **What the model "found" is managed beta.** Regress the stack's daily returns on plain SPY:
  β = **0.16**, α = **+0.0040%/day**, R² = **0.05**. Regress instead on the **vol-managed-SPY**
  factor (constant-long × the same σ̂): the alpha collapses to **+0.0006%/day** (≈ 0.15 bp,
  R² = **0.10**). The stack *is* the vol-targeting factor — Moreira–Muir managed beta, not alpha.
- **Costs only deepen the hole.** The forecast flips sign relentlessly — **141× annual turnover**
  versus the control's near-zero — so there is no positive break-even to find: the increment starts
  negative (−0.35 at zero cost) and falls to **−0.55 at 2 bps** and **−0.85 at 5 bps**. The stack's
  *own* net Sharpe crosses **zero at ≈1.4 bps** round-trip; the vol-targeting control sits unbothered
  at **+0.52** throughout (it barely trades).
- **Win rate is ~50%.** Per-day win rate **51.98%** — the "win every single trade" headline is
  empirically a coin flip.

> 🔬 **For the quants** — directional edge `sign(f̂)·r` pooled over 8,091 days, HAC *t* = +0.85
> (Newey–West). Sharpe decomposition (annualised): stack +0.174 · vol-targeting control +0.527 ·
> forecast-flat +0.138 · buy-and-hold +0.454. Increment = Sharpe(stack) − Sharpe(control) = −0.352,
> joint day-resampling bootstrap CI [−0.74, +0.07]. Alpha vs vol-managed factor +0.0006%/day,
> R² 0.096. Turnover 141×/yr; net increment −0.40 / −0.45 / −0.55 / −0.85 at 0.5 / 1 / 2 / 5 bps.
> Reproduce via [`examples/verify.py`](examples/verify.py) → [`docs/results.md`](docs/results.md).

<details>
<summary>🔬 Why vol-targeting beats the forecast — and why even cheating in-sample doesn't help</summary>

Vol-targeting raises the Sharpe of a *long* equity position because realized volatility is
persistent and inversely related to forward risk-adjusted returns (Moreira–Muir 2017): scaling by
`1/σ̂` de-levers exactly when forward Sharpe is worst. That lift (+0.45 → +0.53 here) is the *only*
working part of the stack, and it needs no forecast — the constant-long control captures all of it.
The ARIMA sign then *removes* value: by going short ~48% of days it throws away the equity drift
the vol-targeting was harvesting, which is why the stack (+0.17) lands below both the control (+0.53)
and plain buy-and-hold (+0.45).

There is a second, quieter finding. The author rightly warns that fitting on the whole series and
grading in-sample is "a curve fit with a label." We measured it: the in-sample directional hit-rate
is **51.91%** versus **51.98%** walk-forward — an inflation of **−0.07 pp**, i.e. none. A low-order
ARIMA(1,0,1) on near-white-noise daily returns has *nothing to overfit*; the absence of an
in-sample mirage is itself confirmation that there is no autocorrelation signal there to begin with.

</details>

## 5 · The Verdict

> *The stamps, and the numbers that earned them.*

- **Signal · `NONE`.** The ARIMA one-step direction is right 51.98% of the time (HAC *t* = +0.85 on
  the directional edge) — indistinguishable from a coin, and the sliver above 50% is the up-drift,
  not skill. There is no tradable directional signal in SPY daily returns at this order.
- **Tradability · `MIRAGE`.** The stack's +0.17 Sharpe is *less* than the same GARCH sizing on a
  constant-long signal (+0.53) and less than plain buy-and-hold (+0.45): the ARIMA increment is
  **−0.35** (95% of bootstraps negative), and against the vol-managed-SPY factor the alpha is
  +0.0006%/day ≈ 0. The "edge" is vol-targeted beta; the forecast is a tax on it that only grows
  with costs (141×/yr turnover).
- **"Win every single trade" · `BUSTED`.** 51.98% per-day win rate. A probabilistic strategy can't
  win every trade, and this one doesn't even tilt the coin.

> 🔬 **For the quants** — decisive numbers in one place: hit-rate 51.98% (HAC *t* +0.85); Sharpe
> stack +0.174 vs vol-targeting +0.527 vs buy-hold +0.454; increment −0.352 (CI [−0.74, +0.07],
> 95% neg); alpha vs managed beta +0.0006%/day (R² 0.096); turnover 141×/yr; net increment −0.55 @
> 2 bps. Fingerprint `0c5568d20239`.

## 6 · Could You Trade It?

> *The honest money question: if you wanted to actually get paid, what would it take?*

Two layers, and the thread sells the wrong one. *First*, the **forecast** is not merely
unprofitable — it is **negative-value**: trading `sign(f̂)` forfeits the equity drift on the ~48% of
days it calls short, and churns **141× a year** doing it, so any spread makes a bad number worse.
There is no break-even cost to quote because the increment is already below zero before the first
cent. *Second*, the part that genuinely posts a Sharpe — the **GARCH vol-targeting** — is real but
is **not an alpha anyone discovered**: it is managed beta (Moreira–Muir 2017), a one-line `1/σ̂`
overlay on a plain long-SPY position that needs no ARIMA, no GARCH forecast of returns, and barely
trades (so it survives costs). The candid bottom line: if you stripped this stack down to the only
thing that works, you'd be left with "hold SPY, size down when vol is high" — a fine idea, freely
available, and the exact opposite of "a time-series model that wins every trade." The forecast on
top isn't an edge; it's a leak.

> 🔬 **For the quants** — the only positive-Sharpe leg (vol-targeting, +0.53) has ~0 turnover and so
> is essentially cost-free; its lift over buy-and-hold (+0.45 → +0.53) is the Moreira–Muir
> managed-beta premium, not alpha (it vanishes against the vol-managed factor by construction). The
> ARIMA overlay's break-even is undefined (negative at zero cost); the stack's own net Sharpe is
> negative beyond ≈1.4 bps round-trip. Capacity is irrelevant — SPY is bottomless — because the
> binding constraint is that the increment is ≤ 0, not that it fails to scale.

## 7 · Going Further

- **Other instruments / the missing tailwind.** Vol-targeting's lift depends on the leverage effect,
  which is strong in equity indices and weaker or inverted elsewhere (commodities, some FX). Re-run
  the exact stack on those and show the "edge" travels with the *sizing tailwind*, not the forecast.
- **Higher-order / different orders.** Does ARIMA(2,0,2), or auto-selected (p,d,q) by AIC, rescue
  the directional signal? Pre-registered expectation: no — more parameters fit more in-sample noise,
  and the walk-forward hit-rate stays at 50%.
- **The author's GARCH-vs-Markov question.** The thread closes by asking when GARCH and a Markov
  regime model *disagree*. With [Study 10](../10-markov-mint/) already in the repo, a follow-up could
  actually run both on the same tape and answer it — turning his rhetorical closer into a measured one.
- **The in-sample mirage that wasn't.** We found the in-sample fit-then-grade hit-rate (51.91%)
  *below* the walk-forward one (51.98%) — no inflation, because a low-order ARIMA has nothing to
  overfit in near-white-noise returns. A richer model (high-order ARIMA, or a neural net like the
  thread's companion piece) *would* manufacture an in-sample mirage; reproducing that gap on the
  same tape would make the teaching point concrete.
- **What to PR:** the other-instruments sweep, the AIC-/high-order control, the
  GARCH-vs-Markov-disagreement experiment, or a richer-model in-sample-inflation figure.

---

## How this study is laid out

| Path | What's inside |
|---|---|
| [`paper_prophet/data.py`](paper_prophet/) | SPY daily loader (cached), log-return transform, the ADF sanity check the article gets right |
| [`paper_prophet/stack.py`](paper_prophet/) | the article's `TimeSeriesTradingSystem` ported verbatim (ARIMA(1,0,1) + GARCH(1,1), rolling 252-day walk-forward), with a switch to replace `sign(f̂)` by constant-long |
| [`paper_prophet/decompose.py`](paper_prophet/) | the Sharpe decomposition: stack vs vol-targeting control vs flat-sized forecast; the alpha-vs-(vol-managed)-beta regression; the cost sweep |
| [`examples/verify.py`](examples/) | the headline run → [`docs/results.md`](docs/) (as-of + fingerprint) |
| [`notebooks/`](notebooks/) | `01_for_the_curious` (the story) and `02_for_the_quants` (the teardown), same seven beats |
| [`docs/references.md`](docs/references.md) | the thread + the efficiency / vol-targeting / ARIMA-GARCH literature it walks into |

The engine that produced every number lives at [`../../quantlab/`](../../quantlab/).
