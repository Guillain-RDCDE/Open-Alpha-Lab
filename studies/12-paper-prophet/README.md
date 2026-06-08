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

<!--
  ───────────────────────────────────────────────────────────────────────────
  PRE-REGISTERED SCAFFOLD. Beats 1·2·3·7 are final (design/narrative, no numbers).
  Beats 4·5·6 and the verdict box carry ‹angle-bracket placeholders› to be filled
  from the real run (paper_prophet/ → examples/verify.py → docs/results.md). No
  number appears on this page until it has been produced and fingerprinted. Until
  then the stamps below are PRE-REGISTERED EXPECTATIONS, labelled as such.
  ───────────────────────────────────────────────────────────────────────────
-->

## Verdict — read this first

> ⚠️ **Pre-registered, not yet earned.** The stamps below are the desk's *announced expectation*
> under beat 3 — written **before** the run so we can't move the goalposts. They become real, with
> numbers, once [`examples/verify.py`](examples/verify.py) has produced
> [`docs/results.md`](docs/results.md) and a sample fingerprint. Until then: design, not findings.

| Axis | Stamp (pre-registered) | Why (one line) |
|---|---|---|
| **Signal** — does ARIMA forecast the *direction* of SPY returns better than a coin? | `NONE` *(expected)* | Daily SPY log-returns are ~unforecastable; expected directional hit-rate ‹HR›% ≈ 50% with HAC *t* = ‹t› — the author's own thread concedes "52–55%". |
| **Tradability** — does the live stack's edge survive once you remove the vol-targeting and charge costs? | `MIRAGE` *(expected)* | The headline Sharpe is expected to be **almost entirely** the GARCH position-sizing — i.e. **vol-targeted beta you were always paid for**, not forecast alpha; net of costs the ARIMA contribution is ‹ΔSharpe›. |
| **"Win every single trade"?** | `BUSTED` *(expected)* | A directional model at <55% hit-rate wins far less than 100% of the time; the headline is mathematically impossible. |

> **In one sentence (pre-registered):** the "complete time-series trading stack" is expected to
> reduce to **volatility targeting** — a well-known risk-premium harvester — wearing an ARIMA
> forecast as a trenchcoat; strip the sizing and the forecast itself prints no tradable directional
> edge on the SPY.

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
> Engine: `quantlab.analytics.mean_tstat_hac`, `quantlab.stats.sharpe_ci_bootstrap`,
> `quantlab.backtest`, `quantlab.repro`. Study code in [`paper_prophet/`](paper_prophet/).

## 4 · The Teardown

> *We run it. Here's what the data actually says.* — **‹PENDING RUN — placeholders below.›**

- **ARIMA's direction is a coin flip.** Walk-forward hit-rate ‹HR›% on SPY daily returns;
  realized directional edge ‹e› bp/day, HAC *t* = ‹t›. ‹fig: hit-rate vs 50% with CI band›
- **The Sharpe is the sizing, not the forecast.** Full stack Sharpe ‹S_stack›; the **constant-long
  × same GARCH σ̂** control posts ‹S_voltgt› — i.e. ‹pct›% of the stack's Sharpe survives with the
  forecast *deleted*. The ARIMA increment is ‹ΔSharpe› (bootstrap CI ‹CI›). ‹fig: three equity curves›
- **What the model "found" is managed beta.** Strategy returns regress on SPY with β = ‹β›, α =
  ‹α› bp/day (HAC *t* = ‹t_α›); against a vol-targeted-SPY factor the alpha falls to ‹α2›. ‹fig›
- **Costs eat the increment.** The forecast flips sign ‹n› times/yr → ‹turnover›× turnover; at a
  ‹c›¢ round-trip the ARIMA increment goes from ‹ΔSharpe› to ‹ΔSharpe_net›. Break-even spread ‹be›¢.
- **Win rate is ~50%.** Per-trade win rate ‹WR›% — the headline is empirically a coin flip.

> 🔬 **For the quants** — ‹decisive stats: HAC t on directional edge, Sharpe decomposition table,
> α before/after the vol-managed factor, turnover×spread×252, break-even cost. Reproduce via
> [`examples/verify.py`](examples/verify.py) → [`docs/results.md`](docs/results.md).›

<details>
<summary>🔬 Why vol-targeting flatters any long-biased signal (and the in-sample trap to avoid)</summary>

‹To fill at write-up: the Moreira–Muir mechanism — realized vol is persistent and inversely
related to forward Sharpe, so scaling by 1/σ̂ raises the Sharpe of a constant-long position; the
ARIMA sign only earns its keep if Sharpe(stack) − Sharpe(constant-long × same σ̂) is positive and
survives the turnover it adds. Plus the author's own warning, quantified: fit-then-grade in-sample
ARIMA on the full series vs strict walk-forward, to show how much "edge" is just the curve fit.›

</details>

## 5 · The Verdict

> *The stamps, and the numbers that earned them.* — **‹PENDING RUN.›**

- **Signal · `NONE` (expected).** ‹Directional hit-rate ‹HR›% ≈ 50%, HAC *t* = ‹t›: the ARIMA sign
  carries no tradable directional edge on SPY daily returns — exactly the 52–55% the author conceded,
  which is a coin flip once you account for the long drift.›
- **Tradability · `MIRAGE` (expected).** ‹The stack's Sharpe is ‹pct›% reproduced by constant-long
  with the same GARCH sizing; the ARIMA increment is ‹ΔSharpe_net› net of costs. The edge is
  vol-targeted beta, not forecast alpha — it doesn't decay to a mirage, it was never the forecast.›
- **"Win every single trade" · `BUSTED` (expected).** ‹Win rate ‹WR›% — a coin flip; a sub-100%
  model makes the headline impossible by construction.›

> 🔬 **For the quants** — ‹decisive numbers in one place once run; fingerprint ‹hash›.›

## 6 · Could You Trade It?

> *The honest money question.* — **‹PENDING RUN; the shape of the answer, pre-registered:›**

The pre-registered story has two layers. *First*, the **forecast** as written is expected to be
**not worth trading on its own**: at a <55% directional hit-rate on a low-autocorrelation series,
the sign barely clears 50%, and the turnover it generates is taxed by the spread twice over. *Second*,
the part that *does* post a Sharpe — the **GARCH vol-targeting** — is real but is **not an alpha the
model discovered**: it's a documented way to hold equity beta more efficiently (Moreira–Muir 2017),
available to anyone via a one-line `1/σ̂` overlay on a plain long-SPY position, *without* ARIMA. So
the candid bottom line is expected to be: there's a genuinely useful idea in here (size down when
vol is up) and a decorative one (ARIMA forecasts the SPY); the thread sells the second on the back
of the first. The number that settles it is the **ARIMA increment net of its own turnover cost** —
if it's ≤ 0, the forecast is paying you nothing.

> 🔬 **For the quants** — ‹break-even spread for the ARIMA increment vs realistic SPY round-trip;
> net-of-cost ΔSharpe; capacity is not the binding constraint here (SPY is deep) — the binding
> constraint is that the increment is ≈ 0, so any cost makes it negative.›

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
- **The in-sample vs walk-forward gap.** Quantify exactly how much Sharpe the "fit on the full
  dataset" mistake (which the author rightly warns against) manufactures — a clean teaching figure.
- **What to PR:** the other-instruments sweep, the AIC-selected-order control, the
  GARCH-vs-Markov-disagreement experiment, or the in-sample-inflation figure.

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
