# Study 08 — True-Strength ⚖️ — is the TSI a *truer* strength gauge than MACD or RSI, or the same momentum trade repainted?

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the house style, see the
> [methodology](../../METHODOLOGY.md). This page follows the desk's standard seven beats.
> Where Study 07 mechanised one retail chart rule and asked "does it pay?", this one steps
> back and asks a question the indicator's **name** makes for us: the "**True** Strength
> Index" claims to be a cleaner, truer read on momentum than the MACD or the RSI. Is it — or
> are the three the same signal wearing different clothes?*

## Verdict — read this first

*Measured on a **reproducible**, costed run over the cached, liquid **174-name** US universe,
1962–2026 (split-only prices; daily returns winsorized at ±100% to kill bad prints). The
three oscillators are computed on their textbook settings — **TSI 25/13/13**, **MACD
12/26/9**, **RSI 14** — each read as a zero-centred momentum *level* and z-scored within its
own name, so the comparison is like-with-like with **no fitted parameter**. As-of 2026-06-01,
price fingerprint `42590aa02dc9`; every number in [`docs/results.md`](docs/results.md).*

| Axis | Stamp | Why (one line) |
|---|---|---|
| **Signal** — is "True Strength" a *distinct* momentum read? | `NONE` | The z-scored TSI is **84% spanned** by the MACD line and the RSI (pooled R² **0.835**); its zero-cross position agrees with the MACD's **99.4%** of days and its long/short **equity curve correlates 0.994** with the MACD's. It is not a different signal — it is the same one, double-smoothed. |
| **Tradability** — a distinct edge once you cost it? | `MIRAGE` | The TSI crossover nets a **0.61** Sharpe at 10 bps — but that is the **long-side equity beta** of a filter that's in the market ~50% of the time: strip the structural long bias (run it long/**short**) and the TSI's *timing* Sharpe collapses to **0.05** (MACD 0.05, RSI **−0.29**). You're paid for holding stocks, not for the oscillator, and the thin remainder decays from Sharpe 0.77→0.15 across a 0→40 bps cost sweep. |
| **"Truer" than MACD/RSI, as the name claims?** | `BUSTED` | Three indicators, one trade. The "true strength" branding promises a distinction the data denies. |

> **In one sentence:** the True Strength Index is a **real but utterly generic** momentum
> oscillator — 84% reconstructable from the MACD and RSI, with a position that agrees with the
> MACD's 99.4% of the time and an equity curve indistinguishable from it (ρ = 0.994) — whose
> standalone "edge" is the long-equity beta you'd get from *any* trend filter, not a truer
> reading of strength.

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

---

## 1 · The Claim

> *"The True Strength Index shows both the direction **and the strength** of momentum while
> smoothing out the noise raw momentum indicators suffer from — a truer strength reading you
> can trade off zero-line and signal-line crossovers."*

The claim is [QuantifiedStrategies.com's TSI write-up](docs/references.md) (their Episode 100),
steelmanned: the TSI, developed by William Blau, is a **double-smoothed** momentum oscillator —
the one-bar price change run through two chained EMAs (the classic 25 then 13), normalised by
the same double-smoothing of the *absolute* change, scaled to ±100. The double smoothing is the
entire selling point: it is what is supposed to make the TSI a *cleaner, truer* strength gauge
than a single-smoothed MACD or a bounded RSI. Their published backtest (gold, GLD) reports a
**1.7 profit factor**, **7.8% CAGR**, **40% win rate** at **50% exposure** — and the actual
trading rules sit behind their Skool paywall.

That last detail matters: we cannot test *their* exact rule, so we don't pretend to. Instead we
test the claim the *name itself* makes — that this is a **distinct, truer** momentum signal —
which is both more fundamental and fully falsifiable.

> 🔬 **For the quants** — H₁ (the claim's strong form): the TSI carries momentum information
> *not* already in the MACD and RSI. Null H₀: the z-scored TSI is spanned by them (high R²),
> its position and equity curve are collinear with theirs, and any standalone Sharpe is the
> long-bias equity beta common to all three. We pre-register that **`NONE`** means R² ≳ 0.8,
> sign agreement ≳ 0.9 vs the MACD, and equity-curve ρ ≳ 0.95.

## 2 · So What?

If the TSI were genuinely a *truer* strength gauge, it would earn its place in a toolkit: a
second, less-correlated momentum vote that improves an ensemble, sharpens an entry, or filters a
trend a beat earlier than the MACD. Traders carry three, four, five oscillators precisely on the
belief that each says something a little different. If instead the TSI is a near-perfect repaint
of indicators that have been on every chart since the 1970s, then stacking it changes nothing —
you are averaging a number with itself and feeling more confident for it. That false
diversification is the costly part: it is *exactly* the illusion that makes a discretionary
trader over-size a "confluence" setup that is really one signal counted thrice.

> 🔬 **For the quants** — collinear signals don't diversify: an equal-weight blend of three
> signals with pairwise ρ ≈ 0.85–0.99 has almost the variance of one. The "stack for
> confirmation" workflow buys no risk reduction and no incremental Sharpe; the stakes are a
> portfolio that *thinks* it has three bets and has one.

## 3 · How We'd Know

We compute all three oscillators with **matched conventions** (each reduced to its zero-centred
momentum *level*, z-scored per name so the ±100 vs [0,100] vs bp scales can't fake a difference),
then ask four escalating questions and **pre-register the mirage line** before running:

- **Same shape?** Per-name Pearson correlation of the z-scored levels, and — sharper — the R²
  of regressing the TSI on the *other two*. `NONE` if the TSI is ≳ 80% spanned.
- **Same position?** Fraction of days the three long/flat crossover positions agree.
- **Same equity curve?** Correlation of the three strategies' net daily returns.
- **Any distinct edge?** Strip the structural long bias by running each long/**short**: if the
  TSI's timing Sharpe collapses toward zero, the standalone number was beta, not the oscillator.

The traps we watch for: **scale artefacts** (handled by z-scoring), **a flattering long bias**
(every long/flat trend filter on a market that drifts up looks good — so we isolate timing with
the long/short version), and **data-snooping on the parameters** (we price a 24-variant TSI grid
with White's Reality Check, so a tuned "best TSI" can't smuggle in an edge).

> 🔬 **For the quants** — protocol (shared desk rubric):
> 1. **Measure** the raw relationship exactly: per-name corr and spanning-R² of z-scored levels.
> 2. **Robust inference** — White (2000) Reality Check on the best of the 24-variant TSI grid
>    (`quantlab.bayes.reality_check`); annualised Sharpe with bootstrap context.
> 3. **Critique the magnitude** — sign-agreement and equity-curve ρ turn "correlated" into
>    "the same trade".
> 4. **Alpha vs beta** — long/short timing Sharpe vs the long/flat number isolates the equity beta.
> 5. **Execution & capacity** — turnover-cost sweep on the crossover.
> 6. **Verdict** — the three stamps.
>
> Engine used: `quantlab.bayes.reality_check`, `quantlab.stats.annualized_sharpe`,
> `quantlab.repro` (as-of + fingerprint). Study code in [`true_strength/`](true_strength/).

## 4 · The Teardown

> *We run it. Here's what the data actually says.*

- **The three oscillators are the same shape.** Median per-name correlation of the z-scored
  levels: **TSI~MACD 0.85**, **TSI~RSI 0.87**, MACD~RSI 0.75. And the spanning test is blunt:
  regress the TSI on the MACD line and the RSI and the **pooled R² is 0.835** (median per-name
  **0.85**). The double smoothing's "true strength" is, to within 16% residual, a linear blend
  of the two oldest oscillators on the chart.
- **The TSI and MACD take the *same position* 99.4% of the time.** On the zero-line rule the
  three agree on **84%** of all days; the TSI and MACD specifically agree on **99.4%**. They are
  not confirming each other — they are the same switch.
- **Their equity curves are indistinguishable.** The TSI and MACD long/short strategies have a
  net-return correlation of **0.994**. You could swap one for the other and not see it in a P&L.
- **The standalone "edge" is equity beta, not the oscillator.** The TSI crossover (long/flat,
  net 10 bps) shows a **0.61** Sharpe, CAGR **6.3%**, **50% exposure** — superficially like
  QuantifiedStrategies' gold result. But run the *same* signal long/**short** to remove the
  always-long bias and the TSI's timing Sharpe falls to **0.05** (MACD **0.05**, RSI **−0.29**).
  The 0.61 was the premium for being long stocks half the time, which any trend filter collects.
- **What's left decays with cost.** Across a 0→40 bps round-trip sweep the crossover's Sharpe
  slides **0.77 → 0.69 → 0.61 → 0.46 → 0.15** (annual turnover ≈ 17×). The Reality Check on the
  24-variant grid returns the best gross Sharpe at **0.77** with **p ≈ 0** — i.e. there *is* a
  faint, real, generic momentum signal here, not pure snooping. That's the honest nuance: the
  TSI isn't fake. It's **redundant** — a real signal you already own twice.
- **The closing argument: the TSI's *unique* part is worse than nothing.** Regress the TSI out of
  MACD+RSI per name (full-sample — the *generous* steelman, hindsight betas) and trade the
  **residual** long/short. It earns a Sharpe of **−0.56** (mean **−2.8 bps/day**): the slice of
  "true strength" the other two can't reproduce isn't a faint edge, it's *anti-signal*. Meanwhile
  the raw long/short TSI is already **+0.05** — so the oscillator's entire honest content is a
  generic momentum whisper MACD captures identically, and its *distinctive* content trades
  negative. There is nothing left to be the "true" in True Strength.

> 🔬 **For the quants** — the spanning R² (0.835 pooled) is the load-bearing number: it says the
> TSI is not merely correlated with but *reconstructable from* MACD+RSI. The long/short timing
> collapse (0.61→0.05) is the alpha-vs-beta split done by construction — symmetrising the
> position nets out the unconditional equity drift, leaving only the oscillator's timing, which
> is ≈ 0. The Reality Check p ≈ 0 is reported against, not for, a snooping story: it confirms a
> weak generic momentum effect, which is exactly why the verdict is redundancy rather than noise.
> Reproduce via [`examples/verify_real.py`](examples/verify_real.py) → [`docs/results.md`](docs/results.md).

<details>
<summary>🔬 Why the long/short test is the clean alpha-vs-beta cut</summary>

A long/flat trend filter earns `E[r | in market] · P(in market)`. On equities, `E[r]` is
positive unconditionally (the equity risk premium), so *any* filter that is long ~half the time
banks roughly half the market's drift regardless of whether its timing is informative. The
long/short version holds `+1` or `−1`, so the unconditional drift cancels in expectation and what
remains is purely the **conditional** signal — the part that actually depends on the oscillator
being right about *direction*. That the long/short Sharpe is 0.05 (TSI) and −0.29 (RSI) while the
long/flat TSI is 0.61 is the whole alpha-vs-beta story in two numbers: the standalone result is
the equity risk premium with an oscillator-shaped on/off switch, not strength-timing alpha.

</details>

## 5 · The Verdict

- **Signal · `NONE`.** The TSI is not a distinct momentum read. It is 84% spanned by the MACD
  line and RSI (pooled R² 0.835), shares the MACD's exact position 99.4% of days, and has an
  equity curve that correlates 0.994 with it. The "true strength" double-smoothing is a
  cosmetic transform of information already on the chart.
- **Tradability · `MIRAGE`.** The crossover's 0.61 net Sharpe is long-equity beta: symmetrise the
  position and the oscillator's own timing is Sharpe 0.05, indistinguishable from the MACD's and
  better than the RSI's only by being less bad. There is no TSI-specific edge to capture, and the
  generic momentum remainder erodes steeply with turnover.
- **"Truer" than MACD/RSI? · `BUSTED`.** Three names, one trade.

> 🔬 **For the quants** — decisive numbers in one place: spanning R² 0.835 (median 0.85); TSI~MACD
> sign agreement 0.994; equity-curve ρ(TSI,MACD) 0.994; long/short timing Sharpe TSI 0.05 / MACD
> 0.05 / RSI −0.29 vs long/flat TSI 0.61; **orthogonalised residual Sharpe −0.56** (the unique part
> trades *negative*); cost-sweep Sharpe 0.77→0.15 over 0–40 bps; grid Reality Check best Sharpe
> 0.77, p ≈ 0. Fingerprint `42590aa02dc9`.

## 6 · Could You Trade It?

You can absolutely trade a TSI crossover — and you'll get a perfectly ordinary long-biased
momentum book: ~50% time in the market, a 0.61 net Sharpe at large-cap costs, a 35% max drawdown,
a 37% win rate carried by a right tail. The catch is that *none of it is the TSI*. The identical
book falls out of a MACD crossover (ρ = 0.994), which most platforms compute for free and which a
reader can reason about without a double-EMA-of-an-EMA in their head. So the honest execution
answer is a question about **why** you'd reach for the TSI: if it's for a distinct signal, there
isn't one; if it's to *confirm* the MACD, you're confirming a number with itself and paying
turnover for the privilege; if it's for the standalone return, that's the equity risk premium you
could harvest more cheaply and with more capacity by just being long a momentum factor. The one
place the TSI is genuinely *worse* is capacity-of-attention: it adds a knob (r, s, signal) to tune
— and the grid Reality Check shows the "best" tuning is a 0.77 Sharpe that is still the same beta.

> 🔬 **For the quants** — break-even isn't the binding constraint here (the trade survives 40 bps
> at a thin Sharpe); **redundancy** is. Adding the TSI to a book already running MACD/RSI raises
> gross exposure and turnover-cost while adding ~16% independent signal variance (1 − R²) that is
> itself *anti-signal*: trade that orthogonalised residual and you earn a **−0.56** Sharpe. The
> marginal contribution of the TSI to a MACD/RSI book isn't just zero — it's negative once costed.

## 7 · Going Further

- **The single-asset GLD claim, head-on.** QuantifiedStrategies tested gold specifically. Our
  universe PF is 1.27 vs their 1.7 — well inside the cross-sectional spread of *single-name* PFs,
  i.e. GLD looks like one favourable draw. A per-name PF histogram with GLD marked would price that
  selection directly (our cache would need GLD added).
- **Other "new" oscillators.** The same machine tests whether the Stochastic, CCI, Williams %R,
  Awesome Oscillator, etc. are anything but momentum repainted. We'd wager the spanning R² is high
  for all of them — a small atlas of oscillator redundancy.
- **Signal-line vs zero-line.** We lead on the zero-cross (where TSI~MACD agreement is 99.4%); the
  signal-cross agreement is lower (90.7%) because it depends on the *second* smoothing span. Worth
  a short note on whether any of the apparent independence there is real or just lag mismatch.
- **What to PR:** GLD-in-cache for the head-on comparison, the oscillator-redundancy atlas
  (Stochastic / CCI / Williams %R / Awesome), or a total-return rerun.

---

## How this study is laid out

| Path | What's inside |
|---|---|
| [`true_strength/oscillators.py`](true_strength/oscillators.py) | the three oscillators on matched conventions, the z-score normaliser, the TSI parameter grid |
| [`true_strength/backtest.py`](true_strength/backtest.py) | oscillator → lagged position → costed daily P&L; the broker statement; the long/short alpha-vs-beta cut |
| [`true_strength/collinearity.py`](true_strength/collinearity.py) | the gauntlet — level collinearity, the spanning-R², sign agreement, equity-curve ρ, the grid Reality Check, the cost sweep |
| [`true_strength/data.py`](true_strength/data.py) | the cached real universe + a synthetic one with **planted trend/cycle structure** to validate the machinery offline |
| [`examples/verify_real.py`](examples/verify_real.py) | the headline run → [`docs/results.md`](docs/results.md) (as-of + fingerprint) |
| [`notebooks/`](notebooks/) | `01_for_the_curious` (the story) and `02_for_the_quants` (the teardown), same seven beats |
| [`docs/references.md`](docs/references.md) | the claim + the indicator-redundancy literature |

The engine that produced every number lives at [`../../quantlab/`](../../quantlab/).
