# Study 09 — Phantom-Kernel 👻 — does market-making's famous "optimal spread" rest on an order-arrival law real markets actually obey?

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the house style, see the
> [methodology](../../METHODOLOGY.md). This page follows the desk's standard seven beats.
> Where Studies 02–07 hunt an edge in **prices**, this one tears down a **prescriptive model**
> — the equations a generation of market-making bots quote from — and asks whether its single
> load-bearing assumption survives the market it claims to describe.*

## Verdict — read this first

*Measured on a **reproducible, seed-fixed order-flow simulator** (Avellaneda-Stoikov is a
theorem about a model world, so the cleanest falsification builds the world — see the data
note below). Two worlds: **A (textbook)**, where every AS assumption holds, and **B
(frictions)**, with the documented realities the paper omits — **heavy-tailed (power-law)
order reach, price jumps, stochastic volatility, and informed flow**. Seed 0; simulated-input
fingerprint `1cb0c6bc010a`; every number in [`docs/results.md`](docs/results.md).*

| Axis | Stamp | Why (one line) |
|---|---|---|
| **Signal** — is the arrival kernel `λ(δ) = A·e^(−kδ)` real? | `NONE` | Under heavy-tailed order reach (the empirically documented case) the kernel is a **power law**, not an exponential — the fit flips from R² **1.00 → 0.68** while a power law scores **0.9996** (AIC prefers it by **+1.26M**); the `k` you'd estimate is **0.20**, a number with no stable meaning. A `k` that drifts 4× intraday (the article's own admission) mis-prices the "optimal" spread by up to **±163%**. **Confirmed on real Binance order books** (Clauset/Vuong tail test): order size is power-law on **4/4** markets, price-distance on **3/4**. |
| **Tradability** — does *skipping* AS "leave money on the table"? | `FRAGILE` | A **brainless inventory clamp** beats full AS on risk-adjusted P&L whenever inventory isn't dangerous (World A Sharpe **3.27 vs 1.59**). AS's genuine benefit shows only in the **hostile** world (Sharpe **2.12**, best of four) — and the article's recommended "rolling-vol" production fix **collapses** there (Sharpe **0.17**). |
| **The famous "optimal spread"** — is it the source of the edge? | `MISATTRIBUTED` | The value lives in the **inventory skew**, which is **algebraically free of `k`**. The phantom kernel corrupts only the *spread width* — the half of the model the article crowns, and the half that doesn't carry the edge. |

> **In one sentence:** Avellaneda-Stoikov's celebrated optimal-spread formula is built on an
> order-arrival law (exponential decay with a constant `k`) that the documented heavy-tailed
> reality breaks — making `k` a phantom that misprices the spread by up to 160% — yet the
> model still earns its keep in hostile markets, because the part that actually works (the
> inventory skew) never depended on `k`; on risk-adjusted P&L a four-line inventory clamp
> matches or beats the whole apparatus whenever inventory isn't dangerous.

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

> **A note on the data choice (house rule: state it as a decision).** Every other study on
> this desk runs on cached market series. This one runs on a **simulator**, deliberately: AS
> is not a claim about a price history, it is a *theorem about a model world*, so the honest
> test is to build two worlds — one that obeys its assumptions (where the machinery must
> validate) and one wired with the frictions it omits — and run the identical code in both.
> The single empirical question the simulator can't answer for itself — *is real order reach
> heavy-tailed?* — is settled by the microstructure literature (power-law trade size and order
> flow: Gabaix et al. 2003, Bouchaud et al. 2009) **and confirmed here directly** on four real
> Binance USD-M futures order books (rigorous Clauset/Vuong tail test): order flow is power-law,
> not exponential — see [`docs/results_real.md`](docs/results_real.md), reproducible from
> [`examples/confirm_heavy_tail.py`](examples/confirm_heavy_tail.py) (and a generic
> bring-your-own-data harness, [`examples/verify_real.py`](examples/verify_real.py)).

---

## 1 · The Claim

> *"Two equations give you the mathematically optimal bid and ask at every instant. Hummingbot,
> HFT desks, on-chain AMMs all run on them. Not implementing this is leaving serious money on
> the table."* — [the viral write-up that prompted this study](docs/references.md)

Avellaneda & Stoikov (2008) is a genuinely beautiful result, and we steelman it at full
strength. From a mean-variance objective it derives two quotes:

1. **The reservation price** — your true fair value, the mid *skewed by your inventory*:
   `r = s − q·γ·σ²·(T−t)`. Hold a long position and your fair value slides below the mid, so
   you lean on buyers to flatten the book. This is the model's real idea.
2. **The optimal spread** — how wide to quote:
   `δ = γ·σ²·(T−t) + (2/γ)·ln(1 + γ/k)`. The second term is the *order-arrival economics*,
   and it is the whole reason the spread has a **closed form**.

That closed form exists only because of one assumption: market orders arrive at a rate that
fades **exponentially** with how far your quote sits from the mid,
`λ(δ) = A·e^(−kδ)`, with `k` a stable constant. Everything the article sells — "the math is
real, the formulas are verified" — hangs on that exponential.

> 🔬 **For the quants** — two falsifiable hypotheses, pre-registered.
> **H1 (the kernel):** order reach is exponential, so `λ(δ) = A·e^(−kδ)` with a stable `k`.
> We reject if a heavy-tailed (power-law) kernel fits better (AIC) on realistic flow, or if
> `k` is so regime-dependent that a static estimate is meaningless.
> **H2 (the payoff):** the AS skew + closed-form spread beats naive market-makers, so skipping
> it "leaves money on the table." We reject if a trivial inventory clamp matches it on
> risk-adjusted P&L, i.e. the elaborate machinery is *not* the source of any edge.
> The clean microstructure reading that makes H1 testable: a resting quote at distance `δ` is
> filled iff an incoming order *reaches* at least `δ` into the book, so
> `λ(δ) = Λ·P(reach ≥ δ) = Λ·S_reach(δ)` — **the kernel is the survival function of order
> reach.** Exponential reach ⟺ the AS kernel exactly; heavy-tailed reach ⟹ a power law.

## 2 · So What?

If the exponential law holds, the optimal spread is a theorem and the article is right: there
is one correct width, and quoting anything else is money left on the table. If it *doesn't*,
the headline equation is calibrated to a parameter (`k`) that doesn't really exist —
re-estimated every fifteen minutes, swinging by factors of three to five (the article concedes
this), and feeding a "closed-form optimum" that is closed-form around a fiction. The stakes
are not academic: this is the formula inside open-source market-making bots that retail traders
point at live order books. Knowing *which half of the model carries the value* — and which half
is decoration resting on a false assumption — is the difference between deploying it with eyes
open and deploying a confident-looking equation you've never stress-tested.

> 🔬 **For the quants** — the quantitative stakes are about *attribution*. The reservation
> price `r = s − q·γ·σ²·(T−t)` contains **no `k`**; the spread's arrival term
> `(2/γ)·ln(1+γ/k)` is the *only* place `k` enters. So if the edge comes from the skew, a
> wrong `k` is nearly free; if it comes from the spread width, a wrong `k` is ruinous. The
> study is, at bottom, a decomposition of where the P&L actually lives.

## 3 · How We'd Know

We mechanise the two AS equations verbatim, build two simulated worlds, and **pre-register what
would make us call mirage** — the desk doesn't move goalposts:

- **Signal `NONE`** if, on heavy-tailed (power-law) order reach — the empirically documented
  case — a power-law kernel beats the exponential on AIC, and the fitted `k` is unstable
  enough that a static value mis-prices the spread by a large margin.
- **Tradability `MIRAGE/FRAGILE`** if a trivial inventory clamp (no reservation price, no
  maths) matches or beats full AS on risk-adjusted P&L — i.e. any edge is "don't hold
  inventory" beta, not AS alpha.
- **Misattribution** if AS's benefit survives a *deliberately wrong* `k`, proving the value
  isn't in the `k`-dependent spread the article celebrates.

The traps we watch for: **a rigged simulator** (so World A must validate the machinery — the
estimator has to recover the planted `k` and the skew has to pay where the assumptions hold,
or the World-B failure is a bug, not a finding); **an unfair tournament** (so the naive
baselines quote at AS's *own* mid-session spread width — the comparison isolates the skew/clamp
logic, not who quotes tighter); and **a single lucky draw** (so every tournament number is
shown across five seeds).

> 🔬 **For the quants** — protocol (shared desk rubric):
> 1. **Measure** the kernel exactly: bucket fills by quote distance, fit `log λ = log A − kδ`
>    by Poisson-weighted least squares.
> 2. **Robust inference / model selection** — exponential vs power-law by weighted R² *and*
>    Poisson AIC; the static-`k` spread error across a 4× regime drift.
> 3. **Critique the magnitude** — the tournament's P&L Sharpe with bootstrap CIs
>    (`quantlab.stats.sharpe_ci_bootstrap`), inventory variance, adverse-selection counts.
> 4. **Alpha vs beta** — AS measured against a brainless inventory clamp (the "don't hold
>    inventory" beta) and a deliberately misspecified `k`.
> 5. **Execution reality** — jumps, stochastic vol and informed flow switched on in World B;
>    the article's "rolling realised-vol" production fix tested under them.
> 6. **Verdict** — the three stamps.
>
> Engine used: `quantlab.stats.sharpe_ci_bootstrap`, `quantlab.repro` (fingerprint). Study
> code in [`phantom_kernel/`](phantom_kernel/).

## 4 · The Teardown

> *We run it. Here's what the simulator actually says.*

- **The machine works where the assumption is true.** In World A (exponential reach) the
  estimator recovers the planted arrival rate **k = 0.597** (true 0.600, error −0.46%) with
  **R² = 0.99998**, and the exponential kernel wins goodness-of-fit by a vast margin. So the
  code measures the kernel correctly — whatever breaks next is the assumption, not a bug.
- **The kernel is the wrong shape on realistic flow.** In World B (heavy-tailed reach) the
  exponential fit drops to **R² = 0.68** while a **power law scores R² = 0.9996** and AIC
  prefers it by **+1.26M**. The exponential `k` you'd fit is **0.20** — and it's not the
  World-A `k`, or any `k`; it's an artefact of forcing a straight line through a curve. **H1
  falsified.**
- **`k` is a phantom: a static value misprices the "optimal" spread by ±163%.** With four
  intraday regimes whose true `k` spans 4× (0.3 → 1.2 — exactly the article's admitted range),
  each regime's `k` is recovered perfectly, but base AS fits one **static k = 0.42** over the
  session. Translated into the quantity that matters, that single number mis-quotes the AS
  optimal half-spread by **−26%, +37%, +100%, +163%** across the four regimes.
- **Yet AS still wins in the hostile world — on a *wrong* `k`.** In World B, full AS posts the
  best risk-adjusted P&L of four quoters (**Sharpe 2.12**, mean over 5 seeds) while holding
  almost no inventory (std 0.6), *despite* quoting from the phantom `k = 0.20` and eating
  adverse selection on more than half its fills (1,338 of 2,490). The reason is structural:
  **the skew carries no `k`** — the phantom only widened the spread, and in a jumpy world a
  wide spread happened to help.
- **The article's celebrated machinery is beaten by four lines of code — when inventory is
  cheap.** In World A (no jumps, low vol) a **brainless inventory clamp** earns the best
  risk-adjusted P&L (**Sharpe 3.27 vs AS's 1.59**, mean over 5 seeds). AS spends its effort
  minimising an inventory variance (0.65 vs the clamp's 8.2 and the no-control quoter's 45.5)
  that nobody was being paid to fear, and gives up P&L to do it.
- **The recommended "production fix" backfires.** The article's first prescribed adaptation —
  feed a rolling realised-vol into the spread — **collapses** in World B (**Sharpe 0.17**,
  only 103 fills) because naive realised vol is jump-contaminated: one jump spikes the
  estimate, the spread blows out, and the book stops trading. The fix needs the article's
  *other* fix (a circuit breaker) to survive — the "adaptations" are not free.

- **Confirmed on real order books — not just in simulation.** On four Binance USD-M futures
  markets (TRX/XRP/ADA/LTC, 2024-01-15), the rigorous **Clauset-Shalizi-Newman + Vuong** tail
  test calls **order size power-law on all four** (Vuong *V* = 2.8–5.5, *p* < 0.01) and the
  price-distance `|price−mid|` power-law on **three** (the fourth merely *inconclusive*, never a
  clean exponential). Real order flow is heavy-tailed on the exact venue the article targets —
  the exponential AS kernel is rejected on live books. Full table in
  [`docs/results_real.md`](docs/results_real.md).

> 🔬 **For the quants** — on real data we drop the binned-count fit (it is grid-sensitive and
> not a distribution test) for the standard **Clauset/Vuong** tail test: MLE power-law and
> exponential above a KS-chosen `x_min`, then a normalised likelihood ratio (`V > 0` ⟹ power
> law, winner only at `p < 0.05`). Order-size exponents land at α ≈ 3.0–3.4 across markets,
> squarely heavy-tailed; price-distance is noisier (α ≈ 2.4–4.2) because concave (square-root)
> impact and book depth attenuate the tail in price terms — the heavy tail is cleanest at its
> source. The simulator's own kernel fit (below) is Poisson-weighted so dense buckets dominate;
> the World-B verdict holds on weighted R² (0.9996 vs 0.6826) *and* Poisson AIC (gap
> +1.26M, power law preferred). The tournament's exogenous order stream is shared across all
> four quoters (the mid moves on information regardless of who filled), so the P&L differences
> are pure strategy. Sharpe CIs are 2,000-resample bootstraps on block-summed P&L; the
> World-A clamp > AS and World-B AS > clamp orderings are stable across all five seeds (see
> [`docs/results.md`](docs/results.md)). Reproduce via
> [`examples/run_experiments.py`](examples/run_experiments.py).

<details>
<summary>🔬 Why a wrong `k` is nearly free but a wrong shape is not</summary>

The two AS equations partition the dependence on the arrival model cleanly:

```
reservation price   r(s,q,t) = s − q·γ·σ²·(T−t)          ← no k
optimal half-spread  ½δ = ½γ·σ²·(T−t) + (1/γ)·ln(1+γ/k)   ← k only here
```

The skew — the model's actual idea, the thing that drives inventory to zero — is a function
of `q, γ, σ, (T−t)` and nothing else. `k` enters **only** the additive arrival term of the
spread *width*. So a misspecified `k` shifts how wide you quote (a level effect on fill rate
and per-trade capture) but leaves the inventory-management behaviour untouched. That is why,
in World B, AS keeps its tight inventory control and positive Sharpe even with `k` off by a
factor of three: the broken half of the model is the half that wasn't doing the work. The
*shape* failure (exponential vs power law) is more damaging in principle, because it means no
single `k` describes the kernel at all — but its damage is confined to the spread term, which
is exactly the term that turned out not to carry the edge. The article inverts this: it spends
its length on the spread formula and the estimation of `k`, the decorative half.

</details>

## 5 · The Verdict

- **Signal · `NONE`.** The exponential arrival kernel `λ = A·e^(−kδ)` — the assumption the
  whole closed form rests on — is rejected the moment order reach is heavy-tailed, which is
  the documented empirical reality. A power law fits essentially perfectly where the
  exponential fails (R² 0.9996 vs 0.68), and the `k` you'd report is a line forced through a
  curve. With `k` drifting 4× intraday, a static estimate misprices the "optimal" spread by
  up to 163%. **Confirmed on real Binance order books** (Clauset/Vuong): order size is
  power-law on 4/4 markets, price-distance on 3/4 — the heavy tail is real, not a sim artefact.
- **Tradability · `FRAGILE`.** The model is not worthless: in a hostile market its tight
  inventory control genuinely wins (Sharpe 2.12, best of four). But "you're leaving money on
  the table without it" is overstated — a four-line inventory clamp beats it whenever
  inventory isn't dangerous (World A Sharpe 3.27 vs 1.59), and the touted rolling-vol upgrade
  is fragile to jumps. The edge is real but narrow, and not where the article points.
- **The famous "optimal spread" · `MISATTRIBUTED`.** The value is in the `k`-free reservation
  skew; the `k`-dependent spread term — the centrepiece of every explainer — both rests on a
  false law and isn't the source of the P&L.

> 🔬 **For the quants** — decisive numbers in one place: World-B kernel power-law R² 0.9996 vs
> exponential 0.6826 (AIC gap +1.26M); static-`k` spread error up to 162.5% over a 4× regime
> drift; tournament mean Sharpe (5 seeds) — World A: clamp 3.27 > symmetric 2.48 > AS-adaptive
> 1.60 ≈ AS 1.59; World B: AS 2.12 > clamp 0.30 > symmetric 0.25 > AS-adaptive 0.17. Estimator
> recovery R² 0.99998. Fingerprint `1cb0c6bc010a`.

## 6 · Could You Trade It?

Walk it from "beautiful equation" to a live quoting bot. You'd estimate `k` and `σ` from
recent flow, compute the reservation price and spread, and post two quotes that update every
tick. **What actually pays is the skew** — keeping inventory near zero — and that part is
robust, cheap, and worth running, *especially* in markets with jumps and informed flow where
inventory is genuinely dangerous (crypto, exactly the venue the article targets). **What
doesn't earn its billing is the `k`-machinery**: you will spend real engineering on a rolling
`k` estimator feeding a "closed-form optimum" that (a) is closed-form around a parameter your
own data says is unstable, and (b) buys you a spread *width* that a fixed, sensible width — or
a brainless inventory band — gets most of the way to. And the first "production fix" everyone
reaches for, adaptive realised vol, will blow your book out the first time the tape jumps
unless you also build the circuit breaker. The candid bottom line: implement the **skew**;
treat the **spread formula** as a starting heuristic, not a theorem; and don't believe the
"leaving money on the table" pitch — on a benign tape the table is being cleared by four lines
of inventory clamp.

> 🔬 **For the quants** — the deployable decomposition: run `r = s − q·γ·σ²·(T−t)` (robust,
> `k`-free) for inventory control; set the half-spread to a fixed multiple of realised
> (jump-robust!) vol rather than `(1/γ)ln(1+γ/k)`, since the arrival term's `k` is the phantom
> and its closed-form optimality is illusory off-assumption. Reserve the full GLT/Cartea–
> Jaimungal extensions (hard inventory bounds; an explicit adverse-selection term) for when
> the venue's informed flow is material — World B shows that's exactly when the skew, not the
> spread, is what saves you.

## 7 · Going Further

- **The empirical leg — done, and extendable.** The one thing the simulator asserts is that
  real order reach is power-law; [`examples/confirm_heavy_tail.py`](examples/confirm_heavy_tail.py)
  now **confirms it** on four real Binance futures books (Clauset/Vuong). It's one trading day
  per market — the obvious next PR is a **multi-day, multi-venue sweep** (spot vs futures, more
  symbols, several dates) to tighten the tail exponents and check stability, plus a fit *in
  ticks* to remove the residual price-discreteness noise on the price-distance leg.
- **A jump-robust adaptive vol.** Re-run the World-B tournament with a bipower/truncated
  realised-vol estimator in `AdaptiveASQuoter` — does the rolling-vol fix stop collapsing?
- **The GLT closed form with hard inventory bounds.** Implement Guéant–Lehalle–Fernandez-Tapia
  quoting and see whether explicit bounds close the gap to (or beat) the brainless clamp.
- **An adverse-selection spread term (Cartea–Jaimungal).** Add an informed-trading widening to
  the spread and re-test whether AS's World-B edge grows or whether the clamp catches up.
- **Calibrate `γ`.** The clamp-beats-AS result in World A is at `γ = 0.1`; sweep `γ` to find
  where AS's inventory aversion stops costing more P&L than it saves.
- **What to PR:** the real-data confirmation, the jump-robust vol, the GLT quoter, or a
  `γ`-sweep that maps the regions where the skew actually earns its keep.

---

## How this study is laid out

| Path | What's inside |
|---|---|
| [`phantom_kernel/sim.py`](phantom_kernel/sim.py) | the two worlds — exponential vs heavy-tailed order reach, jumps, stochastic vol, informed flow; the order-reach law whose survival function *is* the AS kernel |
| [`phantom_kernel/estimator.py`](phantom_kernel/estimator.py) | fit the kernel (`A`, `k`); exponential-vs-power-law goodness-of-fit; the static-`k` spread error |
| [`phantom_kernel/strategies.py`](phantom_kernel/strategies.py) | the two AS equations + four quoters (AS, adaptive-AS, symmetric, inventory clamp) + the market loop |
| [`phantom_kernel/experiments.py`](phantom_kernel/experiments.py) | the teardown: estimator recovery, kernel falsification, the tournament |
| [`examples/run_experiments.py`](examples/run_experiments.py) | the simulator headline run → [`docs/results.md`](docs/results.md) (fingerprinted) |
| [`examples/fetch_binance.py`](examples/fetch_binance.py) | download + join real Binance futures trades to the live book mid |
| [`examples/confirm_heavy_tail.py`](examples/confirm_heavy_tail.py) | the **real-data confirmation** (Clauset/Vuong over 4 markets) → [`docs/results_real.md`](docs/results_real.md) |
| [`examples/verify_real.py`](examples/verify_real.py) | generic bring-your-own-data harness — the tail test on any `trades.parquet` |
| [`notebooks/`](notebooks/) | `01_for_the_curious` (the story) and `02_for_the_quants` (the teardown), same seven beats |
| [`docs/references.md`](docs/references.md) | the model, its extensions, and the heavy-tail literature |

The engine that produced every number lives at [`../../quantlab/`](../../quantlab/).
