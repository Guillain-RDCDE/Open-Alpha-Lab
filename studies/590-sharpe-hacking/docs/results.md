# Results — Study 590 (Sharpe-Hacking): manufacturing a Sharpe with financial engineering

*Generated from [`sharpe_hacking/`](../sharpe_hacking/) on a **deterministic, offline synthetic
tape** (seed 590). This is a research-method demo, so the tape is built on purpose: the **null**
carries an honest annualised Sharpe of ≈ 0.24 and **zero timing/selection alpha** (`alpha = 0`), and
the **positive control** plants a genuine edge (`alpha > 0`). Real free data can never certify "zero
alpha", so there is no real-tape stamp; the data-availability limitation is named on the SIGNAL axis
and the study is capped at `NONE`. Null tape fingerprint `807d5cddc33f` (3,000 daily rows,
2008-01-02 → 2019-07-02). As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Can financial engineering fake a Sharpe?" `CONFIRMED`

We take a synthetic return stream with an honest annualised Sharpe of **0.24** and **no genuine
edge**, and run three pieces of pure financial engineering over it — none of which adds a cent of
real return — while watching two Sharpe numbers: the **naive** reported Sharpe everyone quotes
(mean/std × √252) and the **honest**, autocorrelation-corrected Sharpe (Lo 2002 / Getmansky-Lo-Makarov
2004), which is the yardstick that *cannot be gamed by smoothing*.

- **Return smoothing (illiquidity / stale marks)** is the big fake. Reporting an AR(1)-smoothed
  version of the returns (θ = 0.5) lifts the **naive** Sharpe from **0.32 → 0.56** (+73%) while the
  **honest** Sharpe moves **0.242 → 0.244** — essentially unchanged. Crank the staleness to θ = 0.8
  and the naive Sharpe balloons to **0.98** (a *3.0× inflation*) while the honest one stays flat at
  ~0.24. The whole "improvement" is the autocorrelation smoothing injects (lag-1 autocorr 0.01 →
  0.81), which naive annualisation mistakes for skill.
- **Naive leverage** does **nothing** to the Sharpe. Scaling every return by 3× leaves both the naive
  and honest Sharpe *exactly* unchanged at **0.323 / 0.242** — mean and standard deviation scale
  together. Leverage buys volatility and drawdown, not risk-adjusted return; the folk claim is
  false.
- **Volatility targeting** is not a free lunch either. On this tape, rescaling to a 10% vol target
  *lowers* the naive Sharpe from **0.32 → 0.16** (and net of a 2 bps turnover cost, **0.145**) — the
  variance-timing "gain" people expect simply isn't there when the mean return isn't concentrated in
  the low-vol days. The data decides, and here it says vol-targeting hurt.

So `NONE` on the signal axis (a synthetic-only method demo — the only Sharpe "gain" is a measurement
artefact, and there is no real edge to detect), `MIRAGE` on tradability (smoothing is an accounting
illusion you cannot spend, leverage changes nothing, and vol-targeting loses net of costs here), and
`CONFIRMED` on the myth-check (yes, you can inflate a *reported* Sharpe by 73–200% with pure
engineering — but only the reported one).

## Data stamp

- **Null tape** (`alpha = 0`, honest Sharpe ≈ 0.24, time-varying volatility): 3,000 daily rows,
  2008-01-02 → 2019-07-02, fingerprint `807d5cddc33f`, seed 590.
- **Positive-control tapes** (`alpha ∈ {0.05, 0.10, 0.20}`): same generator with a genuinely higher
  risk-adjusted return baked in.

## The headline — three levers, one honest yardstick

| Transform | Naive Sharpe | Honest Sharpe | Naive inflation | Honest inflation |
|---|--:|--:|--:|--:|
| **raw (baseline)** | 0.323 | 0.242 | 0.000 | 0.000 |
| **smoothed (θ = 0.5)** | **0.559** | **0.244** | **+0.236** | +0.002 |
| **levered (L = 3)** | 0.323 | 0.242 | 0.000 | 0.000 |
| **vol-targeted (tgt 10%)** | 0.158 | 0.136 | −0.165 | −0.106 |

The **naive inflation** column is the game: smoothing adds +0.24 of *reported* Sharpe out of thin
air, leverage adds exactly nothing, and vol-targeting *subtracts* here. The **honest inflation**
column is the truth: every lever moves the autocorrelation-corrected Sharpe by ≈ 0 (smoothing
+0.002) or genuinely down (vol-targeting −0.106). No lever manufactures real edge.

## Smoothing is the fake — the naive Sharpe climbs with the staleness, the honest one doesn't

| Smoothing θ | Naive Sharpe | Honest Sharpe | Lag-1 autocorr |
|---|--:|--:|--:|
| 0.0 (no smoothing) | 0.323 | 0.242 | 0.01 |
| 0.2 | 0.394 | 0.242 | 0.21 |
| 0.4 | 0.492 | 0.243 | 0.42 |
| 0.6 | 0.647 | 0.246 | 0.61 |
| 0.8 | **0.979** | 0.251 | 0.81 |

More staleness → more injected autocorrelation → a bigger *reported* Sharpe, all the way to a
**3.0× inflation** at θ = 0.8. The honest Sharpe barely twitches (0.242 → 0.251). This is the exact
mechanism behind the suspiciously smooth, high-Sharpe track records of illiquid books that mark
themselves.

## The bootstrap bands — which metric is the artefact

A circular-block bootstrap (block = 20 days) on the θ = 0.5 inflation:

| Inflation | Point estimate | 95% CI |
|---|--:|--:|
| **Naive** (reported Sharpe) | **+0.236** | firmly positive across the θ sweep |
| **Honest** (corrected Sharpe) | **+0.002** | **[−0.012, +0.013]** — straddles 0 |

The honest-inflation band **contains zero**; the naive inflation is a systematic, mechanical lift.
The fraction of bootstrap draws whose honest inflation reaches the naive point estimate is **0.00** —
the honest metric never mimics the game.

## Costs — the vol-targeting lever, net

| Quantity | Value |
|---|--:|
| Gross naive Sharpe (vol-targeted) | 0.158 |
| Net naive Sharpe (2 bps per exposure change) | **0.145** |
| Gross honest Sharpe | 0.136 |
| Net honest Sharpe | **0.125** |
| Turnover per day | 0.030 |

Vol-targeting trades every day; even a mild 2 bps cost trims the (already-below-baseline) Sharpe
further. There is nothing here to harvest — `MIRAGE`.

## Synthetic positive control — the honest Sharpe tracks REAL edge, the game does not (25 seeds)

The same protocol on tapes with a genuinely planted edge, averaged over 25 seeds (the house rule):

| Planted `alpha` | Mean honest Sharpe (raw) | Mean **naive** inflation (smoothing) | Mean **honest** inflation |
|---|--:|--:|--:|
| 0.00 (null) | **0.30** | +0.218 | +0.002 |
| 0.05 | **0.57** | +0.396 | +0.003 |
| 0.10 | **0.83** | +0.573 | +0.004 |
| 0.20 | **1.37** | +0.927 | +0.007 |

Two things hold across every world: (a) the **honest** Sharpe of the raw series *rises with the
planted alpha* — the corrected metric tracks real edge; and (b) the **naive** smoothing inflation
stays large and positive *regardless of alpha* — smoothing games the report the same way whether or
not there is real skill underneath. The honest inflation is a negligible **+0.002 to +0.007** Sharpe
units — economically zero, ~100× smaller than the naive game (the tiny residual is a second-order
interaction with the vol clustering; its across-seed *t* rises only because the bias is *consistent*,
not because it is *large*). The correction cleanly isolates real edge from measurement games.

## Why the verdict is what it is

1. **The only Sharpe "gain" is a measurement artefact.** On a tape we *built* to have an honest
   Sharpe of 0.24 and zero alpha, smoothing manufactures a *reported* 0.56 (θ 0.5) or 0.98 (θ 0.8),
   the honest Sharpe never budges, leverage does literally nothing, and vol-targeting loses. There is
   no real edge to detect. **Signal `NONE`.**
2. **Nothing to trade.** A smoothed Sharpe is an accounting illusion you cannot spend; leverage
   changes no risk-adjusted return; vol-targeting is below baseline and negative net of costs.
   **Tradability `MIRAGE`.**
3. **The myth is confirmed — with a sharp caveat.** *Yes*, you can inflate a **reported** Sharpe by
   73–200% with pure financial engineering (smoothing), and leverage/vol-targeting are the folk
   levers people *think* help but don't. The catch is that only the naive metric moves; the
   autocorrelation-corrected Sharpe is immune. **`CONFIRMED`.**

## The honest takeaway

A Sharpe ratio is only as honest as the way it is measured. Return smoothing — the mechanical
signature of illiquid, self-marked books — inflates the *reported* Sharpe without adding a cent of
return, and the inflation grows with the staleness. Leverage does nothing to the ratio at all, and
volatility targeting is no free lunch. The autocorrelation-corrected Sharpe (Lo 2002 / GLM 2004) sees
through the smoothing, and the synthetic control shows it still tracks *genuine* edge. `NONE` ×
`MIRAGE`, myth `CONFIRMED`. This is a method demo on a synthetic world by design — it can never earn
`REAL`, which requires a robust *t* ≥ 2 on a real tape.
