# Results — Study 551 (Netflix-Top10): streaming-engagement momentum as an alt-data signal

*Generated from [`netflix_top10/`](../netflix_top10/) on the **deterministic synthetic world**
(`netflix_top10/data.py`, seed 551): 208 weekly Top-10 engagement reports → 197 usable weekly
rows after the momentum lookback and the 4-week forward window (world fingerprint `a31af2c4d9d7`).
**There is no real tape** — no free, machine-readable, point-in-time Netflix Top-10 hours feed
exists for a no-key retail stack (the series is short, 2021+, and methodology-revised), so this is
a **synthetic-only** study, capped at `WEAK`/`NONE` by house rule. As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Alt-data edge?" `MIRAGE`

The claim: **streaming-engagement momentum** — the acceleration in how many hours the world
watches Netflix's Top-10 — *predicts* forward returns for NFLX and, by spillover, consumer
discretionary (XLY). We build a standardised engagement-momentum signal, regress forward returns
on it, run a label-shuffle placebo, a long/flat (and long/short) timing backtest with costs and a
short borrow, a forward-horizon robustness sweep, and a seed-robust synthetic positive control.

Because no research-grade real tape exists, the honest headline is the **null synthetic world**
(`beta = 0`, engagement carries *no* return information by construction). On the seed-551 draw the
plain-OLS slope of NFLX forward return on engagement momentum is **−0.0106** with **OLS *t*
−2.03** — which *looks* significant. It is a **false positive**: the 4-week forward windows
**overlap**, inflating the naive OLS *t*. The overlap-robust **Newey-West *t* is −1.48** (below
the bar), the **placebo *p* is 0.046** (borderline, the shuffle sees the same overlap), and across
**20 seeds the mean NW *t* is +0.11 with 0/20 clearing |*t*| ≥ 2**. So there is no signal — the
single-seed "significance" is exactly the overlap/multiple-testing trap the desk exists to catch.
The XLY spillover leg is flat too (slope −0.002, *t* −0.47). `NONE` on signal; the timing rule
*loses* to buy-and-hold before costs (`MIRAGE`); and since the whole premise rests on a synthetic
stand-in for data you cannot actually buy, the "alt-data edge" is a `MIRAGE`.

## Data stamp

- **Synthetic weekly world** (seed 551, `beta = 0` null): 197 rows, fingerprint `a31af2c4d9d7`
- **Columns**: `engagement` (index level), `eng_mom` (z-scored momentum), `nflx_ret`/`xly_ret`
  (weekly), `nflx_fwd`/`xly_fwd` (4-week forward)
- **Real tape**: *none* — `fetch_series(fetch=True)` raises with the reason (no free point-in-time
  Top-10 hours feed); the caveat is executable so it cannot rot

## The predictive regression — NFLX forward return on engagement momentum (null world)

| | value |
|---|---|
| Slope | **−0.0106** per momentum unit |
| OLS *t* (naive) | **−2.03** — *looks* significant |
| Newey-West *t* (overlap-robust) | **−1.48** — below the bar |
| corr(eng_mom, nflx_fwd) | **−0.14** |
| Placebo *p* (2000 shuffles) | **0.046** |
| n (weeks) | 197 |

The naive OLS *t* clears −2, but the forward windows overlap (each 4-week return shares 3 weeks
with its neighbour), so the naive standard error is too small. The Newey-West correction — the
honest stat — is **−1.48**, and it is the *wrong sign* anyway (a spurious negative, not the
claimed positive predictive slope).

## The seed-robust view — the single seed is a mirage

| | Newey-West slope *t* |
|---|---|
| Seed 551 (headline draw) | **−1.48** |
| Mean over 20 seeds (551–570) | **+0.11** |
| Fraction of seeds with |NW *t*| ≥ 2 | **0/20** |

Averaged over 20 null worlds the predictive *t* is essentially zero and never clears the bar. The
seed-551 "−2.03" is one unlucky draw magnified by overlap — a textbook false positive.

## The XLY spillover leg — no consumer-discretionary echo

| | value |
|---|---|
| Slope (xly_fwd on eng_mom) | **−0.002** per unit |
| OLS *t* | **−0.47** |
| corr | **−0.03** |

By construction XLY loads on the common consumer factor but **not** on engagement — and the fit
confirms it. Even a genuine NFLX-specific engagement effect would not mechanically spill to XLY.

## Robustness — forward-horizon sweep (null world, seed 551)

| Forward horizon | Slope | OLS *t* | NW *t* |
|---|---|---|---|
| 2 weeks | −0.0032 | −0.96 | −0.89 |
| 4 weeks (headline) | −0.0106 | −2.03 | −1.48 |
| 8 weeks | −0.0334 | −4.34 | −3.29 |
| 13 weeks | −0.0410 | −3.92 | −3.12 |

The longer-horizon "significance" is **pure overlap**: a 13-week forward return shares 12 of 13
weeks with its neighbour, so hundreds of near-duplicate rows masquerade as independent
observations. This is *why* the naive OLS *t* is untrustworthy on overlapping windows — the effect
is a null artifact, not a signal (the seed-averaged mean is ~0 at every horizon).

## Costs — the timing rule loses before frictions

| Rule (null world, NFLX) | Gross ann. | Net ann. | Buy-and-hold ann. |
|---|---|---|---|
| Long/flat (in when eng_mom > 0) | **−3.6%** | **−4.4%** | **+21.8%** |
| Long/short (short when eng_mom < 0, pays borrow) | **−25.7%** | **−27.1%** | **+21.8%** |

Timing on a *noise* signal underperforms simply holding the stock — dramatically so when it also
shorts. Net of 5 bps/side one-way costs (56 position changes) and a 100 bps/yr borrow on the short
leg, both variants are deep in the red versus buy-and-hold. `MIRAGE`.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `beta` | Mean NW slope-*t* (25 seeds) | |
|---|---|---|
| 0.000 (null) | **−0.08** | flat — no false signal |
| 0.005 | +0.49 | emerging |
| 0.010 | +1.01 | visible |
| 0.020 | +1.78 | strong |
| 0.040 | **+2.51** | clears the bar |

At the null the mean NW *t* is ≈ 0; planting a genuine engagement→return link (`beta > 0`) drives
the *overlap-robust* *t* positive and past +2 as it grows. The detector works — so the null-world
non-result is a real statement, not a broken engine. (Control only; never cited for a real-tape
stamp, because there is no real tape.)

## Why this can never be `REAL` here

1. **No real tape.** The public Top-10 hours series is short (2021+), methodology-revised, and
   published as PDFs / a JS dashboard — not a free, point-in-time, machine-readable panel. A
   synthetic-only study cannot earn `REAL` (which requires a robust *t* ≥ 2 on a real tape).
2. **Overlap eats the *t*.** Weekly signals against multi-week forward returns overlap heavily;
   the naive OLS *t* is inflated, and the honest (Newey-West, seed-averaged) view is flat.
3. **Reflexivity even if real.** Engagement reports lag the viewing week and are widely covered,
   so any true edge would be arbitraged before a retail reader could act — the alt-data would be
   priced in.

## The honest takeaway

Streaming-engagement momentum is a seductive alt-data story, but on this desk it fails three ways:
there is **no free real tape** to test it on (so it is synthetic-only and cannot be `REAL`); the
synthetic *null* already prints a single-seed "significant" slope that **evaporates** under an
overlap-robust standard error and seed-averaging (mean NW *t* +0.11, 0/20 seeds clear the bar);
and a timing rule on it **loses to buy-and-hold** before costs. The synthetic positive control
confirms the engine *would* catch a genuine engagement→return link — so this is a statement about
the data and the overlap trap, not the code. `NONE` × `MIRAGE`.
