# Results — Study 550 (Box-Office-Momentum): a synthetic teardown of the alt-data claim

*Generated from [`box_office_momentum/`](../box_office_momentum/) on the **deterministic,
offline synthetic panel** (seed 550, 520 weekly rows; the free retail stack cannot reach an
honest historical box-office tape — see the data-availability caveat below and in
[`data.py`](../box_office_momentum/data.py)). Null panel fingerprint `985a49d136c2`; positive-control
panel (`predictive_beta = 0.009`) fingerprint `4efecda3764a`. As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE`

The claim (alt-data folklore): a strong **weekend box office** is a consumer-sentiment *leading
indicator* — busy cinemas foreshadow gains for media/studio stocks and even the broad tape. We
build a box-office **momentum** signal (this week's gross vs its trailing norm) and test whether it
predicts *next-week* forward returns of a media basket and SPY, with a placebo null, a
confound-controlled regression, a signal-timed strategy net of costs, a threshold-robustness sweep,
and a seed-robust synthetic positive control.

Two things kill the signal even before real data:

1. **Data availability.** There is no free, survivorship-honest historical box-office API a no-key
   stack can use, and the *studios* you would trade have been acquired/restructured out of any clean
   15-year panel (Fox → Disney; Time Warner → WBD; Viacom/CBS → Paramount). The study is therefore
   **synthetic-only** — which by house rule caps it at `WEAK`/`NONE` (a `REAL` stamp needs a robust
   *t* ≥ 2 on a **real** tape). Stated on the SIGNAL axis.
2. **The confound the folklore ignores.** Box office and stocks *both* load on the same
   contemporaneous consumer/market factor, and that factor is *persistent* — so a naive predictive
   regression finds a "lead" that is really the market predicting itself. On the honest,
   confound-controlled test the box-office slope collapses.

## Data stamp

| Component | What | Fingerprint |
|---|---|---|
| Synthetic null panel (`predictive_beta = 0`) | 520 weekly (bo-momentum, media-fwd, spy-fwd) rows | `985a49d136c2` |
| Synthetic positive-control panel (`predictive_beta = 0.009`) | same schema, planted lead | `4efecda3764a` |
| Curated box-office index (illustrative shape only) | 144-month stylised level series | hardcoded constant |

## The headline test — media basket (does box office lead media stocks?)

| Test | Value | Reads as |
|---|--:|---|
| Naive predictive slope-*t* (media_fwd on bo_mom) | **+1.53** | below the *t* ≥ 2 bar — no signal |
| R² of the naive regression | **0.0045** | box office explains ~0.5% of next-week returns |
| Placebo *p* (circular-shift null) | **0.139** | consistent with noise |
| **Confound-controlled slope-*t*** (adds contemporaneous market) | **+0.65** | the apparent lead **collapses** |
| Contemporaneous-market control *t* | **+6.20** | the market move is the real driver |

The naive box-office slope is already too weak to clear the bar (*t* +1.53), and once you control
for the contemporaneous market return the box-office slope falls to *t* +0.65. What little
co-movement there was is the common factor, not a genuine lead.

## The broad-tape test — does box office predict SPY? (the mirage)

| Test | Value | Reads as |
|---|--:|---|
| Naive predictive slope-*t* (spy_fwd on bo_mom) | **+3.42** | looks significant… |
| Placebo *p* (circular-shift null) | **0.0005** | …and the placebo is *fooled* |
| Confound-controlled slope-*t* | **+3.11** | stays inflated — an *imperfect* control cannot remove a persistent common factor |

This is the instructive trap. A naive regression *and even the placebo* flag a "signal" — but it is
manufactured entirely by a **persistent common factor** (confident weeks cluster: busy cinemas and
rising markets co-occur, and both persist). You cannot cleanly separate box office's "lead" from the
market's own autocorrelation with a noisy contemporaneous control. A number that survives a placebo
but is a known artifact is exactly what `MIRAGE` means.

## Tradability — the signal-timed strategy (net of costs)

| | value |
|---|--:|
| Buy-and-hold media (annualised) | **+6.55%** |
| Signal-timed long/flat, **gross** | **+8.76%** |
| Signal-timed long/flat, **net** (5 bps one-way per switch) | **+7.53%** |
| **Net − buy-and-hold** | **+0.98 pp/yr** |
| Turnover | **~25 switches/yr** |
| Exposure (fraction of weeks long) | **0.50** |

A thin +1 pp/yr edge — and it is the same factor-persistence artifact, not box office. It does not
survive the threshold sweep:

| Signal threshold | Net − buy-and-hold (pp/yr) |
|---:|--:|
| −0.50 | **−2.0** |
| −0.25 | **−2.5** |
| 0.00 | **+1.0** |
| +0.25 | **+4.0** |
| +0.50 | **+2.9** |

The edge is **negative** at the looser thresholds and only positive when the timer happens to be
long during the persistent up-weeks. Sign-unstable across a trivial knob → `MIRAGE`.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `predictive_beta` | Mean naive slope-*t* | Mean controlled slope-*t* |
|---:|--:|--:|
| 0.000 (null) | **+1.21** | **+0.84** |
| 0.003 | +3.29 | +2.97 |
| 0.006 | +5.37 | +5.09 |
| 0.009 | +7.45 | +7.22 |
| 0.012 | **+9.53** | **+9.34** |

When a *genuine* box-office lead is planted, the engine recovers it and it **survives the control**
(the controlled *t* tracks the naive one and climbs monotonically past *t* = 2, because a real lead
is orthogonal to the contemporaneous market). At the null both sit near ~1 and the control *lowers*
the *t* (+1.21 → +0.84), the honest signature of a co-movement artifact. On the single headline seed
the planted `β = 0.009` world
gives naive *t* **+7.84**, controlled *t* **+7.10**, placebo *p* **0.0005**, and a signal-timed net
edge of **+19.5 pp/yr** — proof the machinery banks a real lead. It does not print one at the null.
*(Control only; never cited for a real-tape stamp — there is no real tape.)*

## Why the claim doesn't certify here

1. **No honest real tape.** Free box-office history is not survivorship-honest and the studio
   universe is a merger graveyard — synthetic-only, so capped at `NONE`.
2. **The common-factor confound.** Box office and stocks co-move because both track the same
   consumer/market factor; the naive "lead" is that factor's persistence. The confound-controlled
   test (media *t* +0.65) is the honest read.
3. **Placebo-fooling by persistence.** The broad-tape "signal" (*t* +3.42) survives a naive placebo
   yet is a pure artifact — a reminder that a single placebo is not proof; a structural control is.
4. **Fragile timing edge.** The net-of-cost strategy beats buy-and-hold by ~1 pp/yr and flips sign
   across thresholds.

## The honest takeaway

Box-office momentum is a fun, intuitive alt-data story, but on an honest test it is a `NONE` × `MIRAGE`:
the media-stock lead is not there once you control for the contemporaneous market (*t* +0.65), the
broad-tape "signal" (*t* +3.42) is a placebo-fooling common-factor artifact, and the signal-timed
strategy's ~1 pp/yr edge flips sign across thresholds. The synthetic control confirms the engine
*would* catch a real lead — so this is the structure of the claim talking, not a broken detector.
And because no survivorship-honest free box-office tape exists, the study can never rise above `NONE`.
