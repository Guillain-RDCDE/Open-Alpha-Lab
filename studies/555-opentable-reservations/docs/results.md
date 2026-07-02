# Results — Study 555 (OpenTable-Reservations): reservations as a restaurant/XLY nowcast

*Generated from [`opentable_reservations/`](../opentable_reservations/). **This study is
synthetic-only**: there is no free, stable, machine-readable OpenTable seated-diners history a
no-key retail stack can fetch and cache reproducibly (see the data-availability note below and in
[`data.py`](../opentable_reservations/data.py)). The numbers here are the **machinery proof** on a
deterministic, seeded synthetic world (`seed = 555`, 312 weekly rows, nowcast **planted** at
`nowcast_beta = 0.35`; demonstrator panel fingerprint `2491c27d3bc8`, null-world fingerprint
`02902c940703`). One documented execution lag: the reservations surprise at the close of week t
positions the basket for week t+1. As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

The claim is an **alt-data nowcast**: a dining-reservation index (OpenTable-style *seated-diners*
YoY) should lead the tape, so this week's reservations *surprise* predicts next week's restaurant-
basket return (and, loosely, consumer-discretionary / XLY). The mechanism is plausible and the
alt-data nowcasting literature is real — but **the free data to test it on a real tape does not
exist**, so the strongest earnable Signal stamp is `WEAK` (literature-plausible, machinery-proven,
never confirmed on a real tape). A `REAL` stamp requires a robust *t* ≥ 2 on a **REAL** tape; there
is none here.

What the code *can* prove — and does — is that the engine is a faithful detector. On the
demonstrator world (nowcast planted), the predictive regression of next-week basket return on this
week's surprise, controlling for the contemporaneous market, has slope-*t* (Newey-West, 4 lags)
**+4.49** with a label-shuffle placebo *p* = **0.0005**. The seed-robust synthetic control (25
seeds) is textbook: at the **null** the mean slope-*t* is **−0.01** (flat — no false signal), and
planting a stronger nowcast drives it to **+1.30 / +2.91 / +4.55** at `nowcast_beta` 0.15 / 0.35 /
0.60. So the machinery works; the study is `WEAK` because there is no real seated-diners tape to
point it at, not because the engine fails.

## Data-availability limitation (the reason for the cap)

OpenTable ran a public "State of the Industry" seated-diners dashboard during ~2020-2022 (daily YoY
diners vs 2019), but it was a moving HTML widget — never a stable CSV/API, repeatedly restated, and
eventually discontinued — and it covers only the pandemic-recovery episode: a single, highly
non-stationary window, not a tradable multi-cycle panel. No cache-first retail fetch can reproduce
it. `fetch_series(...)` therefore returns an **empty frame by design**, `HAVE_REAL` is always
False, and the Signal axis is capped below `REAL`. Named openly here and on the front-card, in the
spirit of the desk's lego-returns / whisky-cask / sneaker-resale studies.

## The predictive regression (demonstrator world, nowcast planted)

Model: `fwd_ret[t+1] = a + b1 * resv_surprise[t] + b2 * mkt_ret[t+1] + e`. The nowcast coefficient
`b1` is the part of next week's basket return explained by this week's reservations surprise
*beyond* the market.

| | value |
|---|---|
| Nowcast slope `b1` | **+0.0050** (return per unit surprise) |
| Slope *t* (Newey-West, 4 lags) | **+4.49** |
| Placebo *p* (2000 label shuffles) | **0.0005** |
| R² (with market control) | **0.55** |
| n (weeks, after the one-week lag + first-year YoY drop) | **259** |

On this *planted* world the nowcast is strongly significant — exactly what a working engine should
report when the effect is present. This is a control statistic, **never** cited as a real-tape
result.

## The timing overlay + costs (gross AND net)

A sign-of-surprise overlay: long the basket when the surprise is positive, flat (or short) when
negative, one-way 5 bps charged on every position change; the short leg pays a 100 bps/yr borrow.

| Overlay | Gross ann. | Net ann. | Buy-and-hold ann. | Net excess ann. | Net-excess IR | Turnover/yr |
|---|---|---|---|---|---|---|
| Long / flat | **+10.3%** | **+9.3%** | +0.4% | **+8.9%** | 0.65 | 20.5 |
| Long / short | **+20.2%** | **+17.6%** | +0.4% | **+17.2%** | 0.63 | 41.2 |

Costs are modest against the *planted* edge (net excess IR ≈ 0.65). But this is the **synthetic
demonstrator**, not a tradable result: with no real seated-diners tape, and a signal a retail desk
cannot license and cache cleanly, the overlay is un-investable in practice — `MIRAGE`.

## Robustness — sub-period slope-*t* (demonstrator world)

Split the sample into four contiguous blocks and re-estimate the nowcast slope-*t*:

| Window | Slope *t* | n | Reads as |
|---|---|---|---|
| 2020-01 → 2021-06 | **+0.12** | 25 | sparse (first-year YoY drop) — no signal |
| 2021-07 → 2022-12 | **+3.00** | 77 | nowcast present |
| 2023-01 → 2024-06 | **+2.04** | 77 | nowcast present |
| 2024-06 → 2025-12 | **+2.72** | 77 | nowcast present |

Sign is stable and positive in the three well-populated blocks; the first block is too short (the
first year is lost to the YoY definition) to say anything. This confirms the planted effect isn't a
single-sub-period artifact — again, a statement about the machinery on synthetic data.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `nowcast_beta` | Mean slope-*t* (25 seeds) | |
|---|---|---|
| 0.00 (null) | **−0.01** | flat — no false signal |
| 0.15 | **+1.30** | nowcast emerging |
| 0.35 | **+2.91** | clears the bar |
| 0.60 | **+4.55** | strong |

At the null the slope-*t* is ≈ 0; planting a genuine nowcast drives it monotonically past +2 as it
grows. The detector works and does not manufacture significance from noise — so if a real
seated-diners tape existed, this engine would give an honest read on it.

## The honest takeaway

Reservation-recovery nowcasting is a plausible, literature-backed idea, and the engine here would
catch it if it were real: on a planted world the surprise predicts next-week basket returns at
*t* +4.49 with placebo *p* 0.0005, the control is flat at the null, and the sign is stable across
sub-periods. **But there is no free real tape** — OpenTable's seated-diners feed was a transient
pandemic-era dashboard, never a cache-able multi-cycle panel — so the claim cannot be confirmed on
real data. `WEAK` on Signal (plausible + machinery-proven, never tape-confirmed), `MIRAGE` on
Tradability (un-licensable/un-cacheable signal, and any edge shown here lives only in the synthetic
world).
