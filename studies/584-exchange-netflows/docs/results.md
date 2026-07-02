# Results — Study 584 (Exchange-Netflows): the "coins-to-exchanges = bearish BTC" claim

*Generated from [`exchange_netflows/`](../exchange_netflows/) on the **deterministic synthetic
world** in [`data.py`](../exchange_netflows/data.py) (seed 584, 1500 business days from 2019-01-01;
series fingerprint `8c7301772d7b`, forward-return fingerprint `e3476cff1898`). **There is no real
exchange-netflow tape**: per-entity, exchange-tagged on-chain flow is a paid address-clustering
product (Glassnode / CryptoQuant / Nansen) with no usable free tier — so a `REAL` stamp (which
needs a robust t ≥ 2 on a REAL tape) is **out of reach by construction**, and this study is capped
at WEAK/NONE. As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Would the engine catch it if it were real?" `YES`

The on-chain folklore (CryptoQuant, Glassnode, Nansen): when BTC **flows onto** exchanges
(positive *net-inflow* = deposits minus withdrawals), holders are staging coins to **sell**, so a
rising exchange net-inflow is a **bearish** lead for BTC's forward return; a net-*outflow* (coins
to cold storage) is bullish accumulation. We build a netflow-vs-forward-return engine — a
cross-sectional slope whose sign *is* the folklore, a high-inflow-vs-low-inflow sort with a Welch
*t*, a label-shuffle placebo, a signal-timed long/short book with costs and borrow, a lag/threshold
robustness sweep, and a seed-robust synthetic positive control.

**The honest headline is the null.** On the fair-null synthetic world (netflow tells you nothing,
`bear_beta = 0`) the slope of forward BTC return on standardised net-inflow is **+4.3 bps/σ**
(slope-*t* **+0.47**), the high-inflow-minus-low-inflow sort spread is a statistically empty
**+10.3 bps** (Welch *t* **+0.38**, placebo *p* **0.72**), and the tradable book *loses* net
(**−6.4 bps/period**, mean net Sharpe **−0.32** across 25 seeds). So `NONE` on the signal axis (no
real tape exists to certify anything, and a fair simulation of the null shows nothing to certify),
`MIRAGE` on tradability (the netflow series you would trade on is itself unreachable without a paid
API, and even the simulated book bleeds costs), and the third axis records the machinery is
faithful: plant the folklore and the engine banks it (below).

## Data stamp

- **Synthetic world**: seed 584, 1500 business days (2019-01-01 →), BTC daily σ ≈ 3.5%, drift
  ≈ 7 bps/day; net-inflow AR(1) φ = 0.6. Null world (`bear_beta = 0`) series fingerprint
  `8c7301772d7b`; forward 1-day return fingerprint `e3476cff1898`.
- **No real data.** A real BTC exchange-netflow tape requires exchange-labelled address clusters —
  a proprietary product. This limitation is named on the SIGNAL axis; the study cannot exceed WEAK.

## The signal — nothing at the null (as it should be)

| Read (null world, lag 1) | Value |
|---|---|
| Slope of forward return on z(net-inflow) | **+4.3 bps/σ** (folklore predicts *negative*) |
| Slope *t* | **+0.47** |
| corr(net-inflow, forward return) | **+0.012** |
| Low-inflow (outflow) days forward mean | **+13.3 bps** |
| High-inflow days forward mean | **+2.9 bps** |
| Sort spread (low − high) | **+10.3 bps** (Welch *t* **+0.38**) |
| Label-shuffle placebo *p* | **0.72** |

At the null the netflow carries no information about the forward return — the slope is a
statistical zero, the sort spread is noise, and the placebo *p* sits mid-distribution. Exactly what
an honest engine must print when there is nothing there.

## Robustness — the null stays null across lags and thresholds

Slope-*t* by execution lag (the sort spread flips sign harmlessly across tail fractions):

| lag | slope-*t* | sort-*t* (frac 0.2) |
|---|---|---|
| 1 | **+0.47** | +0.38 |
| 2 | **+0.58** | +0.09 |
| 3 | **+0.68** | −0.48 |
| 5 | **+0.53** | −0.05 |

No lag or threshold pushes the relation anywhere near significance — there is no fragile corner
where a signal hides. (Full grid in [`strategy.robustness_sweep`](../exchange_netflows/strategy.py).)

## Costs — the simulated book loses before and after frictions

| Netflow-timed long/short book (null world) | Value |
|---|---|
| Gross mean per period | **−1.1 bps** |
| Net (10 bps/turn one-way + 300 bps/yr short borrow) | **−6.4 bps** |
| Turnover (avg |Δposition|) | **0.49** |
| Mean net Sharpe (25 seeds) | **−0.32** |

Trading the netflow sign on the null world is a slow bleed: near-zero gross, negative net after a
crypto-realistic per-turn cost and a short-borrow/funding charge. `MIRAGE`.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `bear_beta` | Mean slope-*t* (25 seeds) | Reads as |
|---|---|---|
| 0.0000 (null) | **−0.01** | flat — no false signal |
| −0.0010 | −1.12 | folklore emerging |
| −0.0025 | **−2.78** | clears the bar |
| −0.0040 | −4.45 | strong |
| −0.0060 | −6.67 | very strong |

At the null the mean slope-*t* is ≈ 0; planting the bearish folklore (`bear_beta < 0`) drives the
slope negative and past −2 as it grows. On a single planted seed (`bear_beta = −0.0025`) the sort
spread is **+80.2 bps** (Welch *t* **+2.95**, placebo *p* **0.005**) and the book earns a gross
Sharpe **0.97** (net **0.66**). **So the detector works** — the empty real-side result is a
statement about *data availability*, not a broken engine. (Control only; never cited for the Signal
stamp.)

## Why this can never certify here

1. **No real tape.** Exchange net-inflow is defined by *exchange labels* on on-chain addresses —
   a paid clustering product. The raw blockchain gives transfers, not the labels. A `REAL` stamp
   (robust *t* ≥ 2 on real data) is unreachable; the study is synthetic-only and capped at WEAK.
2. **Even the fair null is empty.** With no planted coupling the engine finds nothing (slope-*t*
   +0.47, placebo *p* 0.72) — the machinery does not manufacture a signal.
3. **The tradable book bleeds.** Sign-timing the netflow costs turnover and (on the short leg)
   borrow; the net Sharpe is negative at the null.

## The honest takeaway

The "coins-to-exchanges = bearish" story is intuitive and widely quoted on on-chain desks — but on
a **no-key retail stack there is no exchange-netflow series to test it on**, and a fair simulation
of the null shows an engine that (correctly) finds nothing. The synthetic positive control proves
the harness *would* bank the folklore if it were real (slope-*t* −2.78 at `bear_beta = −0.0025`,
25-seed) — so the verdict is `NONE` × `MIRAGE`, gated entirely by the paywall on the underlying
data, with the machinery certified honest.
