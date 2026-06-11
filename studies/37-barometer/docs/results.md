# Results — Study 37 (Barometer): cross-asset macro momentum & inflation hedging

> **Real run.** Numbers below are from `examples/verify.py` on the desk's cached tape — three parquets:
> `macro_us` (BLS CPI level + industrial production), `us_treasury_yields` (the `y10y − y3m` slope, a daily
> growth proxy), and `cross_asset_etfs` (18 liquid ETFs). All macro drivers are **lagged one month** for
> publication delay (CPI/IP are released ~2 weeks after the month they describe; the slope is observable in
> real time but lagged the same month for one clean alignment). The sample begins **2007** — when the
> real-asset ETFs that make the inflation hedge tradable (DBC, DBA, SLV, USO, GLD, TIP, UUP, HYG) exist —
> and ends where the cached CPI does. Offline, fully reproducible.
>
> ```
> [data] real cross-asset book: 217 months  2007-02-28 -> 2025-02-28  as-of 2025-02-28  fingerprint baa416a9db25
> ```
>
> **Short-sample caveat, stated up front.** 217 months (~18 years) post-2007 is *one* macro cycle and one
> big inflation episode (2021-22). It is long enough to test the *mechanism* and the *sign*, far too short
> to pin a magnitude or clear a significance bar — every t-stat below is small. Read the verdict as a
> direction, not a tradable Sharpe.

## The verdict — Signal `REAL` (direction) / `WEAK` (size) · Tradability `FRAGILE` · Real-tape run? `DONE`

The *trend* in fundamental macro data is a real, slow, cross-asset predictor — but on this short real tape
it is **weak and noisy**, and neither macro book beats a passive hold of the same assets. The headline:

| book (net @5 bp) | Sharpe | CAGR | maxDD | turnover | HAC *t* (NW, 6 lags) | break-even |
|---|---|---|---|---|---|---|
| **macro-momentum** | **−0.05** | −0.7% | −29% | 7.3×/yr | −0.22 | 0 bp (gross ≤ 0) |
| **inflation-hedge** | **−0.12** | −0.8% | −31% | 5.5×/yr | −0.42 | 0 bp (gross ≤ 0) |
| *benchmark* equal-weight 18 ETFs | **+0.57** | +5.2% | −30% | — | — | — |
| *benchmark* 60/40 SPY/IEF | **+0.84** | +7.8% | −29% | — | — | — |

So as *standalone* dollar-neutral books, both are flat-to-negative and dominated by simply holding the
assets. **Cost is not what kills them** — turnover is low (5-7×/yr) and the cost sweep degrades linearly
(net Sharpe −0.01 → −0.05 → −0.10 → −0.24 → −0.47 at 0/5/10/25/50 bp) — they make nothing *gross* to begin
with. The kill is the short, noisy sample and the modest, slow nature of the premium.

## But the inflation hedge *does* pay in the regime it targets — directionally

The honest conditional test (beat-7) splits the inflation book by realized inflation regime, and it lands
on the right side of zero:

| inflation-hedge book | rising inflation | falling inflation |
|---|---|---|
| Sharpe | **−0.08** | −0.16 |
| ann. return | **−0.5%/yr** | −0.9%/yr |
| months | 104 | 107 |

The book is **less bad when inflation is rising** than falling — the conditional shape the steelman
predicts — but the momentum-*timed* signing whipsaws enough to keep it under water in both. Strip the
timing out and ask the barest question — does an always-long real-asset basket simply out-earn nominal
bonds (TLT/IEF) when inflation rises? — and the effect is unambiguous:

| raw real-minus-nominal spread (always long) | rising inflation | falling inflation |
|---|---|---|
| Sharpe | **+0.10** | −0.01 |
| ann. return | **+1.8%/yr** | −0.3%/yr |

**Real assets out-earn nominal bonds specifically in rising-inflation months (+1.8%/yr) and not in falling
ones (−0.3%/yr).** That is the inflation-hedge mechanism, confirmed on the real tape — the *direction* is
real. What's `FRAGILE` is harvesting it with a monthly-momentum timing rule: the signal is so slow and the
sample so short that the timed book gives the raw edge back to noise.

## What the synthetic control proves (offline, reproducible — the machinery)

The mechanism is validated end-to-end on a synthetic cross-asset world (5 assets: equities, nominal bonds,
commodities, a TIPS/real-rate proxy, gold) driven by two latent, persistent, regime-switching macro states
— *growth* and *inflation* — whose **momentum** (one-month change, lagged) predicts next-month returns
through fixed signed betas (seed 37, 50 years, gross of cost):

- **Macro momentum is real and recovered:** the book earns **+5.1%/yr** at Sharpe **+1.09**, maxDD **−12%**,
  on low turnover (**5.6×/yr**); on the `macro_strength = 0` **null** (pure noise) it collapses to Sharpe
  **−0.17** — the apparatus measures the effect, not itself.
- **The inflation hedge pays and is regime-dependent:** the tilt earns **+2.3%/yr** at Sharpe **+0.55**
  (null **−0.02**), and the [regime split](extension.md) shows **more in rising-inflation regimes (Sharpe
  +0.59) than falling (+0.46)** — the same conditional shape the real tape confirms in direction.
- **Cost is not the threat:** break-even **~91 bp** (macro momentum) / **~60 bp** (inflation) per unit —
  far above realistic cross-asset costs.

The control proves the machinery *can* extract the premium when it is strong and clean; the real tape shows
that, over one short post-2007 cycle, the live premium is too weak and slow to clear noise as a standalone
dollar-neutral book — even as its core *directional* claim (real assets beat nominal bonds when inflation
rises) holds.

## Reproduce

```
python examples/verify.py             # real book, 18 ETFs + cached macro, offline + fingerprinted
python examples/run_synthetic_demo.py  # the offline machinery proof (control vs null)
```

*Sources & literature map: [docs/references.md](references.md). Engine: [`quantlab/`](../../../quantlab/).
**Not investment advice** — research & education.*
