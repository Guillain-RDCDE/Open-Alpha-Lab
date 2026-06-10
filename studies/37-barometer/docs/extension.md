# Beat 7 — worked complement: does the inflation hedge pay when it's supposed to? (Study 37)

> ⚠️ **Real run pending a reliable FRED macro fetch.** This is the beat-7 regime split on the **synthetic
> control**; the real-tape version (the actual rising-inflation episodes — the 1970s, 2008, 2021–22) is
> PENDING a reliable FRED macro fetch, exactly as in [`results.md`](results.md). The synthetic control
> below is the validated proof of the mechanism.

## The conditional claim the hedge has to answer

The entire steelman for the inflation-hedge tilt (Neville et al. 2021) is *conditional*: real assets
(commodities, TIPS, gold) are supposed to earn their keep **specifically when inflation is rising**, and to
be dead weight — or a drag — when it is falling. A tilt that pays the *same* in both regimes isn't an
inflation hedge; it's just a long-real-assets bet wearing a macro costume. So the honest test is to **split
the book's returns by inflation regime** and ask whether the premium really concentrates where the theory
says it should.

We label each month rising- vs falling-inflation from that month's inflation momentum (a regime
*attribution*, not a trading signal — it conditions only on the same-month realized macro state, exactly
what an ex-post regime study does), and report each book's Sharpe and annualised return in each regime.

## The synthetic control (seed 37, 50 years, gross of cost)

| | rising inflation | falling inflation |
|---|---|---|
| **inflation-hedge book** — Sharpe | **+0.59** | +0.46 |
| **inflation-hedge book** — ann. return | **+2.5%/yr** | +1.9%/yr |
| macro-momentum book — Sharpe | +0.99 | +1.33 |
| macro-momentum book — ann. return | +4.8%/yr | +6.0%/yr |

**The inflation-hedge book earns materially more in the rising-inflation regime** (Sharpe +0.59 vs +0.46;
+2.5% vs +1.9%/yr) — it pays when it is supposed to, the conditional behaviour the steelman claims. Note it
is still mildly positive in the falling regime, because the real-asset basket carries its own drift — an
honest reminder that the hedge is *not* free insurance.

By contrast the broader **macro-momentum book is steadier across both regimes** (indeed slightly stronger
in the falling-inflation months), because it also rides the *growth* driver, which is independent of the
inflation cycle — so it does not depend on inflation rising to make money.

**Takeaway.** `WEAK`/`REAL` · `FRAGILE` survives the worked complement, and the complement sharpens *why*
the inflation hedge is `FRAGILE`: it is an **episodic** premium that concentrates in rising-inflation
regimes. The honest framing — and the fork for the next contributor — is to use the inflation tilt as a
*conditional overlay* (size it up only when inflation momentum is clearly positive) rather than a permanent
sleeve, and to combine it with the always-on macro-momentum book for diversification. The real-tape version
(via `--fetch`) tests the same split on the actual historical inflation episodes.

*Engine: [`quantlab/`](../../../quantlab/). Not investment advice — research and education.*
