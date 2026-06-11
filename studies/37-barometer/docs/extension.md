# Beat 7 — worked complement: does the inflation hedge pay when it's supposed to? (Study 37)

> **Real tape + synthetic control.** This beat-7 regime split runs on **both** the real cross-asset tape
> (18 ETFs + cached CPI / yield-curve macro, 2007-2025, lagged one month — see [`results.md`](results.md))
> and the synthetic control. The control proves the mechanism; the real tape tests whether it survives a
> short, noisy, post-2007 sample.

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

## The real tape (18 ETFs + cached CPI/yield-slope macro, 2007-02 → 2025-02, net @5 bp)

| inflation-hedge book | rising inflation | falling inflation |
|---|---|---|
| Sharpe | **−0.08** | −0.16 |
| ann. return | **−0.5%/yr** | −0.9%/yr |
| months | 104 | 107 |

The timed inflation book is **less bad when inflation is rising** (the right-sided shape) but the slow
monthly-momentum signing whipsaws it under water in both regimes over this short sample. Strip the timing
out and the directional mechanism is unambiguous — an always-long real-asset basket vs nominal bonds
(TLT/IEF):

| raw real-minus-nominal spread | rising inflation | falling inflation |
|---|---|---|
| Sharpe | **+0.10** | −0.01 |
| ann. return | **+1.8%/yr** | −0.3%/yr |

**Real assets out-earn nominal bonds specifically in rising-inflation months (+1.8%/yr), not in falling
ones (−0.3%/yr)** — the inflation-hedge mechanism, confirmed on the real tape in *direction*. What is
`FRAGILE` is monetising it with a slow monthly timing rule on one short cycle.

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
sleeve, and to combine it with the always-on macro-momentum book for diversification. The real tape
sharpens the point: the *direction* survives (real assets beat nominal bonds when inflation rises) but the
*timed monthly book* does not clear noise on one short post-2007 cycle — so the honest use is as a slow,
conditional overlay sized off a clear inflation-momentum signal, not a standalone Sharpe source.

*Engine: [`quantlab/`](../../../quantlab/). Not investment advice — research and education.*
