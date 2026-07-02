"""Study 580 — Gold-Lease-Rate (commodity-microstructure folklore).

The **gold lease rate** is the cost to *borrow physical bullion*. Classically it is backed out of
two quoted rates: GOFO (the Gold Forward Offered rate — the rate to swap gold for USD) and LIBOR:

    lease_rate ≈ LIBOR − GOFO

Bullion-market folklore holds that the lease rate *leads* the gold price: a spike in the cost to
borrow gold (a scramble for physical metal, backwardation) is said to foreshadow a rally, and a
collapse toward zero (ample metal, contango) a drift lower. This study asks whether an *obscure
microstructure rate can time gold* — the kind of clean, seductive, and usually spurious lead-lag
claim the desk exists to stress-test.

**Data caveat (front and centre):** the LBMA *discontinued* the daily GOFO benchmark on
30 January 2015, and no free, continuous, long-history gold-lease-rate series exists on a no-key
retail stack. So this study is **synthetic-only**: a deterministic generator plants (or withholds)
a lease-rate→gold lead-lag, and the engine proves it can catch a real one and stays flat at the
null. With no real tape, the SIGNAL axis is capped at WEAK — a synthetic-only study can never earn
REAL (that needs a robust t ≥ 2 on a real tape).

Distinct from the desk's other gold work — [113 gold-silver-ratio](../113-gold-silver-ratio/),
[305 gold-oil-ratio](../305-gold-oil-ratio/), [388 lumber-gold-ratio](../388-lumber-gold-ratio/)
(cross-asset *ratios*) and [208 gold-miners](../208-gold-miners/) (the equity beta): this is a
*microstructure carry/borrow* signal, not a price ratio.
"""
