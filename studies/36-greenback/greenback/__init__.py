"""Study 36 — Greenback: dollar-carry and the carry⊕momentum combo in currencies.

Where [Study 27 (Steamroller)](../../27-steamroller/) measured the *G10 carry premium itself* — high-rate
currencies out-earn low-rate ones, with a brutal negative-skew crash — Greenback builds on it and asks the
*next* question: how do you combine carry with its natural complement, **momentum**, and what does the
**dollar-carry tilt** (be long/short USD vs a basket by the average rate gap) add? The thesis (Lustig–
Roussanov–Verdelhan 2011 on the dollar factor; Koijen–Moskowitz–Pedersen–Vrugt "Carry" 2018; Asness–
Moskowitz–Pedersen "Value and Momentum Everywhere" 2013): carry and momentum pay at *different times*, so
the **carry⊕momentum combo** earns a higher Sharpe than either standalone precisely because the two legs
decorrelate — momentum tends to ride the trend *out of* a carry crash, dulling the steamroller.

This study mirrors Steamroller's honesty pattern exactly: a fully-offline **synthetic control** proves the
machinery; the **real-tape verdict is PENDING** one networked fetch of short rates (FRED) — which times out
in this sandbox — so [`docs/results.md`](../docs/results.md) is a transparent pending-fetch stub.
"""

from . import costs, data, extension, strategy  # noqa: F401
