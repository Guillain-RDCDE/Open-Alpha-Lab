"""Study 36 — Greenback: dollar-carry and the carry⊕momentum combo in currencies.

Where [Study 27 (Steamroller)](../../27-steamroller/) measured the *G10 carry premium itself* — high-rate
currencies out-earn low-rate ones, with a brutal negative-skew crash — Greenback builds on it and asks the
*next* question: how do you combine carry with its natural complement, **momentum**, and what does the
**dollar-carry tilt** (be long/short USD vs a basket by the average rate gap) add? The thesis (Lustig–
Roussanov–Verdelhan 2011 on the dollar factor; Koijen–Moskowitz–Pedersen–Vrugt "Carry" 2018; Asness–
Moskowitz–Pedersen "Value and Momentum Everywhere" 2013): carry and momentum pay at *different times*, so
the **carry⊕momentum combo** earns a higher Sharpe than either standalone precisely because the two legs
decorrelate — momentum tends to ride the trend *out of* a carry crash, dulling the steamroller.

A fully-offline **synthetic control** proves the machinery; the **real G10 tape** (OECD 3-month short rates
+ yfinance FX, 2003→2024-01, served from cache) delivers the earned verdict in
[`docs/results.md`](../docs/results.md): carry is real with the steamroller crash, FX momentum decayed to
*negative* over 2003–2024, and the carry⊕momentum combo — though it can't beat carry on Sharpe with a
losing momentum leg — still **diversifies the crash** (decorrelated legs, shallower worst months).
"""

from . import costs, data, extension, strategy  # noqa: F401
