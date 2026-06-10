# Beat-7 worked complement — "does adding a momentum sleeve diversify the carry book?"

> ⚠️ **Real-tape numbers pending a term-structure fetch.** The carry leg needs roll yield, which needs the
> term structure the desk does not yet cache (see [`results.md`](results.md)). The momentum sleeve needs
> only front-month prices (which *are* cached), so the *combine* machinery is fully exercised on the
> synthetic control below; the real blend is pre-registered and runs once a curve source is wired in via
> [`examples/verify.py`](../examples/verify.py).

## The idea

Carry and time-series momentum are the two classic commodity premia, and Koijen–Moskowitz–Pedersen–Vrugt
(2018) and Moskowitz–Ooi–Pedersen (2012) show they are **lowly correlated** — carry leans on the term
structure, momentum on the price trend, and they pay off in different regimes. The natural question after
measuring carry alone: does blending in a commodity time-series-momentum sleeve raise the combined book's
risk-adjusted return *above carry standalone*? If the legs are lowly correlated, a 50/50 risk blend should
beat the stronger leg — the textbook diversification free lunch.

## The result on the synthetic control (offline, reproducible)

A 50/50 blend of the carry book and a 26-week cross-sectional momentum sleeve, net @5 bp, on the seeded
12-commodity panel (seed 35):

| | carry | momentum | 50/50 blend |
|---|---|---|---|
| net Sharpe @5 bp | 1.80 | 1.43 | **2.03** |

- Leg-to-leg **correlation +0.27** — low, as the literature predicts.
- The blend's Sharpe **(2.03) exceeds either standalone leg** (carry 1.80, momentum 1.43): the
  diversification works exactly because the two premia are nearly orthogonal. Adding a trend sleeve to a
  carry book is a genuine improvement, not a redundant bet.

## What it means

The worked complement sharpens the verdict: commodity carry is `REAL` but `FRAGILE` (volatile,
crash-prone) — and the standard institutional fix is **not** to lever carry harder but to **diversify it**
with a lowly-correlated momentum sleeve, which lifts the combined Sharpe and softens the carry crash. This
mirrors the lesson of [Study 31 (Trade-Winds)](../../31-trade-winds/): on this desk the edge is
diversification, not prediction. The real-tape blend (pre-registered here) is expected to show the same
qualitative shape — a positive, lowly-correlated diversification gain — on the actual commodity curves.

## Forks worth a PR

- **Carry + momentum + value** — add a long-horizon mean-reversion/value sleeve (5-year reversal, Asness
  et al. 2013) for the full three-factor commodity book.
- **Vol-targeted blend** — scale the combined book to constant risk and test whether it tames the carry
  crash better than the standalone (cf. [Study 27](../../27-steamroller/), where vol-targeting failed on FX
  carry's jump).
- **Liquid-only carry** — restrict the carry leg to the deeply liquid contracts and model their honest
  costs; does the diversification gain survive when carry is weakest?
