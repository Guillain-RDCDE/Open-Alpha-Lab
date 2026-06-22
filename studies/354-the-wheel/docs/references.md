# References & literature map — Study 354 (The Wheel)

## The claim under test

- **The retail "Wheel" pitch.** A staple of options-income communities (r/thetagang,
  r/options, countless YouTube/Substack "passive income" videos): *sell a cash-secured
  at-the-money **put** on an index ETF (SPY/QQQ); if assigned, you now own the shares, so sell
  at-the-money covered **calls** until they are called away; repeat forever and collect premium
  as "income" regardless of market direction.* The framing is that you are paid "theta" /
  "rent" on your capital — a market-neutral money machine. The testable claims: (H₁) the Wheel
  generates a real return *advantage* (the "income") over simply holding the index; (H₂) it is
  lower-risk / market-neutral; (H₃) it is robust enough to "set and forget."
- **What it actually is.** A short cash-secured put plus a short covered call are, leg by leg,
  **short volatility**: you are short a straddle/strangle rolled monthly. The Wheel is therefore
  a packaged **short-vol / covered-call** exposure — the question is only whether its premium
  out-earns the upside it surrenders, and at what tail cost.

## The option-pricing model (transparent, clearly labelled)

- **Black-Scholes (1973), *The Pricing of Options and Corporate Liabilities*** (JPE) and
  **Merton (1973), *Theory of Rational Option Pricing*** (Bell J. Econ.). We price the
  one-month at-the-money call and put with the closed-form BS formula. With strike = spot and
  rate r = 0 the ATM call and put are equal and reduce to **2Φ(σ√t/2) − 1 ≈ 0.4·σ·√t** — used
  as the deterministic positive-control identity (`strategy.fair_premium_identity`). We feed
  **VIX/100 as the implied vol** — a transparent proxy for the at-the-money one-month IV, *not*
  a live option chain. This is stated as a model choice on the axes.
- **VIX construction.** CBOE, *The VIX White Paper* (2003/2019). VIX is the 30-day risk-neutral
  implied vol of SPX options; using it as the ATM IV is a clean, public, reproducible stand-in.

## Why the premium is the variance risk premium, not free income

- **The variance risk premium (VRP).** Carr & Wu (2009), *Variance Risk Premiums* (RFS);
  Bollerslev, Tauchen & Zhou (2009), *Expected Stock Returns and Variance Risk Premia* (RFS).
  Implied vol (VIX) runs persistently **above** realised vol — option *sellers* are paid for
  bearing variance risk. Our model bakes this in (VIX ~19.6% vs ~14.9% realised over the
  sample), so the Wheel's apparent edge **is** the VRP. At a *fair* IV (the synthetic control,
  `vix_premium = 1`) the Wheel **underperforms** buy-and-hold — proving the mechanic itself has
  no edge.
- **Covered-call "income" is gains given back.** Israelov & Nielsen (2015), *Covered Calls
  Uncovered* (FAJ); Whaley (2002), *Return and Risk of CBOE Buy-Write Monthly Index* (BXM).
  Writing calls truncates the upside; the fat distribution feels like income but is largely the
  asset's own appreciation handed back. The BXM/PUT index literature (CBOE) shows buy-write and
  put-write returns track the index with **lower** vol and **worse** crash skew — exactly the
  short-vol payoff.
- **Put-writing = covered-call by put-call parity.** Selling a cash-secured ATM put has the
  **same** payoff as a covered call on the same strike (put-call parity), so the Wheel's two
  legs are economically one short-vol position rolled monthly — not a clever directional switch.

## Why high Sharpe ≠ harvestable — the short-vol tail

- **Short-vol skew and the steamroller.** Selling options is "picking up pennies in front of a
  steamroller": small steady premium, rare catastrophic loss, strongly **negative skew**. Our
  Wheel shows skew −1.87 (vs SPY −0.36) and a fat left tail (worst-5 months = −51%), the
  signature decomposed in **[Study 63 — Free-Fall](../../63-free-fall/)** (naive short-vol blows
  up) and **[Study 62 — Premium-Seller](../../62-premium-seller/)** (covered-call ETFs trail
  their own index on total return).
- **Costs kill thin edges.** The bid/ask on monthly ATM index options, paid on every write and
  every assignment/roll, is where the gross edge dies; our cost sweep flips the edge negative at
  ~25 bp/option. Charge costs against the *alpha*, not the gross (house rule).
- **Regime dependence / data-snooping.** The Wheel's edge over buy-and-hold flips sign across
  the two halves of the sample (+6.2%/yr then −1.2%/yr); a sign-flipping, sub-2-t effect is
  `MIXED`, not `REAL` (METHODOLOGY verdict rubric).

## Method lineage (the desk's shared engine)

- **Paired-difference inference.** The Signal axis is a paired *t* of the Wheel's monthly excess
  return over buy-and-hold (`strategy.paired_t`); the `REAL` bar is a robust t ≥ 2 on the real
  tape, not literature support.
- **Deterministic synthetic control.** A fixed-seed lognormal-with-jumps SPY world
  ([`data.synthetic_months`](../the_wheel/data.py)) priced at its **own** vol is the positive
  control that proves the engine measures the option payoff (and that the *fair* price gives a
  negative edge to the writer on a drifting index). Runs with no network.
- **Black-Scholes closed form as the machinery proof.** `strategy.fair_premium_identity`
  asserts the engine's ATM premium equals `2Φ(σ√t/2) − 1` to machine precision — exact, not
  fitted.

## Data sources used here

- **yfinance**: SPY total-return adjusted close and ^VIX level, 1993–2026, cached under
  `_cache/` (no key). All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 62 — Premium-Seller](../../62-premium-seller/)**: covered-call "income" ETFs (QYLD)
  trailing their own underlying — the upside/downside capture asymmetry, on a real fund.
- **[Study 63 — Free-Fall](../../63-free-fall/)**: the naive short-vol carry (SVXY) and its
  −95% crash — the steamroller this study's −1.87 skew gestures at.
- **[Study 351 — BTC 5m Polymarket](../../351-btc-5m-polymarket-momentum/)** and
  **[Study 301 — Triple-RSI](../../301-triple-rsi/)**: the high-win-rate / negative-tail shape,
  in other venues.
