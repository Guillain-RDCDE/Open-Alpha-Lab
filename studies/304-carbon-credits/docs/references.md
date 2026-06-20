# References & literature map — Study 304 (Carbon-Credits)

## The claim under test

- **"Buy and hold carbon credits — the cap only shrinks."** The marketing case for carbon
  as a one-way buy-and-hold: under cap-and-trade (the EU ETS, California, RGGI), the
  regulator retires allowances every year, so a fixed and falling supply meets inelastic
  demand from polluters — a structurally rising price. KraneShares makes this case directly
  for **KRBN** (KraneShares Global Carbon Strategy ETF), *"the first US-listed ETF designed
  to track carbon allowance markets"* (KraneShares fund literature and the *Carbon Market
  Outlook*). The implicit hypothesis: $\mathbb{E}[\text{KRBN excess-of-cash return}] > 0$
  and large enough to reward a passive hold. We test that directly, then ask whether a
  simple trend-timing overlay would have done better than naively holding.

## Why carbon allowances might (or might not) drift up

- **Cap-and-trade design.** Tietenberg (2006), *Emissions Trading: Principles and
  Practice* (RFF Press). The EU ETS Market Stability Reserve (Directive (EU) 2018/410) was
  built precisely to tighten supply and lift the price floor — the regulatory tailwind the
  buy-and-hold case leans on.
- **Carbon as a commodity, not a bond.** Empirical work finds EU allowance (EUA) prices are
  driven by fuel prices, weather, industrial output and policy headlines — i.e. they behave
  like a **volatile commodity**, not a smoothly compounding asset. Chevallier (2009),
  *Carbon futures and macroeconomic risk factors* (Energy Economics); Hintermann (2010),
  *Allowance price drivers in the first phase of the EU ETS* (J. Environmental Economics &
  Management). The 2021 doubling and the 2022/2024 declines on this tape are exactly that
  commodity behaviour.

## The timing overlay — time-series (absolute) momentum

- **Time-series momentum.** Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*
  (Journal of Financial Economics) — a trailing-return sign rule that has historically added
  value across futures including commodities. We apply the canonical "hold while trailing
  return > 0, else cash" form ([`strategy.momentum_signal`](../carbon_credits/strategy.py)),
  which is the natural steelman for "you should have timed it, not just held it."
- **Absolute (dual) momentum.** Antonacci (2014), *Dual Momentum Investing* (McGraw-Hill) —
  the popular retail framing of the same idea: stay in the risky asset only while its
  absolute momentum is positive. The overlay tested here is its single-asset case.
- **Trend-following on a single volatile asset.** The known failure mode — whipsaw in a
  choppy, mean-reverting tape — is exactly what we observe: the overlay flips out of the
  2021 melt-up and underperforms buy-and-hold at every lookback.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../carbon_credits/strategy.py), used for the excess-of-cash mean.
- **Circular block bootstrap.** Politis & Romano (1992), *A circular block-resampling
  procedure for stationary data* — the Sharpe-difference CI in
  [`strategy.bootstrap_sharpe_diff`](../carbon_credits/strategy.py); i.i.d. resampling would
  destroy the volatility clustering the inference must respect.
- **Excess-vs-excess Sharpe races.** When one arm sits in cash part-time, the fair
  comparison is excess-of-cash to excess-of-cash (a raw-vs-excess race manufactures
  verdicts) — the convention the engine enforces.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`, `auto_adjust=True` ⇒ total return). KRBN
  (carbon allowances) and BIL (1-3m T-bills, the cash proxy), joint window from KRBN's
  2020-07-30 inception. All headline numbers are pinned with an as-of date and content
  fingerprint (see [`docs/results.md`](results.md)). The offline reproducible core and the
  test-suite run on the deterministic [`data.synthetic_carbon`](../carbon_credits/data.py)
  generator, never the network.

## Related desk studies

- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: the same excess-vs-excess Sharpe
  race + block-bootstrap machinery, applied to the 60/40 blend. Sister study on the
  buy-and-hold-vs-alternative question.
- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)**: another "just buy
  and hold this allocation" claim taken literally — the family this study belongs to.
- **[Study 152 — Inflation-Hedge](../../152-inflation-hedge/)**: a real-asset buy-and-hold
  thesis (gold/commodities as a hedge) tested on a single-cycle tape — the same
  short-sample, commodity-beta caution applies here.
