# References & literature map — Study 30 (House-Edge)

## The claim under test — the steelman

The pitch this study takes apart is the standard one behind leveraged-ETF and CFD timing products: *a
good timing model plus leverage beats buy-and-hold.* Its respectable components are real and well-studied:

- **Volatility targeting / risk parity.** Scaling exposure inversely to realised volatility raises
  risk-adjusted returns and cuts tail risk in many studies — e.g. Alan Moreira & Tyler Muir, *"Volatility-
  Managed Portfolios"*, **Journal of Finance** 72(4), 2017; Tony Cooper, *"Alpha Generation and Risk
  Smoothing using Managed Volatility"* (2010). The mechanism in this study (cut leverage when vol spikes)
  is exactly theirs.
- **Short-term mean reversion / dip-buying.** The contrarian RSI(2) entry is the documented short-horizon
  reversal in equity indices — Larry Connors & Cesar Alvarez, *Short Term Trading Strategies That Work*
  (2008); the academic reversal is Bruce Lehmann, *"Fads, Martingales, and Market Efficiency"*, **QJE**
  1990, and Narasimhan Jegadeesh, *"Evidence of Predictable Behavior of Security Returns"*, **Journal of
  Finance** 1990.
- **Trend filters.** Flattening below a long moving average to avoid sustained bear markets — Meb Faber,
  *"A Quantitative Approach to Tactical Asset Allocation"*, **Journal of Wealth Management** 2007.

Each ingredient is real. The question this study asks is whether, *assembled and levered and charged its
true financing cost*, the package beats simply holding the index it trades.

## The honest counter — why the verdict is `REAL` / `MIRAGE` / `Busted`

- **Financing is charged on the whole notional, not the excess.** A levered or CFD/futures position funds
  its *entire* exposure at a money-market rate plus a markup, every day it is held — not merely the slice
  above 100%. The convention of financing only `max(exposure − 1, 0)` (and ignoring the dividend a long
  index position receives) is the single accounting choice that turns a losing levered backtest into a
  winning-looking one. The cost-of-carry that makes this unavoidable is textbook: John Hull, *Options,
  Futures, and Other Derivatives* (futures/forward pricing, the cost-of-carry relation).
- **Volatility drag and the leverage tax.** Constant leverage compounds a volatility penalty
  (≈ ½·(k²−k)·σ² for leverage *k*); levering a noisy series amplifies variance faster than mean. This is
  why even a zero-markup funding rate leaves the levered book trailing the index (the sweep in
  [`results.md`](results.md)).
- **Risk management trades return for drawdown, ~1:1 here.** The identical Calmar (0.19) of strategy and
  buy-and-hold is the crisp statement: per unit of max-drawdown survived, neither is ahead. Drawdown
  protection is real and worth having (survival, leverage headroom), but it is *not* a free return edge.

## The desk's own method — engine and reproducibility

- **Data.** Real run uses **Yahoo `^GSPC`** (auto-adjusted, a total-return proxy) and **`^IRX`** (13-week
  T-bill) as the financing rate, daily, 1990–2026. Pinned with [`quantlab.repro`](../../../quantlab/repro.py)
  (as-of date + input fingerprint). The offline control is a seeded synthetic GARCH(1,1)-with-bear-regimes
  index (`house_edge.data.synthetic_market`).
- **Cost models.** `house_edge.costs.net_returns(mode="idealized" | "honest")` — the two accountings,
  applied to one exposure path so only the cost model varies.

## Caveats stated in the open (house rule)

- **`^GSPC` auto-adjusted ≈ total return, not exact.** Yahoo's adjustment approximates dividend
  reinvestment; the honest model also credits an explicit constant dividend yield to the long, so the
  comparison is like-for-like on the total-return basis (stated, not hidden).
- **One strategy parameterisation.** The headline uses `target_vol = 0.15`, `lev_cap = 2.0`; the financing
  sweep and the synthetic control show the conclusion is not specific to the markup, and the verdict is a
  structural one (volatility drag + full-notional carry), not a tuning artefact. Other parameterisations
  move the numbers, not the sign.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
