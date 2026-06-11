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

- **Two account types, each charging every dollar exactly once.** A *margin account* borrows only the
  slice above 100% at the bill + markup and parks idle capital in bills; a *futures/CFD account* carries
  its **entire notional** at the bill + markup while the capital itself stays in bills (the cost-of-carry
  baked into a future's or swap's pricing is textbook: John Hull, *Options, Futures, and Other
  Derivatives*). At a zero markup the two are algebraically identical; what separates them is only **where
  the broker's markup lands** — the borrowed slice, or the whole position. (Beware the half-and-half
  accounting that charges full-notional financing *and* credits cash interest only on the un-deployed
  fraction: it pays the risk-free rate twice and manufactures a phantom ~`avg_exposure × rf` pts/yr drag.
  An earlier version of this study did exactly that.)
- **Sharpe in excess of the bill, both sides.** A financed strategy's net return already has funding in
  it; quoting buy-and-hold's Sharpe on raw returns hands it the risk-free rate for free. Every Sharpe in
  this study is computed on returns minus the T-bill, strategy and benchmark alike.
- **Volatility drag and the gate's opportunity cost.** Levering a noisy series compounds a volatility
  penalty (≈ ½·(k²−k)·σ² for constant leverage *k*), and time flattened by the trend gate forgoes the
  equity premium — which is why even at a zero markup the book *ties* rather than beats the index
  (−0.5 pts/yr in [`results.md`](results.md)). The return edge never existed; the markup turns a tie into
  a structural loss.
- **Risk management here is cheap insurance, not alpha.** A margin account pays ~0.7 pts/yr of markup
  (plus the 0.5-pt structural shortfall) for a 31-point cut in max-drawdown and a doubled Calmar (0.38 vs
  0.19). Worth having for survival and leverage headroom — but it out-earns nothing, and routed through a
  CFD the same insurance costs 2.65 pts/yr because the markup hits the full notional.

## The desk's own method — engine and reproducibility

- **Data.** Real run uses **Yahoo `^GSPC`** (auto-adjusted, a total-return proxy) and **`^IRX`** (13-week
  T-bill) as the financing rate, daily, 1990–2026. Pinned with [`quantlab.repro`](../../../quantlab/repro.py)
  (as-of date + input fingerprint). The offline control is a seeded synthetic GARCH(1,1)-with-bear-regimes
  index (`house_edge.data.synthetic_market`).
- **Cost models.** `house_edge.costs.net_returns(mode="margin" | "futures")` — the two account types,
  applied to one exposure path so only the funding (and where the markup lands) varies.

## Caveats stated in the open (house rule)

- **`^GSPC` auto-adjusted ≈ total return, not exact.** Yahoo's adjustment approximates dividend
  reinvestment; the honest model also credits an explicit constant dividend yield to the long, so the
  comparison is like-for-like on the total-return basis (stated, not hidden).
- **One strategy parameterisation.** The headline uses `target_vol = 0.15`, `lev_cap = 2.0`; the markup
  sweep and the synthetic control show the conclusion is not specific to one fee assumption: the CAGR edge
  vs buy-and-hold is negative in every row of the sweep, and the gap widens with the markup — fastest in
  the CFD, where the markup hits the full notional. Other parameterisations move the numbers, not the sign.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
