# References & literature map — Study 20 (Freight-Train)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entry is **strategy §10.4, "Trend
  following (momentum)"**, which sets each futures position to `w_i = γ·η_i/σ_i` with
  `η_i = sign(R_i)` over a trailing window and `σ_i` the historical vol, normalised so `Σ|w_i| = 1` —
  the inverse-vol-sized time-series-momentum rule this study implements verbatim.

## The claim under test — the steelman

- **Time-series momentum.** Tobias Moskowitz, Yao Hua Ooi & Lasse Heje Pedersen, *"Time Series
  Momentum"*, **Journal of Financial Economics** 104(2), 2012. The central evidence: across 58 liquid
  futures and forwards (equity indices, bonds, currencies, commodities) and decades of data, an asset's
  own past 12-month excess return positively predicts its next-month return, and a diversified,
  vol-scaled trend portfolio earns a large, significant Sharpe with low correlation to traditional
  asset classes. The §10.4 rule is exactly their construction.

- **Trend as the core of managed futures.** Brian Hurst, Yao Hua Ooi & Lasse Pedersen, *"A Century of
  Evidence on Trend-Following Investing"* (AQR, 2017): the time-series-momentum return is robust back to
  1880, across regimes — one of the most durable systematic edges documented.

## The honest counters — why the verdict is `WEAK` / `FRAGILE` / `CONFIRMED`

- **The edge is a *cross-asset* effect.** TSMOM's Sharpe comes from diversifying many low-correlation
  timed bets; on a small, equity-heavy menu (this study's 14 ETFs — SPY/QQQ plus mostly country equity
  funds) most of the diversification is gone, so the standalone result is far thinner than the 58-future
  academic portfolio. The `WEAK` stamp is about *this menu*, not the effect.

- **Post-2009 decay.** Managed-futures trend had a celebrated 2008 and a difficult 2010s — widely
  discussed (e.g. AQR, *"You Can't Always Trend When You Want"*, 2017). The study's sub-sample Sharpe
  (+0.53 → +0.27 → −0.08) reproduces that fade on the ETF basket.

- **Why hold it anyway — crisis convexity.** William Fung & David Hsieh, *"The Risk in Hedge Fund
  Strategies: Theory and Evidence from Trend Followers"*, **Review of Financial Studies** 14(2), 2001:
  trend-following has a *long-straddle* (long-volatility) payoff, profiting in market crises. Mark
  Hutchinson & John O'Brien on "crisis alpha" extend this. `extension.crisis_convexity` measures exactly
  this — and it is the one leg that survives, hence the `CONFIRMED` third axis.

## The desk's own method — engine and reproducibility

- **HAC / Newey–West inference.** Newey & West, *Econometrica* 1987 — the pooled time-series-momentum
  *t* and the strategy-stream *t* (`trend.predictability`, `decompose.mean_tstat_hac`).
- **Reproducibility.** Headline numbers are pinned with
  [`quantlab.repro`](../../../quantlab/repro.py) (an as-of date + a content fingerprint of the basket
  returns).

## Caveats stated in the open (house rule)

- **An equity-heavy ETF menu, not cross-asset futures.** 14 retail-accessible ETFs, dominated by equity
  beta, understate the diversified trend edge by design — stated plainly, and the first beat-7 fork.
- **Split-only closes.** The signal is a months-long return sign, so dividends shift the long-run drift
  a little but not the timing; a total-return variant is a fork.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
