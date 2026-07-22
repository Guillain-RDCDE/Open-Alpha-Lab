# References & literature map — Study 796 (Corporate-Bond-Low-Risk)

## The claim under test

- **The source paper.** Frazzini & Pedersen (2014), *"Betting Against Beta"*, **Journal of
  Financial Economics** 111(1). Because many investors are leverage-constrained, they bid up
  high-beta (risky) assets and shun low-beta (safe) ones, so the safe assets earn higher
  *risk-adjusted* returns. A "BAB" factor — long low-beta assets levered up to beta 1, short
  high-beta assets levered down to beta 1 — earns a large, significant premium across US and
  international equities **and across asset classes, including Treasuries and credit**. The
  bond version of the claim: shorter-duration / higher-grade / lower-volatility exposures
  deliver a higher Sharpe than long-duration / junk / high-volatility ones.
- **The wider family.** The low-risk anomaly is one of finance's most-replicated facts:
  low-volatility and low-beta stocks earn higher Sharpe ratios than high-risk ones (Black,
  Jensen & Scholes 1972; Haugen & Heins 1975; Baker, Bradley & Wurgler 2011, *"Benchmarks as
  Limits to Arbitrage"*, FAJ; Ang, Hodrick, Xing & Zhang 2006 on idiosyncratic volatility).
  Leverage aversion and benchmarking frictions are the leading explanations.
- **What we test, and its honest limit.** We take the claim to a **tradable bond-ETF panel**:
  rank an 11-name credit + Treasury bond-ETF basket on trailing *volatility* (the risk proxy
  the claim itself names when beta is unavailable), then build a vol-scaled low-minus-high
  spread. This is the *retail-accessible* version — but it is a far **coarser instrument** than
  the single-name cross-section Frazzini-Pedersen use: a dozen broad ETFs cannot reproduce the
  within-sleeve risk dispersion the anomaly lives in, and the calmest names (short-duration
  Treasuries) are poor things to lever. A null on this panel is therefore consistent with — not
  a refutation of — the equity/single-name evidence.

## What we measure, and the honesty rails

- **The vol-scaled low-minus-high (BAB) spread.** Each month-end, rank on trailing 1-year
  daily-return volatility; long the bottom-third (low-vol) leg and short the top-third (high-vol)
  leg, each **scaled to a common ex-ante risk target** (the leg's own trailing vol gives the
  leverage), formed on the month-*t* close and earning month *t+1* (one documented execution
  `shift`, no same-bar fill; the scaling uses ex-ante vol, known at *t*). The inference-bar
  number is a **Newey-West (HAC) one-sample *t*** on the monthly spread mean; we also report the
  plain *t*, Sharpe, a Wilson interval on the hit rate, and the max drawdown, plus the
  descriptive low-vs-high **basket Sharpe** the claim is really about.
- **The vol-rank-shuffle placebo.** Keep each month's realised return cross-section and the
  vector of trailing vols exactly, but randomly permute which asset carries which volatility,
  2,000 times — destroying the low-vol→return link while preserving both marginals and leg
  sizes. The one-sided share of placebos beating the real mean is the permutation *p*. (Here it
  is damning: a random ranking earns *more* than the real one.)
- **Robustness rails.** The vol-estimation window is swept (63d / 126d / 252d); a **2022 myth-
  check** drops the year long-duration bonds crashed (when the safe leg would have shone); the
  sample is split into 2008-2015 and 2016-2026; a cost sweep charges one-way turnover × NAV plus
  a single financing/borrow rate on the borrowed (levered) notional, and reports the break-even.
- **Survivorship** is named on the **Signal** axis: the basket is the current-membership ETF
  list projected backwards. For the BAB direction the bias would *understate* the risky leg's
  tail and thus *understate* the low-risk premium, so the null we find is not manufactured by it.

## Data sources

- **Bond-ETF total-return prices** — yfinance (no key), auto-adjusted close (coupons folded in),
  cached under `_cache/bondlowrisk_prices.parquet`, 2007-01-03 → 2026-06-30 (the same tape as
  Study 795, so the low-risk leg and the momentum leg are graded on identical data).
- **The synthetic positive control** — a deterministic, seeded total-return panel with a planted
  low-risk (Sharpe-tilt) knob (null at 0); no network. It proves the BAB engine recovers a real
  planted low-risk premium and scores the null at zero.
- All headline numbers are pinned in [`docs/results.md`](results.md) (fingerprint `1f2efa58efab`)
  and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [238-betting-against-beta](../../238-betting-against-beta/) — **BAB in equities**: single
  stocks sorted on market **beta**, the canonical Frazzini-Pedersen construction where the
  premium is real. This study is the **bond** leg, ranking bond ETFs on **volatility** — a
  different asset class and a coarser instrument.
- [330-low-volatility-anomaly](../../330-low-volatility-anomaly/) — the **low-volatility anomaly
  in stocks** (low-vol equities earn a higher Sharpe). Same *idea* as ours, but the equity
  cross-section, where the effect survives. We ask whether it carries to a bond-ETF basket (it
  does not, tradably).
- [795-corporate-bond-momentum](../../795-corporate-bond-momentum/) — the **momentum** leg on
  the **same 11-ETF bond tape**: a trailing-*return* rank (a *change* signal). This study is the
  **risk-level** leg: a trailing-*volatility* rank scaled into a risk-matched spread. Same
  universe, orthogonal signal — both come up empty on the ETF panel for the same coarse-
  instrument reason.

None of the siblings test a **trailing-volatility low-risk sort on a bond universe** — that is
this study's own axis.
