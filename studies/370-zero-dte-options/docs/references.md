# References & literature map — Study 370 (Zero-DTE Options & the SPX tape)

## The claim under test

- **The 0DTE boom.** Options expiring the *same trading day* ("zero-days-to-expiry") on the
  S&P 500 went from a Friday-only product to a **daily** one over 2022, as Cboe rolled out
  Tuesday/Thursday SPX expirations (April 2022) and then completed the Mon–Fri calendar. By
  2023 0DTE contracts were a large and rising share of total SPX option volume (Cboe and
  exchange volume reports widely cited in the financial press, 2023–2024). The *product* is
  not in dispute here; the *tape effect* is.
- **The pinning / mean-reversion story.** The popular thesis (financial-media and sell-side
  notes, e.g. Nomura's Charlie McElligott and JPMorgan's Marko Kolanovic commentary, 2022–2023)
  is that dealers short a wall of at-the-money, same-day gamma must hedge *with* the market on
  the way out and *against* it near expiry, **damping and reversing** intraday moves into the
  close — so the post-2022 SPX intraday tape should be **more mean-reverting / more pinned**.
  This is the believers' framing we steelman and test.
- **Academic read on 0DTE.** Early studies find 0DTE flow is large but its *destabilising*
  effect is contested: Bandi, Fusari, Renò and co-authors, and Cboe-commissioned work
  (e.g. "0DTE options and market liquidity", 2023–2024 working papers), generally find 0DTE
  trading does **not** clearly raise index volatility — consistent with our null. We cite the
  debate rather than a single verdict because the literature itself is unsettled.

## Why we can only use a daily proxy — and what we do instead

- **No free intraday option tape.** Tick-level SPX option prints, dealer gamma exposure, and
  intraday index ticks are vendor data (Cboe DataShop, OptionMetrics, etc.), not on the free
  yfinance endpoint, which serves daily OHLCV plus a *current* option-chain snapshot. We
  therefore test the claim on **explicit daily proxies** (named throughout): the lag-1
  autocorrelation of the daily **open→close** ('intraday') leg as a mean-reversion gauge, the
  same-day **range** log(high/low) as a Parkinson-style vol proxy, and a one-shot nearest-expiry
  chain snapshot as direct (current-only) 0DTE evidence. The daily proxy is a methodological
  choice, not a fabrication: every input is a public price.
- **Return-leg decomposition.** Splitting the daily return into an **overnight** (prevClose→open)
  gap and an **intraday** (open→close) leg is standard (Lou, Polk & Skouras, 2019,
  *A tug of war: Overnight versus intraday expected returns*, JFE). The intraday leg is the
  only daily object that even *touches* the within-day behaviour the 0DTE story is about.
- **Realised-range volatility.** Parkinson (1980), *The extreme value method for estimating the
  variance of the rate of return* — the high/low range as an efficient daily vol estimator,
  used here as the "pinned ⇒ compressed range" check.

## Why the cross-2022 change is inconclusive — the statistics

- **Autocorrelation and mean reversion.** A negative lag-1 autocorrelation of returns is the
  classic signature of mean reversion / "pinning"; Lo & MacKinlay (1988),
  *Stock market prices do not follow random walks*, RFS, on autocorrelation-based tests and the
  small-sample fragility of variance-ratio / autocorrelation statistics.
- **Block bootstrap for serially-dependent data.** Künsch (1989), *The jackknife and the
  bootstrap for general stationary observations*, Annals of Statistics; Politis & Romano (1994),
  *The stationary bootstrap*, JASA. We use overlapping blocks so the standard error of the
  autocorrelation estimate respects the tape's serial structure — a naive iid SE would
  overstate significance.
- **Structural-break selection / placebo test.** Placing the break at the "right" date is a
  researcher degree of freedom; testing the observed |Δ| against a distribution of random break
  dates is the honest small-sample instrument (Fisher's randomization logic; Efron & Tibshirani,
  1993, *An Introduction to the Bootstrap*). Bai & Perron (1998), *Estimating and testing linear
  models with multiple structural changes*, Econometrica, on why an a-priori break is weaker
  than a tested one — our break-date sweep shows the effect *grows* the later we fish, a known
  selection artefact.
- **Multiple testing on a famous narrative.** Harvey, Liu & Zhu (2016),
  *…and the Cross-Section of Expected Returns*, RFS, and Bailey & López de Prado (2014),
  *The Deflated Sharpe Ratio* — a single narrative-driven "the regime changed in 2022" claim,
  discovered ex-post on the proxy that happens to agree, needs a far higher bar than a naive
  *t*-stat.

## Method lineage (the desk's shared engine)

- **Block-bootstrap Welch t + placebo-break p-value.**
  [`strategy.welch_t_autocorr`](../zero_dte_options/strategy.py) and
  [`strategy.placebo_break_pvalue`](../zero_dte_options/strategy.py) — the Signal-axis tests:
  the cross-break change in lag-1 autocorrelation with block-bootstrap SEs, and a 5,000-draw
  random-break null.
- **Deterministic synthetic control.**
  [`data.synthetic_tape`](../zero_dte_options/data.py) plants a *pinning regime switch* (the
  intraday leg becomes AR(1) with coefficient −pin after a known date); the offline core runs
  with no network. The control confirms the detector finds a *real* switch and does **not**
  manufacture one when `pin = 0`.
- **Execution-lagged, cost-charged trade.**
  [`strategy.mean_reversion_trade`](../zero_dte_options/strategy.py) fades the prior intraday
  leg with a 1-day entry lag and a one-way cost × turnover — the tradability axis.

## Data sources used here

- **yfinance** daily SPY OHLC, 2010-01-05 → 2026-06-18, cached under `_cache/spy_ohlc.csv`; and
  the nearest-expiry SPY **option-chain snapshot** as-of 2026-06-22, cached under
  `_cache/spy_option_chain.csv` (+ `_cache/option_meta.json`). All headline numbers are pinned
  in [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 130 — Vol-Risk-Premium](../130-vol-risk-premium/)**: the broader question of whether
  options markets carry a harvestable signal/premium — 0DTE is the newest, fastest corner of
  that market.
- **[Study 111 — VIX-Term-Structure](../111-vix-term-structure/)**: the volatility-surface
  context the 0DTE story sits inside (very-short-dated implied vol vs the rest of the curve).
- **[Study 06 — Clockwork-Vol](../06-clockwork-vol/)**: intraday/seasonal volatility patterns —
  the same "is the within-day tape predictable?" question on a different axis.
