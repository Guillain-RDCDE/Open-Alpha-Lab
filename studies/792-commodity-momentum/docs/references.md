# References & literature map — Study 792 (Cross-sectional commodity momentum)

## The claim under test

- **The anchor paper.** Joëlle **Miffre & Georgios Rallis (2007)**, *"Momentum
  strategies in commodity futures markets"*, *Journal of Banking & Finance* 31(6). Ranking
  a cross-section of commodity futures on their **past 12-month performance** and going
  **long the top third / short the bottom third**, rebalanced monthly, earned a
  significant premium (~9-13%/yr over 1979-2004) — a commodity analogue of the
  Jegadeesh-Titman equity momentum factor. This study rebuilds that exact sort on a modern,
  *investable* proxy: a basket of liquid single-commodity ETFs.
- **The equity ancestor.** Narasimhan **Jegadeesh & Sheridan Titman (1993)**, *"Returns to
  buying winners and selling losers"*, *Journal of Finance* 48(1) — the canonical **12-1**
  construction (rank on months *t*−11 … *t*−1, skipping the most recent month to dodge
  short-term reversal). We use their signal definition verbatim, applied across commodities
  rather than stocks.
- **The multi-asset umbrella.** Asness, Moskowitz & Pedersen (2013), *"Value and momentum
  everywhere"*, *Journal of Finance* — documents momentum in commodities as one of eight
  sleeves. Bakshi, Gao & Rossi (2019) and others note the **post-2008 decay** of many
  commodity-factor premia, which is exactly what our pre/post-2019 split surfaces.
- **The open question we test.** Does the Miffre-Rallis cross-sectional sort still pay on a
  *free, investable, modern* commodity-ETF tape (2009-2026), net of realistic costs — and
  is it certifiable, or has it faded into the recent decade?

## What we measure, and the honesty rails

- **Decisive statistic: HAC/Newey-West t** on the monthly L/S mean (Newey-West 1987;
  monthly factor returns are serially correlated, so an i.i.d. SE would overstate
  significance). A one-sample t and an annualised Sharpe accompany it. `REAL` requires
  HAC t ≥ 2 **on this tape**; a sub-2 t with strong literature support reads **`WEAK`**.
- **One documented execution lag.** Weights formed at the close of month *t* earn month
  *t*+1 — a single `shift`, applied once. Zero look-ahead: the 12-1 signal uses only data
  through month *t*.
- **Costs are one-way × traded notional**, the short leg pays **borrow** (50 bps/yr on 1×
  NAV). Gross is labelled gross, net is labelled net, in every table.
- **Survivorship is named on the Signal axis.** The ETF basket is current membership; a
  fund that trended to closure is absent from the loser leg, biasing the premium *upward* —
  the momentum numbers are an **upper bound**, and the caveat travels with the stamp.
- **A random-rank placebo (40 seeds)** replaces the momentum signal with noise ranks and
  asks how often that matches the real HAC t; the sub-period split is tested as a
  **difference** (Welch t), not eyeballed; the synthetic positive control plants a known
  momentum edge and confirms the sort recovers it while the null does not systematically
  fire.

## Data sources

- **13 liquid single-commodity ETFs**, total-return closes (yfinance `auto_adjust=True`,
  distributions folded in), cached under `_cache/` (`cm_etf_daily.parquet`): GLD, SLV,
  PPLT, PALL, CPER (metals); USO, UNG, UGA (energy); CORN, WEAT, SOYB, CANE, DBA (ags).
- Monthly returns are the month-end-to-month-end pct-change of the adjusted closes. The
  ETF proxy is the *investable* reading of the futures claim — a real portfolio holds
  funds, and each fund embodies the roll of the underlying front-month futures.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [638-value-momentum-everywhere](../../638-value-momentum-everywhere/) — the **mixed
  multi-asset** value+momentum *combo* across equities, FX, bonds AND commodities, testing
  the AMP diversification claim. Commodities are one sleeve there; here they are the whole
  cross-section, and there is **no value leg and no cross-asset blend** — a pure commodity
  momentum sleeve, measured on its own.
- [507-cross-sectional-momentum](../../507-cross-sectional-momentum/) — the identical 12-1
  sort but on **equities** (a large-cap survivor basket; verdict `NONE`). Same signal,
  entirely different asset class — commodity momentum has different economic drivers (roll
  yield, storage, supply shocks) than equity momentum.
- The single-commodity **seasonality** studies —
  [226-crude-seasonality](../../226-crude-seasonality/),
  [307-coffee-seasonality](../../307-coffee-seasonality/),
  [648-grain-seasonality](../../648-grain-seasonality/),
  [649-gold-seasonality](../../649-gold-seasonality/),
  [650-heating-oil-seasonality](../../650-heating-oil-seasonality/),
  [651-sugar-seasonality](../../651-sugar-seasonality/) — test **calendar** effects within
  *one* commodity (a time-series, single-name claim). This study is **cross-sectional**
  (relative strength *across* commodities) and carries **no calendar hook** at all.

None of the siblings test the pure cross-sectional commodity-momentum sort on an
investable ETF basket — which is this study's own axis.
