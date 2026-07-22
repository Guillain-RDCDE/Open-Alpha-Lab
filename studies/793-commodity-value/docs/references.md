# References & literature map — Study 793 (Cross-sectional commodity value)

## The claim under test

- **The anchor paper.** Cliff **Asness, Tobias Moskowitz & Lasse Pedersen (2013)**,
  *"Value and momentum everywhere"*, *Journal of Finance* 68(3). They define a **value**
  signal in *every* asset class, and for commodities it is a **long-horizon reversal**: the
  log of the average spot price ~5 years ago (their 4.5–5.5-years-ago reference) minus the
  log of the current spot price. A commodity whose price has *fallen* over five years is
  **cheap**; buy the cheap third, short the expensive third, rebalance monthly. This study
  rebuilds that exact commodity value leg on a modern, *investable* proxy — a basket of
  liquid single-commodity ETFs.
- **The mechanism.** Long-horizon **reversal / mean reversion** in asset prices is the old
  De Bondt & Thaler (1985) "overreaction" idea (equities: past 3–5-year losers beat past
  winners). AMP's commodity value is the same 5-year reversal, and it is the deliberate
  *opposite horizon* to short-run **momentum** (past-12-month continuation) — the two are
  negatively correlated, which is why AMP combine them.
- **The commodity-value literature.** Asness, Moskowitz & Pedersen document a positive
  commodity value premium in a broad futures cross-section; subsequent work (e.g. Bhardwaj,
  Gorton & Rouwenhorst on commodity factors; Blitz & de Groot 2014 on commodity allocation)
  finds commodity value **weaker and less robust than momentum**, sensitive to the exact
  basket, the spot-vs-total-return choice, and the sample. That fragility is exactly what a
  thin, investable ETF proxy would surface.
- **The open question we test.** Does the AMP commodity value leg — long the 5-year-fallen,
  short the 5-year-risen — still pay on a *free, investable, modern* single-commodity-ETF
  tape (2012–2026), net of realistic costs, or does the investable reading collapse?

## What we measure, and the honesty rails

- **Decisive statistic: HAC/Newey-West t** on the monthly L/S mean (Newey-West 1987;
  monthly factor returns are serially correlated, so an i.i.d. SE would overstate
  significance). A one-sample t and an annualised Sharpe accompany it. `REAL` requires
  HAC t ≥ 2 **on this tape**; strong literature support with a sub-2 t reads at best
  **`WEAK`** — and here the tape gives t = 0.17, i.e. `NONE`.
- **Signal on the price level, not total return.** Value is a *price-level* reversal claim,
  so the 5-year ratio is measured on the **raw close** (`auto_adjust=False`); the total-
  return (adjusted) close is used only for the P&L a held position earns. Folding five years
  of carry/roll into the signal would be a look-alike, not the AMP construction.
- **One documented execution lag.** Weights formed at the close of month *t* earn month
  *t*+1 — a single `shift`, applied once. Zero look-ahead: the 5-year signal uses only data
  through month *t*.
- **Costs are one-way × traded notional**, the short leg pays **borrow** (50 bps/yr on 1×
  NAV). Gross is labelled gross, net is labelled net, in every table.
- **Survivorship is named on the Signal axis.** The ETF basket is current membership; a fund
  that fell far enough to delist is a "deep value" name absent from the cheap/long leg,
  biasing the premium *upward* — the numbers are an **upper bound**. **Roll contamination is
  also named:** chronic-contango energy ETFs read as perennially "cheap" for a mechanical
  reason, which the ETF proxy cannot fully separate from genuine spot value.
- **A random-rank placebo (40 seeds)** replaces the value signal with noise ranks and asks
  how often that matches the real HAC t; the sub-period split is tested as a **difference**
  (Welch t), not eyeballed; the synthetic positive control plants a known reversal edge and
  confirms the sort recovers it while the null does not systematically fire.

## Data sources

- **13 liquid single-commodity ETFs** (the same basket as sibling 792): GLD, SLV, PPLT,
  PALL, CPER (metals); USO, UNG, UGA (energy); CORN, WEAT, SOYB, CANE, DBA (ags). Cached
  under `_cache/` (`cv_etf_daily.parquet`) with both a raw-close series (signal) and an
  adjusted-close series (P&L).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py) (fingerprint `0524773df8ac`).

## Related desk studies (the dedup map — what this study is NOT)

- [792-commodity-momentum](../../792-commodity-momentum/) — the **opposite horizon** on the
  *same 13-ETF basket*: 12-1 momentum (past-year *continuation*, Miffre-Rallis 2007), which
  reads `WEAK` (HAC t ≈ 1.66). Value here is the **5-year reversal** — mechanically the
  mirror image (and empirically negatively correlated with momentum). Same universe,
  opposite signal.
- [638-value-momentum-everywhere](../../638-value-momentum-everywhere/) — the **mixed
  multi-asset** value+momentum *combo* across equities, FX, bonds AND commodities (the full
  AMP diversification claim). Commodities are one sleeve there, blended *with* momentum and
  *with* other asset classes; here we isolate the **commodity value leg alone**, with no
  momentum blend and no cross-asset diversification.
- The single-commodity **seasonality** studies (226, 307, 648-651) test **calendar** effects
  within *one* commodity (a time-series claim). This study is **cross-sectional** (relative
  cheapness *across* commodities) and carries no calendar hook.

None of the siblings isolate the pure cross-sectional commodity **value** leg on an
investable ETF basket — which is this study's own axis.
