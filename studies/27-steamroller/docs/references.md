# References & literature map — Study 27 (Steamroller)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entry is **strategy §8.2 (carry trade)** —
  it states uncovered interest-rate parity (UIRP) and notes that, because UIRP does not always hold, a
  rate differential between two currencies is not fully offset by spot depreciation, leaving a tradable
  premium.

## The claim under test — the steelman

- **The forward-premium puzzle / carry premium.** The empirical failure of UIRP is one of the oldest in
  international finance (Eugene Fama, *"Forward and Spot Exchange Rates"*, **Journal of Monetary
  Economics** 1984). The cross-sectional carry portfolio earns a large, significant premium: Hanno Lustig
  & Adrien Verdelhan, *"The Cross-Section of Foreign Currency Risk Premia and Consumption Growth Risk"*,
  **American Economic Review** 2007; Lukas Menkhoff, Lucio Sarno, Maik Schmeling & Andreas Schrimpf,
  *"Carry Trades and Global Foreign Exchange Volatility"*, **Journal of Finance** 2012.

## The honest counter — the steamroller, and why it's `FRAGILE` / `Severe`

- **Carry crashes.** Markus Brunnermeier, Stefan Nagel & Lasse Pedersen, *"Carry Trades and Currency
  Crashes"*, **NBER Macroeconomics Annual** 2008: carry returns are negatively skewed, and the crashes
  coincide with the sudden unwinding of crowded, leveraged positions when funding liquidity dries up — a
  jump, correlated across all carry pairs at once. This is the "picking up nickels in front of a
  steamroller" tail the study measures (skew, drawdown, downside concentration).
- **Carry as a volatility / liquidity risk premium.** Menkhoff et al. (2012) show carry returns load on
  global FX volatility — i.e. the premium is compensation for doing badly exactly when volatility spikes.
  This is *why* the desk's vol-targeting overlay ([Study 16](../../16-storm-shy/); applied to momentum in
  [Study 24](../../24-stampede/)) fails here: the crash *is* the volatility spike, not a build-up that
  precedes it, so a trailing-vol estimate can lever the book *into* the jump.

## The desk's own method — engine and reproducibility

- **HAC / Newey–West inference** (Newey & West, *Econometrica* 1987) on the carry portfolio's mean, plus
  a circular-block-bootstrap Sharpe CI ([`quantlab.stats`](../../../quantlab/stats.py)).
- **Data.** The real run uses monthly **G10 3-month interbank short rates (OECD MEI, via DBnomics)** and
  **USD FX from yfinance** — the desk's shared cache, the *same tape* as
  [Study 36 (Greenback)](../../36-greenback/), fetched once by
  [`tools/fetch_altdata.py`](../../../tools/fetch_altdata.py) and pinned with
  [`quantlab.repro`](../../../quantlab/repro.py). The synthetic control bakes a partial-UIRP premium with
  a sticky two-state risk-off crash regime.

## Caveats stated in the open (house rule)

- **The sample starts in 2001 and ends 2024-01.** OECD MEI (the rates source) was discontinued at
  2024-01, and the FX history reaches back to ~2001 — so the fat carry decades of the 1980s–90s are out
  of reach of this tape. The `WEAK` signal stamp is a statement about *this* post-2000 sample; the
  literature's full-sample evidence is stronger.
- **Monthly horizon, USD base, spot (not forward) carry.** The carry signal is the rate differential;
  using spot FX plus the rate gap approximates the forward-based carry return and is the cleanly-available
  construction from public data — a stated simplification.
- **The synthetic control has zero turnover by construction** (constant per-currency rates → the ranking
  never changes), so the "cheap to run" claim is exercised only on the real tape (turnover 0.55×/yr).

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
