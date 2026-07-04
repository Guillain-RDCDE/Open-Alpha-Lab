# References — Study 615 (Yen Safe Haven)

## The claim's source

The "yen rallies when stocks crash" legend is trading-floor folklore with a real academic
spine: the yen is the classic **funding currency** of the global carry trade (borrow at
~0% in Tokyo, buy anything that yields), and risk-off episodes force carry books to buy
back borrowed yen. Every carry-unwind headline — Oct-1998 (USDJPY −15% in two days),
Lehman 2008, Aug-2015, Feb/Mar-2020, Aug-2024 — reinforced it. The sharpest recent
statement of the *failure* mode is 2022 commentary: the yen fell ~20% against the dollar
in the middle of a −25% equity bear.

## Key papers

- **Ranaldo, A. & Söderlind, P. (2010)** — *Safe Haven Currencies*, Review of Finance
  14(3). JPY (and CHF) appreciate when US equities fall and FX volatility rises — the
  canonical conditional-beta result this study replicates.
  <https://doi.org/10.1093/rof/rfq007>
- **Brunnermeier, M., Nagel, S. & Pedersen, L. (2008)** — *Carry Trades and Currency
  Crashes*, NBER Macro Annual 23. Carry-trade positioning creates crash risk in the
  funding currency's favor: negative skew for high-carry targets, unwind spikes for JPY.
  <https://www.nber.org/papers/w14473>
- **Habib, M. & Stracca, L. (2012)** — *Getting Beyond Carry Trade: What Makes a Safe
  Haven Currency?*, Journal of International Economics 87(1). Safe-haven status comes
  from net foreign asset position more than the rate differential — consistent with our
  finding that the daily hedge survives ZIRP regimes.
  <https://doi.org/10.1016/j.jinteco.2011.12.005>
- **BIS (2020)** — *The dollar funding squeeze*, BIS Bulletin No. 2 (Apr-2020) — the
  Mar-2020 second-leg failure: everything, including JPY, fell against the scrambling
  dollar. <https://www.bis.org/publ/bisbull02.htm>
- **BIS (2024)** — *The market turbulence and carry trade unwind of August 2024*, BIS
  Bulletin No. 90 — the Aug-2024 episode used in the event table.
  <https://www.bis.org/publ/bisbull90.htm>
- **Erb, C. & Harvey, C. (2013)** — *The Golden Dilemma*, FAJ 69(4) — the gold sibling's
  frame; cited here for the "safe haven vs insurance bill" trade-off.
  <https://doi.org/10.2469/faj.v69.n4.1>

## Named siblings on this desk (dedup guard)

- [69-safe-haven](../../69-safe-haven/) — **gold** as the crash hedge / inflation hedge.
  Same folk question ("what protects me when stocks crash?"), different asset and
  different mechanism: gold is a store-of-value story; the yen is a **positioning**
  story (carry unwind). The two studies also fail differently — gold is a coin-flip in
  crashes but carries no bill; the yen hedges *days* decisively yet charges ~4.7%/yr of
  negative carry and loses in rate-shock bears.
- [613-currency-hedged-etf-carry](../../613-currency-hedged-etf-carry/) — the same
  USD-JPY rate differential seen from the *hedger's* side (earning the carry the yen
  sleeve pays here).

## Data sources

- **Yahoo Finance via `yfinance`** — `JPY=X` (USD/JPY spot; the yen's return is minus
  the USDJPY change; price-only), `SPY` (total-return, auto-adjusted), `^IRX` (13-week
  US T-bill discount yield, the carry / foregone-bill proxy; Japanese short rates ~0
  for nearly the whole sample), `^TNX` (US 10-year yield).
  <https://finance.yahoo.com/quote/JPY=X/> · <https://finance.yahoo.com/quote/SPY/>
- Event window dates hardcoded in [`yen_safe_haven/data.py`](../yen_safe_haven/data.py)
  with per-episode source comments (BIS bulletins, SPY peak/trough from the tape).

## Shared method citations

- **Newey, W. & West, K. (1987)** — HAC standard errors (daily conditional betas).
- **Welch, B. L. (1947)** — unequal-variance t (quintile and regime group splits).
- House rules: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar, the
  synthetic-control discipline, excess-vs-excess races, one documented execution
  convention.
