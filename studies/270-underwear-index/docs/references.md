# References & literature map — Study 270 (Underwear-Index)

## The claim under test

The **Men's Underwear Index (MUI)**, popularly attributed to **Alan Greenspan**:
men treat underwear as a near-necessity replaced on a steady schedule, so a *dip*
in men's-underwear sales is supposed to flag household stress and an oncoming
recession. The anecdote circulated through Greenspan's Fed years and resurfaced
in mainstream coverage during 2008 (e.g., NPR *Planet Money*, "The Men's Underwear
Index", and numerous trade-press write-ups citing Mintel / NPD Group point-of-sale
commentary). It is folklore: there is no peer-reviewed paper establishing the MUI
as a leading indicator, and the mechanism (deferring a cheap, hidden staple) is
plausible but unquantified.

## Why the index looks compelling — and why that is misleading

- **Coincident, not leading.** Like most discretionary-ish consumer spending,
  underwear sales soften *during* downturns. A series that falls while a recession
  is already underway describes the slump; it does not forecast it. The only
  testable forecasting claim pairs a year-Y dip with the year-(Y+1) outcome.

- **Reconstruction caveat.** No clean, free, long-run machine-readable men's-
  underwear unit-sales series exists. The series in `data.py` is a *curated
  reconstruction*: a smooth secular trend (~+1.4%/yr from population + replacement)
  with stylised recession-year softness. This means the *coincident* relationship
  is partly built-in and must NOT be read as evidence — we flag this loudly and
  test only the leading relationship as the genuine signal.

- **Tiny-n / base-rate neglect.** Within 1992–2024 there are ~33 annual
  observations, **4 dip years**, and **5 NBER recession years**. With counts this
  small, one coincidence flips a Fisher exact test, and recessions are rare enough
  that "all calm" is a near-costless default forecast.

## Recession-indicator literature (the honest comparators)

- **Estrella, A. & Mishkin, F. S. (1998).** "Predicting U.S. Recessions: Financial
  Variables as Leading Indicators." *Review of Economics and Statistics*, 80(1),
  45–61. The yield-curve slope (10y–3m) is the canonical, evidence-based recession
  predictor — the standard against which folklore gauges should be measured.

- **Sahm, C. (2019).** "Direct Stimulus Payments to Individuals" (the *Sahm Rule*).
  A real-time recession-onset rule based on the unemployment-rate moving average;
  a genuine, validated coincident-to-leading signal, unlike the MUI.

- **NBER Business Cycle Dating Committee.** The official US recession calendar used
  here as ground truth (contractions: 2001; 2007–2009; 2020 within our window).

## Spurious-indicator and small-sample literature

- **Leinweber, D. J. (2007).** "Stupid Data Miner Tricks." Reprinted commentary on
  the Bangladesh-butter / S&P 500 correlation. The MUI sits squarely in this class
  of vivid-but-empty macro folklore.

- **Harvey, C. R., Liu, Y. & Zhu, H. (2016).** "… and the Cross-Section of Expected
  Returns." *Review of Financial Studies*, 29(1), 5–68. The multiple-testing case
  for raising significance hurdles; a single anecdotal indicator discovered ex-post
  deserves heavy skepticism.

- **Fisher exact test.** The correct inference for a 2×2 dip × recession table with
  small cell counts (`scipy.stats.fisher_exact`), where the chi-square approximation
  is invalid.

## Method lineage used here

- **Fisher exact test** on the 2×2 (dip × next-year recession) table — exact, valid
  for 4 dips.
- **Permutation test** — shuffle dip labels 10,000 times; p = fraction of shuffles
  with excess recession rate ≥ observed.
- **Newey-West HAC t** (1 lag) on the timing-strategy-minus-benchmark return spread,
  to avoid over-stating significance from any serial dependence.
- **No look-ahead** — a year-Y dip is known only after Y closes, so it is paired
  with year-(Y+1) outcomes for both the recession test and the trade.

## Data sources

- **Shiller S&P 500 monthly dataset** (`_cache/shiller_sp500.parquet`), December-to-
  December nominal **price** returns (no dividends), 1992–2024 → 33 calendar-year
  returns. Optional `--fetch` pulls ^GSPC from yfinance as an alternative real tape.
- **Underwear index** — curated reconstruction, hardcoded in `data.py` (illustrative,
  not audited; see the reconstruction caveat above).
- **NBER recession calendar** — hardcoded `US_RECESSIONS` constant.

## Related desk studies

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the same tiny-n, base-rate
  teardown applied to a sports-folklore market predictor.
- **The lipstick / hemline / skyscraper folklore family**: vivid consumer-behaviour
  recession gauges in the same spurious class.
