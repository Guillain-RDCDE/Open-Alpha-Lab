# References & literature map — Study 681 (Relative-Rotation-Graph)

## The claim under test

- **The folklore.** The **Relative Rotation Graph (RRG)**, invented by **Julius de
  Kempenaer** (RRG Research; commercialised on StockCharts.com and Bloomberg's `RRG<GO>`
  function since the early 2010s), plots each sector's **RS-Ratio** (the level of its
  relative strength vs a benchmark) against its **RS-Momentum** (the rate of change of
  that relative strength). Sectors are said to rotate **clockwise** through four
  quadrants across a cycle — Lagging → Improving → Leading → Weakening → Lagging — and
  the trading rule is: buy sectors that have just entered (or are inside) **Leading**,
  trim/avoid **Lagging**. De Kempenaer's own book, *Relative Rotation Graphs* (2016,
  self-published / RRG Research), and the StockCharts ChartSchool page
  (https://school.stockcharts.com/doku.php?id=chart_analysis:rrg_charts) are the primary
  sources for the method; there is no peer-reviewed academic paper behind it — it is a
  **practitioner charting technique**, patent-pending visualisation (US application
  publications under Kempenaer's name) rather than a published, testable factor model.
- **The academic anchor (indirect).** The RRG is, by construction, a two-dimensional
  repackaging of **relative-strength / cross-sectional momentum** — the same object
  Jegadeesh & Titman (1993, *Returns to buying winners and selling losers*, JF) and
  Moskowitz & Grinblatt (1999, *Do industries explain momentum?*, JF) study directly.
  The RRG's own claim is that splitting momentum into a *level* axis and a *rate-of-
  change* axis adds information a plain 1-D momentum rank misses — an empirical claim
  we test head-on against the classic 6-1 sort.

## What we measure, and the honesty rails

- **RS-Ratio / RS-Momentum, our own explicit construction.** RRG vendors do not publish
  their exact smoothing constants, so we define both axes ourselves, in the open, as
  rolling z-scores of the relative-strength line and its rate of change (see
  [`relative_rotation_graph/strategy.py`](../relative_rotation_graph/strategy.py)) —
  W = 63 trading days (~one quarter, the classic RRG "tail" scaled from weekly to daily
  bars), M = 21 trading days (~one month). Chosen once, before looking at results, and
  never re-tuned to chase a number.
- **One documented execution lag.** The quadrant is read off the **month-end close**;
  the position it implies is held over the **following** month (weights formed at
  month-end *t* are applied to month *t+1*'s realised return) — no same-bar fill, no
  look-ahead.
- **The decisive control is matched-random, not plain equal-weight.** The rule "go to
  cash when nothing is Leading" mechanically drags a positive-drift market vs a
  fully-invested equal-weight basket — that is a *design* feature of the rule, not
  evidence of skill. To isolate whether the *quadrant selection itself* (which sectors,
  not how many) adds value, the headline test compares RRG to a Monte-Carlo-averaged
  control that picks the **same number of random sectors** RRG actually held each month
  (including the same cash months). SPY and the plain equal-weight basket are reported
  too, but labelled for what they are — count/cash-timing conflated with selection.
- **The 1-D control is the classic 6-1 momentum sort** (cf. sibling
  [225-sector-rotation](../225-sector-rotation/)) on the *same* universe and *same*
  monthly cadence — the direct test of the RRG's own claim that splitting momentum into
  two axes beats reading it as one.
- Newey-West HAC *t* on every active-return series; a synthetic positive control proves
  the quadrant machinery lights up on a planted persistent relative-drift and stays null
  with none planted.

## Data sources

- **Daily adjusted closes**, 11 SPDR sector ETFs (XLK/XLV/XLF/XLY/XLI/XLP/XLE/XLU/XLB/
  XLRE/XLC) + **SPY**, yfinance (no key), cached under `_cache/rrg_prices.csv`,
  1998-12 → 2026-06 (XLRE from 2015-10 inception, XLC from 2018-06 inception — same
  known quirk as siblings 225/506).
- Julius de Kempenaer, *Relative Rotation Graphs: How to Guide Global Asset Allocation
  and Sector Rotation with RRG Charts* (RRG Research, 2016).
- StockCharts ChartSchool, "Relative Rotation Graphs (RRG Charts)":
  https://school.stockcharts.com/doku.php?id=chart_analysis:rrg_charts
- Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers: Implications
  for Stock Market Efficiency*, Journal of Finance.
- Moskowitz & Grinblatt (1999), *Do Industries Explain Momentum?*, Journal of Finance.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [225-sector-rotation](../225-sector-rotation/) — the **same 11 SPDR sectors**, the
  same monthly cadence, but a plain **one-dimensional** 6-1 top-3 momentum sort with no
  level/momentum split and no quadrant machinery. This study uses 225's exact strategy
  as its **1-D control** to test the RRG's own claim of adding value over it — and
  reaches the same qualitative "no active edge over equal-weight" verdict independently.
- [506-industry-momentum](../506-industry-momentum/) — the same universe again, a
  12-1 long-short industry-momentum race against single-name momentum (Moskowitz &
  Grinblatt's own test). Long-short, not long-only-quadrant; no RS-Ratio/RS-Momentum
  split.
- [246-defensive-sectors](../246-defensive-sectors/) — asks whether **two specific**
  defensive sectors (XLP+XLU) co-moving is a risk-off *timing* signal for SPY. A
  two-sector canary, not an 11-sector rotation rule, and the question is market-timing
  (forecast SPY), not sector selection.
- None of the siblings implement the RRG's own two-axis (level + rate-of-change)
  quadrant construction — this study is the first to build and test the quadrant rule
  itself, using the others' shared universe and cadence as the fair 1-D benchmark.
