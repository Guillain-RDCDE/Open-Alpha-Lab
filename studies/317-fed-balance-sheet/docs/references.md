# References & literature map — Study 317 (Fed-Balance-Sheet)

## The claim under test — "Don't fight the Fed"

- **The slogan.** Attributed to Martin Zweig (*Winning on Wall Street*, 1986): align with
  the direction of Fed policy. In the QE era it mutated into a balance-sheet version —
  *"liquidity drives everything; long stocks when the Fed is expanding its balance sheet
  (QE), get out (or short) when it is shrinking it (QT)."* It is a staple of finance
  Twitter/macro commentary (e.g. recurring "Fed liquidity vs S&P 500" overlay charts). The
  testable hypothesis: **daily equity returns are higher in QE regimes than in QT regimes**,
  by enough to time on. We test exactly that contrast.

## The regime variable — the Fed balance sheet (WALCL) and its programmes

- **FRED series `WALCL`** — *Total Assets (Less Eliminations from Consolidation), Wednesday
  Level*, Board of Governors of the Federal Reserve System. The canonical weekly balance-sheet
  level. **Network-blocked in this environment**, so we encode the announced *direction* of
  the balance sheet as a hand-built table; the level itself is not needed to test a
  direction-based slogan.
- **The programme dates** (transcribed into [`data.REGIME_TABLE`](../fed_balance_sheet/data.py)):
  QE1 (Nov 2008–Mar 2010), QE2 (Nov 2010–Jun 2011), Operation Twist (size-neutral, 2011–12),
  QE3 (Sep 2012–Oct 2014), the 2017–19 balance-sheet **runoff** (the first QT), COVID QE
  (Mar 2020–Mar 2022), and the 2022→ QT. Sources: Federal Reserve FOMC statements and the
  Fed's "Credit and Liquidity Programs and the Balance Sheet" pages; summarised in
  Federal Reserve Bank of St. Louis / Cleveland reviews of unconventional policy.

## What the academic evidence actually says

- **Announcement-window effects, not regime drift.** Krishnamurthy & Vissing-Jorgensen (2011),
  *The Effects of Quantitative Easing on Interest Rates* (Brookings Papers) and Gagnon, Raskin,
  Remache & Sack (2011), *The Financial Market Effects of the Federal Reserve's Large-Scale
  Asset Purchases* (IJCB) document large effects of QE on **bond yields around announcements**,
  not a persistent daily equity drift over the months a programme is running. The QE→stocks
  link is far weaker and far more contested than the QE→yields link.
- **Pre-FOMC drift is the real calendar anomaly.** Lucca & Moench (2015), *The Pre-FOMC
  Announcement Drift* (Journal of Finance) — equity returns concentrate in the 24h before
  scheduled FOMC meetings. That is a *meeting-calendar* effect, distinct from balance-sheet
  direction; the desk tests it in [Study 67 — Fed-Drift](../../67-fed-drift/) and the even-week
  cycle in [Study 135 — FOMC-Cycle](../../135-fomc-cycle/).
- **Reverse causation / endogeneity.** The Fed eases *because* the economy and markets are
  weak (and tightens because they are strong). Cieslak & Vissing-Jorgensen (2021), *The
  Economics of the Fed Put* (RFS) — policy responds to the stock market. So any raw "QE
  coincides with rallies" is contaminated by the Fed easing into the very crashes that then
  mechanically rebound — exactly the QE−FLAT artefact this study isolates.

## Method lineage (the desk's shared engine)

- **HAC / Newey–West *t*-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../fed_balance_sheet/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992), *A circular block-resampling
  procedure for stationary data* — used for the CI on the QE−QT mean contrast, preserving the
  volatility clustering an i.i.d. bootstrap destroys — [`strategy.diff_block_bootstrap_ci`].
- **Excess-vs-excess Sharpe race.** The desk convention (METHODOLOGY → House rules): a
  part-time-in-cash book is raced against a fully-invested one only after both are netted of
  the cash rate — [`strategy.race`].

## Data sources used here

- **Yahoo! Finance SPY daily bars** (via the shared [`quantlab.data`](../../../quantlab/data.py)
  cache), split-only / price-only, 1993→. Regime labels from the hand-built table above.
  Headline numbers are pinned with an as-of date and content fingerprint
  ([`docs/results.md`](results.md)). The offline reproducible core and test-suite run on the
  deterministic [`data.synthetic_daily`](../fed_balance_sheet/data.py) generator, never the
  network.

## Related desk studies

- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: the pre-FOMC announcement drift — a real,
  calendar-based Fed effect (Real/Fragile), distinct from balance-sheet direction.
- **[Study 135 — FOMC-Cycle](../../135-fomc-cycle/)**: the even-week FOMC cycle (Weak/Mirage).
- **[Study 47 — Paper-Moon](../../47-paper-moon/)** & **[Study 118 — Fed-Model](../../118-fed-model/)**:
  the Fed Model (E/P vs the 10-year yield) — the *valuation* flavour of "watch the Fed".
