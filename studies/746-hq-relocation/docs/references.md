# References & literature map — Study 746 (HQ-Relocation)

## The claim under test

- **The folklore.** A company that announces **relocating its headquarters** — an *inversion*
  abroad (Eaton→Ireland 2012, Medtronic→Ireland 2014, Johnson Controls→Ireland 2016), or a jump
  to a **lower-tax U.S. state** (Schwab, CBRE, HPE, Oracle→Texas in 2019-20, Tesla 2021,
  Caterpillar 2022, Chevron 2024) — is said to be *information*. Two camps read it opposite ways:
  the **signal** camp ("the market prices in the tax saving and cost cut — buy it") and the
  **distraction** camp ("a splashy new address is management theatre that masks a weak business —
  fade it"). We test **both** legs: an announcement pop and a longer post-announcement drift.
- **Where it's repeated.** Financial media covered the 2012-2016 **corporate-inversion wave** and
  the 2019-2024 **"great Texas migration"** as market-moving events; sell-side notes routinely
  attach a valuation story to a domicile change. The believers' framing is that the *reason*
  (tax/incentive vs talent/cost) predicts the reaction — which is exactly the split we score.

## The academic anchor — inversions and relocations in the event-study literature

- **Desai & Hines (2002), *Expectations and Expatriations: Tracing the Causes and Consequences of
  Corporate Inversions*, National Tax Journal.** The canonical study of inversion announcements:
  positive average abnormal returns around the announcement, but small and heterogeneous, and
  consistent with tax savings being partly anticipated. The effect is neither large nor clean.
- **Cloyd, Mills & Weaver (2003), *Firm Valuation Effects of the Expatriation of U.S.
  Corporations to Tax-Haven Countries*, Journal of the American Taxation Association.** Mixed
  valuation effects of expatriation/inversion; the market reaction depends heavily on the firm's
  tax position, not on the address per se.
- **Seida & Wempe (2004), *Effective Tax Rate Changes and Earnings Stripping Following Corporate
  Inversion*, National Tax Journal**, and **Babkin, Glover & Levine (2017)** on inversion gains
  accruing to shareholders vs the tax base — the literature's consensus is that any effect is
  *conditional and modest*, not a reliable buy signal.
- **Headquarters-relocation studies** (e.g. **Ghosh, Rodríguez-Serrano & Chetty**; the
  urban-economics and management literatures on HQ moves) find real *operational* consequences
  (labor markets, agglomeration) but no robust, tradable stock-price signal around the
  announcement — the move is a long-horizon strategic decision, not a short-window catalyst.

## Why the honest answer is "non-event" — small samples and salience selection

- **Small-sample inference.** With ~20 documented moves (14 tax / 6 other), the cross-section of
  abnormal returns has a large standard error. We test each bucket's mean CAR against zero with a
  **one-sample / Welch (1947) t**, the tax−other gap with a two-sample Welch t, and — because the
  sample is tiny — a **placebo / randomization null**: draw the same number of random non-event
  windows on the same tickers and ask how often chance matches the observed CAR (Fisher's
  randomization logic; Efron & Tibshirani, 1993, *An Introduction to the Bootstrap*).
- **Event-study machinery.** The market-model CAR and its standard errors follow **MacKinlay
  (1997), *Event Studies in Economics and Finance*, Journal of Economic Literature**, and **Brown
  & Warner (1985), *Using Daily Stock Returns*, Journal of Financial Economics** — a clean
  pre-event estimation window, a gap to avoid leakage, and abnormal returns cumulated over a
  short event window.
- **Selection on salience + survivorship.** The table is the set of moves *famous enough to be
  remembered and dated*, and every name **survived** to keep trading. Building a "law" from
  memorable cases is the classic data-snooping trap (Harvey, Liu & Zhu, 2016, *…and the
  Cross-Section of Expected Returns*, RFS); a representative, dated table — not just the legends —
  is the honest test, and the mild survivorship tilt is named on the Signal axis (Brown,
  Goetzmann, Ibbotson & Ross, 1992, *Survivorship Bias in Performance Studies*, RFS).

## Method lineage (the desk's shared engine)

- **Market-model CAR.** [`strategy.event_car`](../hq_relocation/strategy.py) fits
  `r_stock = a + b·r_SPY` on a 120-day estimation window (10-day gap) and cumulates abnormal
  returns over the event window, with an optional entry lag for tradable variants.
- **Placebo null + Welch t.** [`strategy.placebo_car_dist`](../hq_relocation/strategy.py),
  [`strategy.welch_t`](../hq_relocation/strategy.py) and
  [`strategy.placebo_pvalue`](../hq_relocation/strategy.py) — the bucket means vs zero, the
  tax−other gap, and a 20,000-draw randomization null sized to the event count.
- **Drift leg + costs.** [`strategy.drift_panel`](../hq_relocation/strategy.py) and
  [`strategy.net_of_costs`](../hq_relocation/strategy.py) — the post-announcement [+1, +63] drift
  the "signal" camp needs, entered the day after the headline, net of a one-way large-cap cost.
- **Deterministic synthetic control.**
  [`data.synthetic_events`](../hq_relocation/data.py) plants a known tax-bucket CAR edge of size
  `car_bps`; with `car_bps = 0` the inference must NOT manufacture significance, and a large edge
  must light up. The control runs offline.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + 20 HQ-move tickers, 2010-01-04 → 2026-06-30
  (as-of), cached under `_cache/` as one parquet per ticker. The relocation table (tickers,
  announce dates, tax/other) is hardcoded in [`data.HQ_MOVES`](../hq_relocation/data.py); famous
  un-priceable moves are listed in [`data.UNPRICEABLE`](../hq_relocation/data.py) for the
  selection caveat. Headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 391 — CEO-Turnover](../391-ceo-turnover/)**: the direct methodological sibling — a
  market-model CAR event study over a hardcoded, labelled table of corporate announcements, same
  small-sample power problem.
- **[Study 389 — Name-Change-Effect](../389-name-change-effect/)**: the adjacent "cosmetic
  corporate change" study — does a theme-chasing rebrand pop? Same family (a label/address, not a
  fundamental), same survivorship-and-selection pathology.
