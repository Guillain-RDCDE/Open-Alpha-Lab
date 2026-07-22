# References & literature map — Study 802 (Stock-Split-Modern)

## The claim under test

- **The anomaly.** **Ikenberry, Rankine & Stice (1996), *What Do Stock Splits Really
  Signal?* (JFQA)** documented post-split *drift*: stocks that split earned an abnormal
  ~+7.9% over the year following the split **announcement**, which they read as the
  market under-reacting to a costly signal of management confidence. Earlier work
  (Grinblatt, Masulis & Titman 1984; Fama, Fisher, Jensen & Roll 1969) established the
  ex-date announcement effect; **Ikenberry & Ramnath (2002)** re-confirmed the
  post-announcement drift out of sample.
- **The modern, retail-facing version.** After **Tesla (2020, 5:1; 2022, 3:1), Apple
  (2020, 4:1), Nvidia (2021, 4:1; 2024, 10:1), Amazon (2022, 20:1) and Alphabet (2022,
  20:1)** all split and all soared through 2020-2024, "buy the splitters" became a
  popular narrative. That is the specific, steelmanned claim this study re-tests: **is
  the classic positive post-split drift still present in the post-2020 mega-cap era?**
- **The honest re-test.** We measure **market-adjusted abnormal returns** (stock total
  return minus SPY total return over the identical window) so that the 2020-2024 bull
  market — a huge confound — is de-trended out. What is left is the *abnormal* drift, the
  only thing the anomaly can honestly claim.

## What we measure, and the honesty rails

- **Abnormal CAR = stock buy-and-hold − SPY buy-and-hold**, over [−1, +1] sessions around
  the ex-date and over 21/63/126/252 trading days after it, entered at the **ex-date
  close** (publicly knowable at that instant; first earned return is t+1 — the single
  documented execution lag). Prices are `auto_adjust=True` (split + dividend adjusted →
  total return), so the split mechanics never contaminate the return.
- **The decisive statistic is a Newey-West (Bartlett) HAC one-sample *t*** on the
  **date-ordered** abnormal CARs — split events cluster in calendar time (the 2022 wave
  especially), so a plain i.i.d. *t* overstates independence. A HAC *t* ≥ 2 on the real
  tape is the desk's bar for `REAL` (METHODOLOGY → *The inference bar*).
- **Cohorts:** all forward splits since 2010, pre-2020, post-2020 (the "modern" era), and
  the seven post-2020 **mega-cap** splits the narrative actually names. The era split
  (2020-01-01) is tested as a **Welch difference**, not eyeballed.
- **Placebo:** the same names entered on **random non-split dates** (matched control),
  abnormal-adjusted — if the split date carries no information, split-date CARs should not
  beat the random-date cloud. **Win rates carry Wilson (1927) intervals.**
- **Survivorship / selection is named on the Signal axis.** The basket is a
  *current-membership* list of today's large-caps — a selection of names that already
  won. That bias points **for** the drift (we are cherry-picking winners), so a *failure*
  to find positive drift even here is a strong `NONE`; a bias cannot be blamed for the
  absence.

## Why the effective-date caveat matters (and keeps this honest)

- `yfinance` reports the **effective (ex-split) date**, which trails the **announcement**
  by ~3-6 weeks. The Ikenberry et al. drift is measured from *announcement*; our
  post-effective window is strictly weaker (by the ex-date much of any announcement drift
  is already in the price). The absence of post-effective drift is therefore **consistent
  with the original signal being fully priced by the ex-date** — we do not claim to
  falsify the 1996 announcement-window result, only to show the ex-date version is dead.

## Data sources

- **Split calendar** — `yfinance` `Ticker(...).splits` ("Stock Splits" action series),
  forward splits (ratio ≥ 1.5) since 2010, cached under `_cache/` (`ssm_splits.parquet`).
- **Prices + market benchmark** — `yfinance` daily total-return (`auto_adjust=True`)
  closes for the basket **and SPY**, cached (`ssm_prices.parquet`).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [142-split-drift](../142-split-drift/) — the **classic** post-split-drift test on a
  2000-2025 large-cap basket, comparing split forward returns to a matched no-split
  baseline. It finds `NONE` (post-*effective* drift flat-to-negative). This study is the
  **modern post-2020 mega-cap re-cut** of the same anomaly, using an **SPY-hedged
  abnormal-return** lens and an explicit era split, aimed squarely at the "Tesla/Nvidia
  splitters" narrative rather than the 2000s basket.
- [250-reverse-split](../250-reverse-split/) — the **opposite** corporate action, the
  "kiss of death" reverse split (`WEAK`/`Mirage`, distress-confounded). Forward splits
  (this study) and reverse splits (250) are different events with opposite signalling.

Neither sibling re-tests the **modern mega-cap forward-split** claim with a market-hedged
abnormal-return measure and a tiny-N honesty caveat — which is this study's own axis.
