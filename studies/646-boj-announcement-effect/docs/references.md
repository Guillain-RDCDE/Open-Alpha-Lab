# References & literature map — Study 646 (BoJ Announcement Effect)

## The claim under test

- **The folklore.** "Japanese equities and the yen react systematically to Bank of Japan
  policy decision days" — especially in the "surprise" eras: the January-2016 negative-rate
  (NIRP) shock, the September-2016 introduction of Yield Curve Control (YCC), and the
  2022-2024 YCC-tweak/exit sequence that helped set off the August-2024 global carry unwind.
  Unlike the FOMC (which announces on a fixed, well-anticipated schedule with a stated
  reaction function), a large share of BoJ moves under YCC were genuine surprises relative to
  consensus — so the claim is explicitly a *tail/surprise* story, not a "set your watch to it"
  scheduled-directional one.
- **The academic anchor.** Ueda (2012, *The Effectiveness of Non-Traditional Monetary Policy
  Measures: The Case of the Bank of Japan*, The Japanese Economic Review) and Fukuda (2021,
  *Bank of Japan's Yield Curve Control and Financial Markets*) document event-study reactions
  to individual BoJ policy surprises. Fratzscher & Straub (2009) and De Santis & Zimic (2018)
  are the FOMC/ECB analogues of "does a scheduled central-bank announcement move risky
  assets systematically" — the general genre this study belongs to. The December-2022 "Kuroda
  shock" (YCC band widened from ±0.25% to ±0.50%) is widely cited in market commentary as the
  cleanest single-day BoJ surprise on record; the July-2024 hike is widely credited as the
  proximate trigger of the August-2024 global carry-trade unwind (a *subsequent-day* event,
  outside this study's single-day window by construction).
- **The adjacent (distinct) result.** The yen's behavior as a general risk-off hedge — driven
  by carry-trade unwinds on ANY risk-off day, not specifically BoJ decision days — is sibling
  study [615-yen-safe-haven](../../615-yen-safe-haven/)'s subject, not this one's. See the dedup
  map below.

## What we measure, and the honesty rails

- **Decision-day return vs all other days**, for EWJ (unhedged Japan equities, USD) and the
  yen (minus the USDJPY change — positive = yen strengthens, matching
  [615-yen-safe-haven](../../615-yen-safe-haven/)'s sign convention). Welch *t* for the split
  (Welch 1947); a **Newey-West (1987)** 5-lag *t* on the decision-day dummy regression is the
  autocorrelation-robust cross-check.
- **Every decision on record, not just "scheduled" ones.** Unlike sibling study
  [637-fomc-vol-crush](../../637-fomc-vol-crush/) (which deliberately excludes FOMC
  emergency/inter-meeting actions because the FOMC claim is specifically about the *scheduled*
  calendar), this study's claim is explicitly about the surprise-driven eras, so the BoJ table
  keeps every decision/statement date on record — scheduled monthly meetings pre-2016, the
  post-2016 8-times-a-year cadence, and the handful of true inter-meeting emergency actions
  (2008 financial crisis, 2010 European-crisis interventions, 2011 post-earthquake, 2020
  COVID).
- **Hit rate carries a Wilson (1927) interval**; the placebo is a 20-seed × 1,000-draw
  random-calendar null, run **two-sided** (a BoJ surprise has no pre-committed direction,
  unlike the FOMC vol-crush claim, which is one-sided by construction); the era split
  (2016-01-29, the NIRP announcement) is *justified, not snooped* — the start of the
  "unconventional surprise" regime as a matter of policy-history record.
- **Realized-range cross-check, promoted to the grey third axis.** (H−L)/day-open on the same
  days: decision days ARE louder for both instruments (Welch *t* = +2.49 EWJ, +6.30 yen) even
  though the average *direction* of that extra motion is a wash — a materially different
  finding from 637's sibling result (where implied vol collapses on a *louder-than-average*
  day), here motion is elevated on *both* sides and cancels.

## A named data quirk — and why it matters here specifically

- From **2022 onward**, yfinance's `JPY=X` daily bars have their `Close` field silently
  **duplicate the same row's `Open`** on the large majority of days (verified: >95% of
  2023-2025 rows vs <5% pre-2022) — a documented Yahoo FX-data limitation, not a study
  artifact. Using naive close-to-close returns would have muted or *mistimed* exactly the
  events this study is built to measure: under the broken field, the December-2022 "Kuroda
  shock" day reads as essentially flat (the real ~4% intraday plunge and partial recovery gets
  smeared across two labeled bars). The fix, applied **uniformly across the whole 2005-2026
  sample** (not switched on only for the broken years, so no era-dependent convention creeps
  in): the yen return dated D is `Open[D+1] / Open[D] − 1`, a 24h window that still starts
  before the ~11:30am JST announcement (zero look-ahead) and is, if anything, slightly WIDER
  than a pure close-to-close window would have been. `High`/`Low` remain reliable throughout
  the sample, so realized range uses `(High−Low)/Open` rather than the broken previous-close.
  This is exactly the kind of silent-drift risk `quantlab.repro`'s fingerprinting exists to
  catch — a future re-run whose ``JPY=X`` cache differs will flag a different fingerprint,
  a visible prompt to re-check this convention.

## Data sources

- **EWJ raw OHLC** and **`JPY=X` (USDJPY) OHLC** — yfinance (no key), cached under `_cache/`
  (`boj_ewj.csv`, `boj_jpy.csv`), 2005-01-03 → 2026-06-30.
- **BoJ Monetary Policy Meeting decision/statement dates 2005 → 2026**, hardcoded in
  [`data.py`](../boj_announcement_effect/data.py). Source: Bank of Japan official archives —
  "Statements on Monetary Policy" (https://www.boj.or.jp/en/mopo/mpmdeci/state_all/index.htm)
  for 2005-2009 and "Past Monetary Policy Meetings"
  (https://www.boj.or.jp/en/mopo/mpmsche_minu/past.htm) plus the current schedule page
  (https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm) for 2010-2026. For two-day meetings
  the date recorded is the **second** (announcement) day.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [615-yen-safe-haven](../../615-yen-safe-haven/) — does the yen rally on **general risk-off
  equity days** (any day SPY sells off hard), driven by carry-trade unwind positioning? A
  *daily-frequency, any-day* signal, found real-but-decayed (HAC *t* = −4.62 full-sample,
  ≈ dead since 2011). This study asks the opposite-shaped question: a specific, named
  **calendar** of ~250 BoJ decision days, and finds **no** systematic reaction even though the
  underlying instrument (the yen) is the same.
- [645-ecb-announcement-effect](../../645-ecb-announcement-effect/) — the equivalent question for
  the European Central Bank's Governing Council decisions. Different central bank, different
  currency, same protocol family — a natural cross-check on whether "does a G4 central bank's
  decision day move its home-market equities/currency" generalizes; see that study's own
  README for its verdict.
- [637-fomc-vol-crush](../../637-fomc-vol-crush/) — the Federal Reserve's decision-day effect,
  but on **implied volatility** (the VIX), not on equities/currency returns, and restricted to
  **scheduled** meetings only (the FOMC claim is explicitly about a metronomic calendar; the
  BoJ claim here is explicitly about the surprise tail). Found Real (VIX crush, *t* ≈ −3.9)
  but Mirage on tradability. This study finds a materially different shape: no directional
  Signal at all on either axis, but the grey third-axis volatility myth-check IS confirmed —
  the mirror image of 637's own realized-vs-implied cross-check.

None of the siblings test what **EWJ and the yen do specifically on a BoJ decision day** across
the full NIRP/YCC surprise history — this study's own axis.
