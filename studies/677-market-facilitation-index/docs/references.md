# References & literature map — Study 677 (Market Facilitation Index)

## The claim under test

- **The folklore.** Bill Williams' **BW-MFI** — *Market Facilitation Index* —
  `(High - Low) / Volume`, introduced in *Trading Chaos* (Williams, 1995) and *New
  Trading Dimensions* (Williams & Williams, 1998). Williams crosses the bar-to-bar
  *direction* of MFI against the bar-to-bar direction of Volume to name four bar
  "colors": **Green** (MFI↑, Volume↑ — the market is "in gear," expect the move to
  **continue**), **Fade** (MFI↓, Volume↓ — traders leaving, the move has stalled),
  **Fake** (MFI↑, Volume↓ — price moved on thin participation, distrust it), and
  **Squat** (MFI↓, Volume↑ — heavy activity, no price progress, the market is "squatting"
  before a violent move; retail folklore reads this as an imminent **reversal**).
- **The academic anchor — there mostly isn't one.** BW-MFI is not a peer-reviewed
  construct; it originates entirely in Williams' own trading books and their
  chaos-theory framing ("fractal geometry," "Alligator," "Gator Oscillator" — the same
  author's companion indicators). We are not aware of a published, out-of-sample test of
  the four-color continuation/reversal claim in the academic literature — which is itself
  informative: a 30-year-old, widely-taught retail rule with **no independent
  replication** is exactly the kind of claim this desk exists to check directly against
  the tape.
- **The closest generic anchor.** The idea that "price move per unit of volume" carries
  information echoes the broader **volume-price** literature (Amihud illiquidity
  (2002), the Elder Force Index, Arms' Ease of Movement — see the dedup map). None of
  them test the specific claim here: a **four-way categorical** state built from *two
  bar-to-bar sign changes*, rather than a continuous timing signal.

## What we measure, and the honesty rails

- **The classifier.** `state(t)` from `sign(ΔMFI(t))` × `sign(ΔVolume(t))`, using RAW
  (unadjusted) High/Low/Volume — adjustment for splits/dividends would distort the
  High-Low geometry that BW-MFI is built on. No look-ahead: `state(t)` needs only bar
  *t*'s own High/Low/Volume and bar *t-1*'s, so it is known at the close of *t*.
- **The continuation score.** `sign(ret(t)) × fwd_ret(t)` — positive means tomorrow
  continued today's direction, negative means it reversed. This is the natural, minimal
  operationalization of "Green predicts continuation / Squat predicts reversal": Welch
  *t* for each state's mean vs all other days, plus a 2,000-draw **label-shuffle
  placebo** (Efron-style permutation on the state assignment, not the returns) so no
  conditional claim ships without an uncertainty check.
- **Pooling and per-ticker replication.** The headline runs on SPY; a pooled six-ticker
  version (SPY, QQQ, DIA, IWM, XLE, GLD) checks whether more state-days rescue either
  claim, and a per-ticker table checks whether the *sign* even replicates across
  independent tapes — the sharpest test a categorical, non-parametric claim like this one
  can face.
- **The timer.** Two state-conditioned position rules (ride Green, avoid Squat) raced
  against buy-and-hold — one execution lag (the color known at close *t* earns the
  return of *t+1*), one-way costs × NAV per flip (1 bp headline, 0-5 bp swept), a
  Newey-West HAC *t* on the daily return difference, and a sign-permutation placebo on
  the timer itself.

## Data sources

- **SPY + 5-ETF basket (QQQ, DIA, IWM, XLE, GLD)** — daily RAW OHLCV (for the BW-MFI
  ratio) and adjusted close (total return, for every return computation), yfinance (no
  key), cached under `_cache/` (`mfi_<ticker>.csv`), 1993-02-01 → 2026-06-30 (each
  ticker's own inception where later — GLD from 2004-11-18).
- BW-MFI definition: Williams, B. (1995). *Trading Chaos: Maximize Profits with Proven
  Technical Techniques*. Wiley. Williams, B. & Williams, J. (1998). *New Trading
  Dimensions*. Wiley.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [418-money-flow-index](../../418-money-flow-index/) — the **Money Flow Index**
  (typical-price × volume, up/down-day RSI-style oscillator by Quong & Soudack) — a
  *continuous* contrarian overbought/oversold signal, not BW-MFI's categorical
  4-color state machine. Same "money flow" name, unrelated construction and claim.
- [419-chaikin-money-flow](../../419-chaikin-money-flow/) — Chaikin's accumulation/
  distribution-based money-flow oscillator — again continuous, and built from the
  close's position within the bar's range, not `(High-Low)/Volume`'s bar-to-bar
  direction crossed with volume's own direction.
- [423-force-index](../../423-force-index/) — Elder's `(ΔClose × Volume)` EMA oscillator,
  tested as a zero-cross timing rule. A single continuous number, not a 4-way
  categorical classifier, and the claim there ("flags reversals") is about the *sign of
  the smoothed indicator*, not a range/volume-agreement pattern.
- [424-ease-of-movement](../../424-ease-of-movement/) — Arms' `(range change / volume)`
  cousin: structurally the *closest* relative (also a range-over-volume ratio), but
  tested as a continuous long/flat trend signal on its own sign, never crossed against
  volume's own direction to build four discrete states. This study is BW-MFI's specific,
  additional claim: **volume's own bar-to-bar direction matters**, not just the
  range/volume ratio's level.
- [676-gator-oscillator](../../676-gator-oscillator/) — Bill Williams' *other* famous
  indicator (the "Gator," built from his Alligator moving-average lines) — same author,
  same trading-chaos framework, but a moving-average-convergence construct with no
  volume term at all. A different mechanism entirely; grouped here only because it
  shares an inventor.

None of the siblings test BW-MFI's specific claim: a **four-way state built by crossing
two bar-to-bar sign changes** (MFI's own direction against volume's own direction), with
Green named for continuation and Squat named for reversal. That categorical construction,
and those two specific predictions, are this study's own axis.
