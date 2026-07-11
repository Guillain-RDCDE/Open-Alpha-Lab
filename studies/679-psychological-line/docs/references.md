# References & literature map — Study 679 (Psychological Line)

## The claim under test

- **The folklore.** The **Psychological Line (PSY)** counts the share of up-closes over a
  trailing window — classically 12 trading days (two-ish weeks) — and reads it as a crowd
  sentiment gauge: **PSY > 75%** means "almost everybody who wanted to buy already has,"
  overbought, sell; **PSY < 25%** means the opposite, oversold, buy. It is one of the older
  members of the Japanese technical-analysis toolkit that reached Western retail charting
  software in the 1990s alongside candlesticks.
- **The academic anchor.** There isn't much of one — PSY has essentially no dedicated
  peer-reviewed literature; it is documented in practitioner references rather than journals.
  Steve Nison's *Beyond Candlesticks* (1994) and *Japanese Candlestick Charting Techniques*
  (1991, 2nd ed. 2001) popularized it in English alongside the rest of the Japanese charting
  canon. It is functionally a **binary-count relative of the Relative Strength Index** — RSI
  weights each day's *magnitude* of gain/loss; PSY only counts whether the day was up or
  down, discarding magnitude entirely (Wilder's RSI, 1978, is the closer-studied cousin). The
  academic evidence on RSI-style oscillators themselves is mixed-to-negative once transaction
  costs and out-of-sample testing are applied (e.g. the broad "does technical analysis work"
  literature reviewed in Park & Irwin 2007, *Journal of Economic Surveys*), which is the
  closest thing PSY has to outside evidence, and it is not encouraging.
- **The steelman.** A pure up/down count over 12 days is, in effect, a crude binomial test of
  "has the crowd been buying or selling more often lately" — if daily direction had any
  short-horizon mean-reverting structure (an overreaction-and-correction pattern), counting
  streak extremity is a defensible, if noisy, way to detect it.

## What we measure, and the honesty rails

- **One documented execution lag.** PSY is computed from closes up to and including bar *t*;
  every position (forward-return window or actual trade) enters at bar *t+1*'s **open** and
  exits at the close of bar *t+h* (*h* = 5 trading days, the study's primary horizon). No
  look-ahead.
- **Trigger events, not every in-zone day.** PSY is a rolling window, so consecutive days
  inside a zone share nearly the same underlying 12 closes and are far from independent
  draws. Only the day PSY *enters* a zone counts as an event (mirroring the desk's other
  oscillator studies' crossover framing), with a cooldown (`min_gap` = the hold period)
  between re-entries of the same side — an undocumented "every day counts" version of this
  same test inflates a null world's false-positive rate roughly 5x on this tape (see
  `docs/results.md`'s synthetic-control section).
- **Welch t is the planned primary** on the (still imperfectly independent) trigger-day
  split; a Newey-West (5-lag) cross-check is reported alongside for the residual overlap in
  the 5-day forward windows.
- **The random-direction control** ("beats a coin?") uses the *identical* entry bars and
  dates as the PSY-driven trade, randomizing only the long/short sign — the fairest possible
  comparison, and the one this desk's other oscillator studies use.
- **Parameter grid, not a single cherry-picked rule.** Window in {10, 12, 14, 20} x
  thresholds in {(20,80), (25,75), (30,70)} — the textbook (12, 25/75) is highlighted, and
  the grid's few significant corners are named as the parameter-mining shape they are, not
  quietly promoted.

## Data sources

- **SPY, QQQ, IWM, AAPL, TSLA, NVDA daily OHLC** (total-return adjusted), yfinance (no key),
  cached under `_cache/` (`psy_<ticker>.csv`), 2003-01-02 → 2026-06-30 (TSLA from its 2010-06
  inception). Named survivorship caveat: the mega-cap sleeve (AAPL/TSLA/NVDA) is three
  well-known liquid names, not a systematically re-derived historical panel — see
  `docs/results.md`'s data-stamp section.
- Steve Nison, *Beyond Candlesticks: New Japanese Charting Techniques Revealed* (Wiley,
  1994) — the standard English-language reference for the Psychological Line and its
  siblings.
- J. Welles Wilder, *New Concepts in Technical Trading Systems* (1978) — RSI, the
  magnitude-weighted relative of PSY's pure up/down count.
- Cheol-Ho Park & Scott H. Irwin, "What Do We Know About the Profitability of Technical
  Analysis?", *Journal of Economic Surveys* 21(4), 2007 — the closest thing to a literature
  anchor for oscillator-style rules broadly.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [107-stochastic-oscillator](../107-stochastic-oscillator/) — George Lane's %K/%D, a
  *magnitude-normalized* position of the close within a trailing high-low **range**. PSY
  only counts up/down **days**, discarding both magnitude and the intrabar range entirely —
  a strictly coarser, binary-count signal.
- [127-williams-r](../127-williams-r/) — Larry Williams' %R, the same close-within-range
  idea as the stochastic, inverted in sign. Same "position in range" family; PSY has no
  range in it at all.
- [179-aroon](../179-aroon/) — Chande's Aroon measures **days since** the most recent
  high/low within a look-back window (a timing/recency signal). PSY measures a **frequency
  count** of up-days, not recency of an extreme — a different statistic on the same kind of
  window.
- [680-disparity-index](../680-disparity-index/) — the disparity index compares the *close*
  to a moving *average* (a magnitude-of-deviation signal). PSY never touches a moving
  average or a magnitude — only the binary sign of each day's change.

None of the siblings reduce a price series to "what fraction of the last N days closed up" —
that specific, deliberately magnitude-blind construction is this study's own axis, and (per
the results above) it does not turn out to buy back the information the others already
discard.
