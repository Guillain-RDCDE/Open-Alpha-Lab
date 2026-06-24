# References & literature map — Study 407 (Dark Cloud Cover & Piercing Line)

## The claim under test

- **The folk recipe.** The **Piercing Line** and **Dark Cloud Cover** are two-candle reversal
  patterns from the Japanese candlestick canon, popularised in the West by Steve Nison,
  *Japanese Candlestick Charting Techniques* (New York Institute of Finance, 1991; 2nd ed. 2001).
  Piercing Line: after a long down (black) day, an up (white) day opens below the prior low but
  closes **more than halfway up the prior body** — a bottom, *buy*. Dark Cloud Cover: the mirror —
  after a long up day, a down day opens above the prior high but closes **more than halfway down
  the prior body** — a top, *sell*. We steelman this as: *the signed forward return after a twin,
  entered the next open and net of costs, is positive (the reversal happens).*

## Why the steelman is *almost* coherent — the real effect it leans on

- **Short-horizon reversal exists, in the right place.** The twins are an intraday-rejection story
  (a gap that fails). Genuine one-month reversal at the single-stock level is documented by
  Jegadeesh (1990), *"Evidence of Predictable Behavior of Security Returns"* (Journal of Finance),
  and long-horizon overreaction by DeBondt & Thaler (1985), *"Does the Stock Market Overreact?"*
  (Journal of Finance) — but those operate over weeks-to-years, not a single day-after-a-gap.
- **Gaps do partially fill.** There is a real microstructure tendency for opening gaps to retrace
  intraday — the seed of the twins' logic. But "the gap fills" and "the trend reverses for days" are
  different claims, and the pattern is sold as the latter.
- **Nison's own caveats.** Nison stresses the twins are *confirmation-dependent* — most reliable
  after an established trend and on heavy volume. We test exactly those filters (the myth-check) and
  find they don't flip the sign.

## The failure mode exposed

- **A failed gap is often just continuation.** On trending US large-caps, a Dark Cloud Cover (a gap
  up that closes weak) is frequently followed by *more* upside, not a top — so the short bleeds
  (HAC *t* down to −4.86 at 5 days in this study). The textbook reads the rejection as exhaustion;
  the tape reads it as a pause in an uptrend.
- **The "good" leg is just beta.** The Piercing leg's positive 5–10-day return underperforms the
  unconditional long drift over the same bars — it is the market's always-up tendency, not reversal
  alpha. This is the same beta-vs-alpha trap documented for the engulfing twin in Study 402.
- **Data-snooping over candlestick rules.** Park & Irwin (2007), *"What Do We Know About the
  Profitability of Technical Analysis?"* (Journal of Economic Surveys), and Marshall, Young &
  Rose (2006), *"Candlestick Technical Trading Strategies: Can They Create Value for Investors?"*
  (Journal of Banking & Finance) — the latter tests candlestick patterns directly on the DJIA and
  finds no value — both show how candlestick-rule "edges" evaporate out of sample. Sullivan,
  Timmermann & White (1999), *"Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap"* (Journal of Finance), is the canonical reality-check warning our placebo embodies.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.hac_t`](../dark_cloud_piercing/strategy.py) and
  [`quantlab.analytics`](../../../quantlab/analytics.py).
- **Label-shuffle / Reality-Check placebo.** White (2000), *"A Reality Check for Data Snooping"*
  (Econometrica), and Politis & Romano (1994), *"The Stationary Bootstrap"* (JASA) — the spirit of
  [`strategy.placebo_pvalue`](../dark_cloud_piercing/strategy.py).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of freeze
  and content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`, `auto_adjust=True`), 2005-01-03 → 2026-06-18,
  across a fixed 30-name basket (29 liquid US large-caps + SPY). The offline reproducible core and
  the synthetic positive control run on the deterministic
  [`data.synthetic_panel`](../dark_cloud_piercing/data.py) generator, never the network. The
  headline run is pinned with an as-of date and a basket content fingerprint (see
  [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 402 — Engulfing Pattern](../../402-engulfing-pattern/)**: the twins' bigger sibling (a
  full real-body engulfing); same basket, same backwards result — the natural companion.
- **[Study 403 — Hammer / Hanging-Man](../../403-hammer-hanging-man/)**: single-candle reversal lore,
  same honest treatment.
- **[Study 404 — Shooting-Star](../../404-shooting-star/)** and
  **[Study 405 — Doji-Reversal](../../405-doji-reversal/)**: more single-candle reversal patterns.
- **[Study 186 — Morning-Star](../../186-morning-star/)** and
  **[Study 187 — Three-Soldiers](../../187-three-soldiers/)**: three-candle reversal/continuation
  patterns — same candlestick family.
- **[Study 178 — CCI](../../178-cci/)** and
  **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: oscillator mean-reversion
  rules — the "buy the dip on a technical extreme" cousins.
