# References & literature map — Study 691 (Homing Pigeon)

## The claim under test

- **The folk recipe.** The homing pigeon is a two-bar bullish reversal: a large **down**
  (bearish) candle followed by a smaller **down** candle whose real body sits entirely
  *inside* the prior one, appearing after a **downtrend**. Both bodies are the *same*
  colour — the geometric cousin of the [harami](../../406-harami-pattern/) (whose two
  bodies are *opposite* colours). The name comes from the shrinking down-day "returning
  home" like a homing pigeon despite the market still nominally falling — read as sellers
  running out of conviction, a bottom, buy. It appears in Steve Nison's *Japanese
  Candlestick Charting Techniques* (1991) and Thomas Bulkowski's *Encyclopedia of
  Candlestick Charts* (2008) as a minor, low-frequency bullish reversal pattern. We
  steelman it as: *the conditional forward (long) return after a homing pigeon, net of
  costs, exceeds both (a) a random day in the same name and (b) buying any dip in the
  same downtrend without the specific two-bar shape.*
- **Bulkowski's own backtest** (thepatternsite.com / *Encyclopedia of Candlestick
  Charts*) ranks the homing pigeon among the *better-performing* bullish candle patterns
  in his proprietary sample — one of the few candlestick shapes with a non-trivial claimed
  edge in that literature, which is exactly why it earns a dedicated teardown here rather
  than being waved through with its zoo of siblings.

## Why the steelman is *almost* coherent — the real effect it leans on

- **A genuine, rarer signature than the harami.** Requiring *both* bars to be down days
  (not opposite colours) and the pair to sit inside a downtrend is a stricter filter than
  most candle rules — rarity is not itself evidence, but a tighter geometric definition at
  least reduces the multiple-comparisons exposure of a looser pattern menu.
- **Short-horizon reversal is a real, if weak, phenomenon.** Jegadeesh (1990) and Lehmann
  (1990) document short-term reversal at the individual-stock level; a shrinking-body
  pattern late in a slide is a plausible (if noisy) proxy for exhaustion of selling
  pressure, the same mechanism the [inverted hammer](../../684-inverted-hammer/)'s long
  upper wick claims to capture from a different angle.
- **Downtrend conditioning is not free of confounds.** Any bullish rule that only fires
  inside a downtrend inherits *some* of the market's well-documented short-horizon
  mean-reversion after a decline (De Bondt & Thaler 1985, *"Does the Stock Market
  Overreact?"*, Journal of Finance) — which is exactly why this study's alpha-vs-beta cut
  compares the pattern's return not just to the unconditional (any-day) base rate but to
  the return from buying *any* dip in the *same* downtrend, pattern or not.

## The failure mode probed (and only partly found)

- **Candlestick patterns under formal testing are usually null.** Marshall, Young & Rose
  (2006), *"Candlestick Technical Trading Strategies: Can They Create Value for
  Investors?"* (Journal of Banking & Finance), and Horton (2009), *"Stars, Crows, and
  Doji"* (Quarterly Review of Economics and Finance), find no value across the
  candlestick zoo on US large-caps — the prior this study starts from, and the reading
  every sibling candlestick study on this desk (403, 404, 406, 684, 186) has confirmed.
- **Multiple testing inside one study.** Four horizons at once is a textbook
  data-snooping exposure (Sullivan, Timmermann & White 1999, *"Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap"*, Journal of Finance); this study reports a
  **Bonferroni**-adjusted placebo *p* for the whole four-horizon family, not just the best
  horizon, following the same convention as [186-morning-star](../../186-morning-star/)
  and [684-inverted-hammer](../../684-inverted-hammer/).
- **Pooling many names can manufacture significance from a shared macro event.** If most
  "events" were really the same 2008/2020 crash week repeated 26 times, a naive pooled
  HAC *t* would overstate how much independent evidence exists. This study checks event
  dispersion explicitly (`strategy.event_clustering`) before trusting the pooled *t* — see
  `docs/results.md` for the count.
- **Pooled significance without per-name reproducibility is fragile.** Following the same
  audit as [684-inverted-hammer](../../684-inverted-hammer/), a per-name |HAC *t*| > 2
  count against its ~5%-of-26 chance baseline (~1.3 names) tests whether the pooled edge
  is a broad, reproducible tilt or an artefact of averaging many small, noisy samples.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.hac_t`](../homing_pigeon/strategy.py).
- **Label-shuffle / permutation placebo.** The per-name shuffle in
  [`strategy.placebo_pvalue`](../homing_pigeon/strategy.py), in the spirit of Brock,
  Lakonishok & LeBaron (1992), *"Simple Technical Trading Rules and the Stochastic
  Properties of Stock Returns"* (Journal of Finance).
- **Bonferroni correction.** Testing the same claim across four horizons (1/3/5/10 days)
  at once and quoting the best is a textbook multiple-comparisons snoop; the family-wise
  correction (`strategy.bonferroni`) is reported alongside the raw placebo *p* for every
  horizon — the same convention as sibling studies
  [186-morning-star](../../186-morning-star/) and
  [684-inverted-hammer](../../684-inverted-hammer/).
- **Reproducibility stamp.** As-of freeze + content fingerprint, `quantlab/repro.py`
  ([`data_stamp`](../../../quantlab/repro.py)), plus a panel-wide content fingerprint over
  the whole basket (`data.fingerprint`).

## Data sources used here

- **Yahoo! Finance daily OHLC** (via `yfinance`, `auto_adjust=False`), full available
  history across the same 26 US large-caps + SPY used by the sibling candlestick studies.
  The offline reproducible core and the notebooks run on cached parquets; the synthetic
  positive control ([`data.synthetic_panel`](../homing_pigeon/data.py)) is deterministic
  and never touches the network. Each headline run is pinned with an as-of date (the last
  complete calendar month) and a content fingerprint (see [`docs/results.md`](results.md)).

## Related desk studies (the dedup map — what this study is NOT)

- **[Study 406 — Harami Pattern](../../406-harami-pattern/)** — the *direct* geometric
  parent: the same "small body fully inside a larger prior body" containment rule, but
  with **opposite**-coloured bodies (a down day then an up day, or vice versa). The
  homing pigeon is the *same-coloured* special case, restricted to the bullish
  (down-then-down) direction and conditioned on a downtrend — 406 tests the harami
  two-directionally without a trend requirement baked into the detector itself; this
  study's detector *requires* the trend as part of the definition, and only ever trades
  long. Different geometry, different (narrower) claim, its own engine and verdict.
- **[Study 403 — Hammer & Hanging Man](../../403-hammer-hanging-man/)** — a **one-bar**
  shape (small body at the top of the range, long lower wick), split by trend into the
  bullish hammer / bearish hanging man. No second candle, no containment rule — a
  completely different detector testing a different (single-bar) floor story.
- **[Study 684 — Inverted Hammer](../../684-inverted-hammer/)** — also a **one-bar**
  bullish-after-downtrend floor claim (long *upper* wick this time), run on the identical
  26-name basket with the identical protocol (base rate, HAC *t*, label-shuffle placebo,
  Bonferroni, cost sweep, per-name breakdown, synthetic control). 684's inverted hammer
  came back a clean **None x Mirage**; this study's own alpha-vs-beta and per-name cuts
  test whether the homing pigeon's two-bar shape does any better — see
  [`docs/results.md`](results.md) for the side-by-side.
- **[Study 186 — Morning-Star](../../186-morning-star/)** — a **three**-candle bullish
  reversal (large bearish, small indecision star gapping below, large bullish recovery)
  tested against a random-day baseline with the same Bonferroni convention. Three bars,
  not two; a gap-down star, not a contained smaller body.

None of the siblings test the **two-bar, same-colour, contained, post-downtrend** claim in
isolation — that is this study's own axis.
