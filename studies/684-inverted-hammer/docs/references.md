# References & literature map — Study 684 (Inverted Hammer)

## The claim under test

- **The folk recipe.** The inverted hammer is a one-bar bullish reversal: a small real body
  at the **bottom** of the session range, a long **upper** shadow (at least ~2× the body),
  little or no lower shadow, appearing **after a downtrend**. The narrative — buyers tested
  higher during the session and, even though sellers clawed most of it back by the close,
  the mere fact that demand showed up signals the sellers are running out of ammunition —
  is taught in every candlestick primer as the bullish twin of the shooting star. We
  steelman it as: *the conditional forward (long) return after an inverted hammer, net of
  costs, exceeds buying a random day in the same name.*
- **Steve Nison** popularised Japanese candlesticks for Western markets in *Japanese
  Candlestick Charting Techniques* (1991) and *Beyond Candlesticks* (1994); the inverted
  hammer (and its bearish mirror the shooting star) is a staple of that taxonomy. The
  technique traces to **Munehisa Homma**'s 18th-century rice-trading methods.

## Why the steelman is *almost* coherent — and where it breaks

- **Intraday rejection is a real microstructure event.** A long upper wick genuinely
  encodes a within-session fact: buyers pushed price up and sellers took it back. The leap
  of faith is that this *one-day* footprint forecasts the *next several days* — the same
  leap every single-candle reversal pattern makes (see sibling studies below).
- **The systematic evidence is largely negative.** Marshall, Young & Rose (2006),
  *"Candlestick Technical Trading Strategies: Can They Create Value for Investors?"*
  (Journal of Banking & Finance), test the full candlestick zoo on Dow stocks and find
  **no value** once the data-snooping inherent in the pattern menu is accounted for.
  Horton (2009), *"Stars, Crows, and Doji: The Use of Candlesticks in Stock Selection"*
  (Quarterly Review of Economics and Finance), reaches the same null. Caginalp & Laurent
  (1998), *"The Predictive Power of Price Patterns"* (Applied Mathematical Finance), found
  a modest, era-dependent signal in some candle configurations — but on an earlier, thinner
  tape than the one used here.
- **Multiple testing inside one study.** With 26 names and four horizons, a lone |t| > 2 is
  expected by chance even in a null world. Sullivan, Timmermann & White (1999),
  *"Data-Snooping, Technical Trading Rule Performance, and the Bootstrap"* (Journal of
  Finance), is the canonical warning; this study reports a **Bonferroni**-adjusted p-value
  across the four-horizon family (not just the best horizon) and a per-name breakdown that
  makes any snooping visible rather than cherry-picking the winner.

## The failure mode exposed

- **An upper wick doesn't reliably distinguish exhaustion from continuation.** On a
  declining large-cap, a long upper shadow can equally mark a failed intraday bounce that
  the *next* session resumes selling into — the opposite of the "sellers are exhausted"
  story. Fama (1970) weak-form efficiency predicts exactly that a visible one-bar shape
  carries no exploitable forecast for liquid names; Lo, Mamaysky & Wang (2000),
  *"Foundations of Technical Analysis"* (Journal of Finance), find chart patterns carry
  *some* statistical information but rarely survive as a tradable edge net of costs.
- **Survivorship cuts the "generous" way here.** The basket is names still trading in
  2026 — a bullish claim (buy after a downtrend) benefits from the fact that survivors are,
  by construction, names that *did* recover from their dips. A null/weak result on this
  basket is therefore the *conservative* reading, not a stacked deck against the claim.

## Method lineage (the desk's shared engine)

- **HAC / Newey–West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  used for the overlapping-window inference in
  [`strategy.hac_t`](../inverted_hammer/strategy.py).
- **Label-shuffle / permutation placebo.** The per-name shuffle in
  [`strategy.placebo_pvalue`](../inverted_hammer/strategy.py) is a randomisation test in the
  spirit of Brock, Lakonishok & LeBaron (1992), *"Simple Technical Trading Rules and the
  Stochastic Properties of Stock Returns"* (Journal of Finance).
- **Bonferroni correction.** Testing the same claim across four horizons (1/3/5/10 days) at
  once and quoting the best is a textbook multiple-comparisons snoop; the family-wise
  correction (`strategy.bonferroni`) is reported alongside the raw placebo *p* for every
  horizon, not just the winner — the same convention as sibling study
  [186-morning-star](../../186-morning-star/).
- **Reproducibility stamp.** As-of freeze + content fingerprint, `quantlab/repro.py`
  ([`data_stamp`](../../../quantlab/repro.py)), plus a panel-wide content fingerprint over
  the whole basket (`data.fingerprint`).

## Data sources used here

- **Yahoo! Finance daily OHLC** (via `yfinance`, `auto_adjust=False`), full available
  history across the same 26 US large-caps + SPY used by the sibling candlestick studies.
  The offline reproducible core and the notebooks run on cached parquets; the synthetic
  positive control ([`data.synthetic_panel`](../inverted_hammer/data.py)) is deterministic
  and never touches the network. Each headline run is pinned with an as-of date (the last
  complete calendar month) and a content fingerprint (see [`docs/results.md`](results.md)).

## Related desk studies (the dedup map — what this study is NOT)

- **[Study 403 — Hammer & Hanging Man](../../403-hammer-hanging-man/)** — the **lower**-wick
  mirror-image geometry (small body at the *top* of the range, long *lower* shadow), split
  the same way by trend into the bullish hammer and the bearish hanging man. Different
  candle shape entirely; this study's inverted hammer is the *upper*-wick shape.
- **[Study 404 — Shooting Star](../../404-shooting-star/)** — the exact **same** one-bar
  geometry as this study (long upper wick, small body, little lower wick), but split by an
  **uptrend** into the bearish shooting-star claim, traded short. This is this study's
  direct look-alike: same detector, opposite trend context, opposite trade direction. 404's
  code even names the `"invhammer"` side in its `strategy.py`, but never measures it — this
  study is the dedicated, from-first-principles teardown of exactly that side, with its own
  package, its own inference run, and its own verdict.
- **[Study 405 — Doji Reversal](../../405-doji-reversal/)** — a *flat*-body candle (the
  session opens and closes at nearly the same price, wicks on either side largely
  irrelevant to the classification) tested as a reversal marker. The inverted hammer
  explicitly requires a **directional** body near one end of the range plus a long wick on
  the *other* end — a doji, by definition, has no meaningful body to place, and this
  study's detector excludes near-zero bodies via a body floor precisely so the two patterns
  don't bleed into each other.
- **[Study 186 — Morning-Star](../../186-morning-star/)** — a **three**-candle bullish
  reversal (large bearish, small indecision star gapping below, large bullish recovery).
  The inverted hammer is a single bar; morning-star's "star" candle is one ingredient in a
  three-bar sequence, evaluated only in the context of its neighbours, not on its own
  one-bar geometry.

None of the siblings test the **single-bar, long-upper-wick, post-downtrend, bullish**
claim in isolation — that is this study's own axis.
