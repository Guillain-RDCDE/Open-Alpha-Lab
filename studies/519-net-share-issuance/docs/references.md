# References & literature map — Study 519 (Net-Share-Issuance)

## The claim under test

- **The factor.** A firm's **net change in shares outstanding** predicts its cross-sectional
  return: firms that **issue** shares (dilute — via secondary offerings, equity-funded M&A,
  generous stock-based compensation) **underperform**, while firms that **buy back** (shrink the
  share count) **outperform**. The signal is the *realised* net change in the split-adjusted
  share count, **not** an announcement — which is precisely what separates this study from the
  event-study cousin below.

## What the academic literature actually says

- **The canonical paper.** Pontiff & Woodgate (2008), *Share Issuance and Cross-Sectional
  Returns*, **Journal of Finance** 63(2). They show that a firm's net share issuance over the
  prior year is a robust negative predictor of subsequent returns in the post-1970 US
  cross-section — issuers underperform, repurchasers outperform — and that issuance subsumes much
  of the size, book-to-market and momentum effects once it is in the model. This is the factor
  this study replicates.
- **The composite-issuance formulation.** Daniel & Titman (2006), *Market Reactions to Tangible
  and Intangible Information*, **Journal of Finance** 61(4), define a **composite issuance**
  measure (the part of a firm's market-cap growth *not* explained by its stock return — i.e. the
  net new equity raised) and show it carries the return predictability, framing the effect as the
  market misweighting **intangible** information. Our split-adjusted share-count change is the
  simple, transparent version of the same idea.
- **The buyback half.** Ikenberry, Lakonishok & Vermaelen (1995), *Market Underreaction to Open
  Market Share Repurchases*, **Journal of Financial Economics** 39, document the long-run drift
  on the *repurchase* side. The issuance factor stitches both halves into one continuous axis:
  dilution at one end, buybacks at the other.
- **Why a small survivor replication is expected to fail.** McLean & Pontiff (2016), *Does
  Academic Research Destroy Stock Return Predictability?*, **Journal of Finance**, show published
  anomalies (issuance among them) decay sharply out-of-sample and post-publication. Harvey, Liu &
  Zhu (2016), *…and the Cross-Section of Expected Returns*, **Review of Financial Studies**, lay
  out the multiple-testing / selection bar a factor must clear. And the original effect lives in
  the **full** cross-section of *thousands* of names (including the small, heavily-diluting firms
  that disappear) with a proper factor model — none of which a 40-name large-cap survivor basket
  reproduces.

## Distinct from the event-study cousin

- **[Study 368 — Buyback-Drift](../../368-buyback-drift/)** times discrete buyback
  **authorization announcements** as an event study (does a stock drift up for months *after the
  press release*?). Study 519 is the **realised composite-issuance factor**: it ignores
  announcements entirely and measures the *actual net change in the share count* (dilution
  included) as a cross-sectional sort. The two ask different questions — one about a headline
  event, one about a balance-sheet quantity — which is why both can live on the bench.

## Why our sample is a fixed survivor basket — and what that costs us

- **No free point-in-time issuance universe.** A clean, point-in-time, survivorship-free panel of
  net share issuance for the whole US cross-section is a CRSP/Compustat product, not a free
  yfinance feed. We therefore fix a **transparent 40-name large-cap survivor basket** and pull
  each name's split-adjusted shares outstanding (`get_shares_full` + the firm's split history) —
  and we **name the survivorship on the Signal axis**. The statistical lesson is robust to the
  exact names: ~40 survivors over 9 annual rebalances cannot certify a few-percent annual spread.
- **The split trap, handled.** Raw shares-outstanding jumps discontinuously on a stock split
  (Apple's 2020 4-for-1 quadruples the raw count overnight with *zero* real issuance). We
  multiply every pre-split observation by the split ratio so the series is on a single
  post-split basis — leaving only **true** issuance/buyback in the level. Without this, every
  splitter would look like a massive issuer and the sort would be garbage.

## How we test it — the statistics

- **One-sample t.** The annual long-short mean against zero
  ([`strategy.one_sample_t`](../net_share_issuance/strategy.py)) — the Signal-axis bar is
  ∣t∣ ≥ 2.
- **Label-shuffle placebo.** [`strategy.placebo_pvalue`](../net_share_issuance/strategy.py)
  permutes, within each formation year, which name carries which issuance value (destroying the
  issuance→return link while preserving both marginals), rebuilds the long-short, and reports
  P[shuffled L−S ≥ real]. This is the strict cross-sectional null: could a random relabelling of
  the signal have produced this spread by luck? (Fisher randomization logic; Efron & Tibshirani,
  1993, *An Introduction to the Bootstrap*.)
- **Seed-robust synthetic control.**
  [`data.synthetic_panel`](../net_share_issuance/data.py) plants a known issuance edge; the
  control ([`strategy.synthetic_control`](../net_share_issuance/strategy.py)) averages the
  one-sample *t* over **25** seeds — never a single lucky RNG draw — confirming the engine
  recovers a planted edge and refuses to manufacture significance at edge = 0.

## Data sources used here

- **yfinance** daily adjusted closes and `Ticker.get_shares_full` shares-outstanding history for
  the 40-name basket, split-adjusted with each firm's split history, resampled to year-ends and
  cached under `_cache/`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 368 — Buyback-Drift](../../368-buyback-drift/)** — the buyback-*announcement* event
  study (the discrete-event cousin of this realised-quantity factor).
- **[Study 363 — PEAD-Drift](../../363-pead-drift/)** and
  **[Study 238 — Betting-Against-Beta](../../238-betting-against-beta/)** — the desk's other
  honest small-basket factor replications, with the same small-sample / survivorship texture.
