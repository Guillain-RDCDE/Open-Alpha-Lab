# References & literature map — Study 15 (Sigma-Sleight)

## The claim under test

- **The AdaptiveRSI framework.** *"RSI Beyond 70/30: The AdaptiveRSI Framework"* — AdaptiveRSI
  (@adaptiveRSI on TradingView), with the free *"RSI Manifesto v2.0"* 38-page download
  (<https://adaptiversi.gumroad.com>) and a YouTube channel. The steelman, taken at full strength:
  fixed 70/30 levels are **not length-aware** — RSI(2) at 70 and RSI(200) at 70 are not the same
  event — so thresholds should be defined as **standardised σ landmarks** in a logit-transformed
  RSI space and translated back into length-specific RSI levels. The framework's own arithmetic:
  RSI is logit-transformed and scaled by a `2/√(n−1)` length factor, under which **RSI(14) = 70
  maps to +1.53σ**. Five tools sit on this one idea (RSI-as-position, Adaptive Zones, RSI Candles,
  Chart Overlay, Rescaled RSI, Logit RSI). The post is careful to say the zones are **"not buy/sell
  signals,"** only a description of the RSI *environment* — the hedge this study keeps in view: we
  do not strawman it as an alpha engine; we test the one falsifiable thing it implies — that
  **length-aware σ-zones read oversold/overbought better than fixed 70/30.**

- **The tell in the framing.** Every adaptive zone is defined first in σ space, *then* "translated
  back" to RSI for the selected length. That translation, `σ = logit(RSI)·√(n−1)/2`, is a strictly
  **monotone** function of RSI at any fixed length — so for a given length a σ-zone is an
  order-preserving rename of a constant RSI level. The "adaptivity" is therefore purely
  *across* lengths (a per-length lookup table), computed once from `n`, with no dependence on
  regime or realised volatility despite the framework invoking Cardwell-style regimes. That is the
  structural fact the whole study operationalises.

## Why the steelman is *almost* right — the real thing underneath

- **Fixed 70/30 really is length-naive.** Short-period RSI swings violently across 0–100 and longer
  RSI compresses around 50 — this is arithmetic, not opinion. So the *motivation* is sound: a
  length-matched threshold is more sensible than one number for every length. The question the
  study isolates is whether the *specific σ-calibration* adds anything beyond "pick a sensible
  per-length number," and whether the σ machinery moves any signal at all within a length.
- **RSI mean reversion is a documented, modest effect.** Short-period RSI(2) oversold/overbought
  reversal is the Connors–Alvarez *Short Term Trading Strategies That Work* (2008) family; the
  genuine, small edge a mean-reversion rule harvests is the real grain of truth. We bake an
  analogous edge into the synthetic (an Ornstein–Uhlenbeck mean reversion) so the strategy is
  testing signal, not noise.
- **RSI as price position.** The framework's "RSI(14)=50 ⟺ price at its 14-period average" reading
  is a real, useful intuition (Wilder 1978, *New Concepts in Technical Trading Systems*, is the
  source of RSI itself; Cardwell, Brown and Connors are the cited regime/level adapters). None of
  that is disputed here — only the claim that the σ relabel turns it into a better signal.

## Method lineage (the desk's shared engine)

- **Monotone-transform invariance.** A strictly monotone map preserves order, hence preserves every
  threshold crossing and every rank statistic (so Spearman rank IC is invariant under it). This is
  the formal core of both the within-length crossing identity and the cross-length rescaling
  identity — implemented as measured facts (`max position diff = 0`, `rank-IC gap = 0`) in
  [`decompose.crossing_identity`](../sigma_sleight/decompose.py) and
  [`decompose.rescale_increment`](../sigma_sleight/decompose.py).
- **Re-optimised-constant control.** The honest way to ask "does the σ-calibration add value" is to
  compare it against the best *constant* threshold for the same length (an in-sample grid search) —
  if a plain constant matches it, the σ apparatus is doing no work.
  [`decompose.strategy_compare`](../sigma_sleight/decompose.py). The grid is in-sample, so the
  control is generous to the framework, and the multiple-testing it incurs is corrected on the real
  run with a Reality Check.
- **Data-snooping / Reality Check.** White (2000), *A Reality Check for Data Snooping*
  (Econometrica) — the correction the `reopt` grid's best-of-many Sharpe needs before it can be
  quoted as an edge. *(`quantlab/stats.py`)*
- **Robust inference & reproducibility.** Newey–West (HAC) errors for the strategy returns
  (`quantlab/analytics.py`), and the as-of freeze + content fingerprint every headline run carries
  ([`quantlab/repro.py`](../../../quantlab/repro.py)).

## Data sources used here

- **Yahoo! Finance daily closes** (via `yfinance`, `auto_adjust=True` — split/dividend-adjusted, a
  data-choice stated per the house rules). SPY & QQQ, ~10 years, for the horse race; daily fidelity
  is chosen because the AdaptiveRSI material is written for daily/weekly charts and the σ↔RSI
  identity is timeframe-agnostic. The offline core needs no network and the real window is pinned
  with `as_of` + a fingerprint.

## Related desk studies

- **Study 12 — Paper-Prophet** and **Study 14 — Gamma-Gospel**: the desk's other "is the fancy
  apparatus doing any work, or is it a relabel of something simpler?" teardowns — there an
  ARIMA+GARCH stack that's vol-targeting, and a GEX regime that's the VIX in a trenchcoat; here a
  σ-transform that's a monotone rename of a constant RSI threshold.
- **Study 08 — True-Strength**: the desk's other oscillator study — the same discipline of testing
  an indicator's *added* information against a plain baseline.
