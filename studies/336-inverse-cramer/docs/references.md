# References & literature map — Study 336 (Inverse-Cramer)

## The claim under test

- **The "Inverse Cramer" meme.** The folk thesis that CNBC's *Mad Money* host Jim Cramer is
  so reliably wrong that doing the **opposite** of his on-air calls is an edge. The meme is
  tracked by accounts such as @CramerTracker and the (now-defunct) "Inverse Cramer Tracker
  ETF" **SJIM** (Tuttle Capital Management, launched 2023-03-02, *closed 2024-02-13* after
  failing to outperform). The testable hypothesis: over a forward window after a call, the
  fade (trade opposite the stated direction) earns a positive return distinguishable from a
  random-direction control. We test it on a **hardcoded, curated** table of notable calls and
  are explicit that the curation is the study's central confound.

- **SJIM, the real product.** The Inverse Cramer ETF's brief life (≈11 months, then closure)
  is itself the cleanest field evidence: a fund built on exactly this premise did not beat the
  market and was shut. The desk's job is to explain *why* the backward-looking meme looks so
  much better than the forward-looking fund.

## The bias that makes the meme look real — selection on the outcome

- **Selection / look-ahead in curated lists.** A list of "his worst calls" is sampled *on the
  dependent variable*: the calls are remembered precisely because they went badly. Leamer
  (1983), *Let's Take the Con Out of Econometrics* (American Economic Review), and the broader
  specification-search literature, are the canonical warnings. Our synthetic `selection_bias`
  knob plants exactly this: a coin-flip pundit whose published call list is filtered to his
  worst moments prints a fade at HAC *t* = +4 with **no** predictive content.

- **Multiple testing / data snooping.** Harvey, Liu & Zhu (2016), *…and the Cross-Section of
  Expected Returns* (Review of Financial Studies); White (2000), *A Reality Check for Data
  Snooping* (Econometrica). Choosing the forward horizon at which a selected sample turns
  significant (here, 63 days clears *t* = 2 while the pre-declared 21-day horizon does not) is
  the snooping these papers formalise.

## Are pundit / expert forecasts predictive at all?

- **Expert forecasts are weak.** Tetlock (2005), *Expert Political Judgment* — credentialed
  pundits barely beat chance and the most famous ones do worse. Bailey, Borwein, López de
  Prado & Zhu (2014), *Pseudo-Mathematics and Financial Charlatanism* (Notices of the AMS) —
  on the ease of manufacturing impressive backtests. These support a *prior* that a pundit's
  directional calls carry little forward information — which cuts **both** ways: if the calls
  are ~noise, *fading* them is also ~noise, not an edge.

- **Media-coverage and attention effects.** Engelberg & Parsons (2011) and Barber & Odean
  (2008), *All That Glitters* (Review of Financial Studies) — attention-driven buying around
  media mentions can create short-lived price pressure, a mechanism by which a high-profile
  endorsement near a peak is *followed* by mean reversion. This is the most charitable real
  channel for the fade, and why a forward window matters.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../inverse_cramer/strategy.py).
- **Block bootstrap CI.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA); the
  circular block bootstrap respects serial dependence the i.i.d. resample would destroy —
  [`strategy.block_bootstrap_ci`](../inverse_cramer/strategy.py).
- **Random-direction control.** The control discipline mirrors the desk's forward-return
  studies (e.g. Study 301 — Triple-RSI, Study 75 — Knee-Jerk): the same entry events, a coin
  for direction, so the fade is measured against its own null.

## Data sources used here

- **Yahoo! Finance daily bars** (via `quantlab.data`, `mode='total_return'`), pinned with an
  as-of date and per-ledger fingerprint (see [`docs/results.md`](results.md)). The offline
  reproducible core and the test-suite run on the deterministic
  [`data.synthetic_calls`](../inverse_cramer/data.py) generator and never touch the network.

## Related desk studies

- **[Study 252 — Google-Trends](../../252-google-trends/)** and the alt-data / sentiment lot
  (252–300): the same question — does a crowd/attention signal forecast returns? — with the
  same recurring answer (mostly Mirage).
- **[Study 301 — Triple-RSI](../../301-triple-rsi/)**: the desk's template for a forward-return
  ledger pinned against a random-direction control and a synthetic positive control; the win-
  rate-illusion discipline reused here.
- **[Study 291 — Doge-Tweets](../../291-doge-tweets/)**: a single-personality "loud voice
  moves the tape" study — the closest cousin to fading a loud pundit.
