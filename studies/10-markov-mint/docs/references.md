# Sources & literature map — Study 10 (Markov-Mint)

## The claim under test

- **Alex (@de1lymoon), *How To Use Markov Chains To Win Every Single Trade + [Quant
  Framework]*** (X / Twitter article, 26 May 2026; the author bills himself "Polymarket
  writer • Quant • AI maxi"). The post prescribes a five-step pipeline for prediction
  markets: (1) discretise a contract's price history into 10 states and build a transition
  matrix; (2) Monte-Carlo 10,000 resolution paths through it; (3) calibrate the raw
  probability against a favorite-longshot table; (4) size with quarter-Kelly; (5) execute as
  a maker. Its evidence is the framework itself plus an unverifiable empirical citation (see
  below); there is no out-of-sample test, no null, and the headline promise ("win every
  single trade") is not a probabilistic statement. This study ports the pipeline verbatim
  ([`markov_mint/markov.py`](../markov_mint/markov.py)) and runs it on markets whose truth we
  control.

- **The "Becker, 72.1 M trades, \$18.26 bn" statistic** quoted throughout the post (maker
  +1.12% / taker −1.12%, NO beats YES at 69/99 price levels, a per-level calibration table)
  has **no locatable primary source** as of this study's as-of date. We reproduce the
  calibration *table* verbatim (it is the only economic content the pipeline uses) but treat
  the surrounding figures as folklore, not data — hence this study tests the *method* on
  ground truth rather than trying to replicate an unverifiable tape.

## Why the method cannot add edge on an efficient market (the null)

- **Samuelson (1965), *Proof That Properly Anticipated Prices Fluctuate Randomly*** and
  **Fama (1970), *Efficient Capital Markets*.** A correctly priced contract's price is a
  martingale: `E[price_{t+1} | history_t] = price_t`. For a binary prediction market the
  price *is* the market's probability estimate. A transition matrix estimated from a
  martingale path reproduces a martingale, so a Monte-Carlo through it returns (up to
  finite-sample noise and the author's mid-grid thresholding artefact) the current price.
  There is, provably, no edge to extract — the study's efficient null is exactly this object,
  built as a Bayesian posterior (posteriors are martingales by the tower property).

- **Doob's martingale convergence / posterior-as-martingale.** The generator in
  [`markov_mint/data.py`](../markov_mint/data.py) builds the price as the exact posterior
  `P(YES | signals so far)` under a Gaussian signal model, which is a martingale by
  construction and converges toward the realized outcome — a textbook efficient market.

## The ingredients, and what the literature already says

- **Markov chains & Monte-Carlo.** Standard applied probability (Norris, *Markov Chains*,
  1997; Metropolis & Ulam (1949) for Monte-Carlo). Sound tools — the study does not dispute
  the machinery, only the claim that applying it to a martingale's price history manufactures
  information.

- **Favorite-longshot bias.** **Thaler & Ziemba (1988)**, **Snowberg & Wolfers (2010,
  *Explaining the Favorite-Longshot Bias*)**, and for prediction markets specifically **Page
  & Clemen (2013)**. A real, well-documented effect — longshots are over-bet — but small, and
  on retail venues largely consumed by spread and fees. The study's planted-wedge market
  reproduces it to size what is *recoverable*.

- **Kelly criterion.** **Kelly (1956)**; **Thorp (2006), *The Kelly Capital Growth Investment
  Criterion*.** Optimal sizing *given a real edge*; applied to a spurious (noise) edge it adds
  variance for zero expected log-growth — i.e. it bleeds. The study's P&L sim shows exactly
  this on the null.

- **Maker vs taker / microstructure.** **Glosten & Milgrom (1985)** and **Kyle (1985)** —
  market-makers earn the spread but bear adverse selection, so a flat "+1.12% per maker trade"
  with no conditioning on fills or adverse selection is not a free lunch. Out of scope for the
  offline core, flagged in the README's beat 6 and beat 7.

## Desk method

- **Newey & West (1987)** (HAC *t*) and the bootstrap — the inference backbone, here applied
  to the realized directional edge per trade. Shared engine:
  [`../../quantlab/`](../../quantlab/). House method: [`../../METHODOLOGY.md`](../../METHODOLOGY.md).

## Related studies in this repo

- **[Study 04 — Social-Oracle](../../04-social-oracle/)** — another viral retail signal sold
  on a framework rather than a null; verdict NONE / MIRAGE.
- **[Study 06 — Clockwork-Vol](../../06-clockwork-vol/)** — "cycles" that turn out to be
  shapes in red noise; same lesson that a method can manufacture structure where none exists.
- **[Study 07 — Coiled-Spring](../../07-coiled-spring/)** — a trading-book rule sold on
  cherry-picked winners; same beat-3 "announce the falsification first" discipline.
