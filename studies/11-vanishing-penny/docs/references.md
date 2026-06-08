# References & literature map — Study 11 (Vanishing-Penny)

## The claim under test

- **The viral thread.** *"The exact maths that pulled \$40,000,000 out of Polymarket
  (complete roadmap)"*, @robrtcode, May 2026 (~2.9M views).
  <https://x.com/robrtcode>. The steelman: single-condition `YES+NO ≠ $1` plus combinatorial
  arbitrage, Bregman projection for the optimal trade, Frank–Wolfe to make it tractable,
  same-block execution — extracted \$39.7M Apr-2024→Apr-2025, top wallet \$2.0M over 4,049
  trades. Ends in a wallet-connect "airdrop" call to action (the funnel this study keeps in
  view). We test the one thing the thread asserts without proof: that a *retail* trader can
  capture any of it.

## The primary source the thread cites

- **Saguillo, Ghafouri, Kiffer & Suarez-Tangil (2025).** *Unravelling the Probabilistic
  Forest: Arbitrage in Prediction Markets.* arXiv:2508.03474.
  <https://arxiv.org/abs/2508.03474>. The real paper behind the \$40M figure. Distinguishes
  **market-rebalancing** arbitrage (within a single market — the `YES+NO` class this study
  measures, ~\$10.6M) from **combinatorial** arbitrage (logical dependencies across markets,
  ~\$29M). Analyses on-chain order-book data and ~86M transactions; reports single-condition
  fill rates ~87% vs combinatorial ~45%. Note: the web abstract does **not** mention Bregman
  projection or Frank–Wolfe — those are grafted on from the foundation paper below, a small
  fidelity gap between the thread and its own citation.

- **Frank–Wolfe / Bregman projection foundation.** arXiv:1606.02825.
  <https://arxiv.org/abs/1606.02825>. The optimisation machinery (projection onto the
  arbitrage-free polytope via a sequence of linear programs) the thread attributes to the
  Polymarket result; it is a general method paper, not a Polymarket study.

## Data sources used here

- **Polymarket CLOB — price history.** `GET https://clob.polymarket.com/prices-history`
  (`market`=token id, `startTs`, `endTs`, `fidelity` in minutes). Finest fidelity ~1 minute;
  a `fidelity=1` window is capped near one day, so [`data.fetch_prices_history`](../prediction_arb/data.py)
  pulls long spans in daily chunks. The endpoint's **order-book snapshots froze ~2026-02-20**,
  so depth (capacity) is only reconstructable before then — the structural reason this study
  measures an *upper bound* on the half-life, not the true sub-block close.
- **Polymarket Gamma API — market discovery.** `GET https://gamma-api.polymarket.com/markets`
  (`closed=true`, ordered by recency, `volume_num_min`), which carries each market's
  `clobTokenIds` (the YES/NO token pair). Used by `verify_real.py --discover` to build the
  committed [`markets_manifest.json`](markets_manifest.json).
- **On-chain (not used here, beat-7 lead).** Polygon `OrderFilled` contract events, the
  paper's own substrate, would let a contributor reconstruct the gap at **sub-second**
  resolution and measure the half-life directly rather than bounding it.

## Method lineage (the desk's shared engine)

- **First-passage / half-life framing.** The study measures the decay time of a mispricing,
  not a return — the relevant quantity for a risk-free arb is *how long it lives*, estimated
  two ways (empirical median time-to-half; pooled log-linear decay fit) that must agree if the
  close is exponential.
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  freeze and content fingerprint every headline run carries, so a rerun that matches the
  fingerprint holds the exact data behind the verdict.

## Related desk studies

- **Study 04 — Social-Oracle** (WSB sentiment → NONE / MIRAGE): the other study where a
  loud, viral claim collapses on contact with real data.
- **Study 05 — Twin-Spread** (pairs trading, decayed): the alpha-decay sibling — there the
  *signal* dies; here the signal is bulletproof and only the *reachability* dies.
