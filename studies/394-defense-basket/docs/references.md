# References & literature map — Study 394 (Defense-Basket)

## The claim under test

- **The folklore.** A perennial financial-media reflex: when war, invasion or a rearmament
  cycle hits the tape, "defense stocks rally" — Lockheed Martin (LMT), RTX (Raytheon),
  Northrop Grumman (NOC), General Dynamics (GD) and the iShares U.S. Aerospace & Defense ETF
  (ITA) are pitched as the obvious "buy the conflict" trade. The pitch surges around every
  shock (e.g. the post-2022-Ukraine "defense supercycle" coverage, and the Israel–Iran 2024
  escalations), and is repeated as a *reliable, tradable reflex*, not just a one-off.
- **Why it is intuitive.** Conflict → higher defense budgets → more orders for the primes →
  higher earnings. The causal chain is real over *years* (procurement cycles, multi-year
  appropriations). The leap the folklore makes is from that slow fundamental story to a *fast,
  reliable, same-month stock rally on the headline* — a very different, testable claim.

## Why this is an event study (and the pitfalls that come with it)

- **Event-study method.** The canonical framework for measuring abnormal returns around dated
  events: A. Craig MacKinlay (1997), *Event Studies in Economics and Finance* (Journal of
  Economic Literature); Brown & Warner (1985), *Using daily stock returns: the case of event
  studies* (Journal of Financial Economics). We use the simplest abnormal-return proxy — return
  **in excess of the market (SPY)** over a fixed post-event window — entered one day after the
  shock to avoid look-ahead.
- **Anticipation / "buy the rumour, sell the news."** Markets price *anticipated* conflict
  before the headline prints, so the abnormal return *on and after* the event date can be flat
  or negative even when the fundamental thesis is correct — a well-documented event-study
  caveat. Our negative point estimate is consistent with the shock being largely priced in by
  the time a retail "buy defense" trade could be placed.
- **War and asset prices.** Brune, Hens, Rieger & Wang (2015) and the broader "geopolitical
  risk" literature (Caldara & Iacoviello, 2022, *Measuring Geopolitical Risk*, American Economic
  Review) find geopolitical shocks move *aggregate* risk premia and commodities far more
  reliably than they hand a clean, tradable rally to a single equity sector.

## Why ~20 events cannot be a reliable edge — the statistics

- **Small-sample inference / power.** With *k* ≈ 20 events, the standard error of a
  conditional-mean estimate is large; a few-point average excess cannot be distinguished from
  luck. We test the event-window mean against zero with a one-sample *t*, against the
  unconditional base rate, and — because daily in-window observations overlap and
  autocorrelate — with a **Newey-West / HAC** *t* (Newey & West, 1987, *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica). Because *k* is tiny we add a **placebo / randomization test** — the share of
  random same-size date draws whose mean event-window excess beats the shock set (Fisher's
  randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **The vivid-event / availability illusion.** A couple of unforgettable rallies (Ukraine 2022,
  post-9/11) are mentally over-weighted, while the equally large *losers* (Desert Fox 1998, the
  2001 Afghanistan invasion, the 2024 Iran barrage) are forgotten — the availability heuristic
  (Tversky & Kahneman, 1973, *Availability: a heuristic for judging frequency and probability*).
- **Base rates.** Defense names carry a mild positive beta drift relative to SPY over the long
  run; the right comparison is the **excess over that unconditional drift**, not the raw sign —
  the base-rate fallacy (Kahneman & Tversky, 1973, *On the psychology of prediction*).
- **Selection on a famous trade.** A reflex that survives into media folklore is selected on its
  memorable wins; Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns* (Review
  of Financial Studies), formalises why an ex-post "everyone knows" rule needs a far higher bar
  than a naive *t*-stat.

## Method lineage (the study's engine)

- **Event-window excess + plain / HAC t.** [`strategy.event_excess`](../defense_basket/strategy.py),
  [`strategy.one_sample_t`](../defense_basket/strategy.py) and
  [`strategy.hac_t_daily`](../defense_basket/strategy.py) — the Signal-axis tests.
- **Placebo null.** [`strategy.placebo_pvalue`](../defense_basket/strategy.py) — 20,000 draws of
  random same-size date sets, sized to the event count.
- **Deterministic synthetic control.**
  [`data.synthetic_defense`](../defense_basket/data.py) injects a known number of shocks and an
  optional known window edge; the offline core runs with no network. The control confirms the
  detector is faithful *and* that ~20 events cannot reach significance unless the planted edge is
  implausibly large.
- **Costs with execution lag.** [`strategy.net_of_costs`](../defense_basket/strategy.py) enters one
  day after the signal (no look-ahead) and charges one round-trip on the long-basket / short-SPY
  pair.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + LMT, RTX, NOC, GD (equal-weight defense basket)
  and ITA (sector-ETF cross-check), 1995-01-04 → 2026-06-18, cached under
  `_cache/defense_prices.csv`. The 20 geopolitical-shock dates are hardcoded in
  [`defense_basket/data.py`](../defense_basket/data.py). All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- The desk's **thematic-basket** and **"buy the headline"** teardowns share this pathology: a
  fundamentally plausible story, a couple of vivid wins, and a sample too small (and too
  anticipated) to certify a tradable reflex. Defense-on-war is the geopolitical cousin of those.
