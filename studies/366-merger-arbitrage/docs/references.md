# References & literature map — Study 366 (Merger-Arbitrage)

## The claim under test

- **The pitch (the merger-arb "coupon").** After an all-cash takeover is announced, the
  target stock trades a few percent **below** the per-share offer (the *arb spread*). The
  classic strategy is to **buy the target and hold to deal close**, collecting that spread —
  pitched as a steady, low-beta, "market-neutral" return uncorrelated with the broad market.
  The strategy is old and well-known: see Ivan Boesky's *Merger Mania* (1985) for the
  practitioner legend, and any modern event-driven hedge-fund overview for the "spread coupon"
  framing.
- **What the spread actually is.** The spread is **not** free money: it is **compensation for
  deal-break risk**. If the deal fails (antitrust block, financing collapse, shareholder vote,
  MAC clause), the target snaps back to its un-bid standalone level — a 20–40% loss in a day.
  A merger-arb position is therefore economically **short a deal-break put**: the arbitrageur
  *writes insurance* against deal failure and the spread is the premium.

## Why merger-arb is short volatility — the academic anchor

- **Mitchell & Pulvino (2001), *Characteristics of Risk and Return in Risk Arbitrage*
  (Journal of Finance).** The foundational study: a portfolio of risk-arbitrage positions
  behaves like a **written index put** — modest steady returns in calm markets, sharp losses
  in falling markets when deals break en masse. The return profile is **negatively skewed**
  and the unconditional alpha is far smaller than the raw spread suggests once the break tail
  is priced. This is the single most important reference for this study's thesis.
- **Baker & Savaşoglu (2002), *Limited Arbitrage in Mergers and Acquisitions* (Journal of
  Financial Economics).** Documents that arb returns compensate for **completion risk** and
  that the spread widens with deal-failure probability — the spread *is* the insurance premium,
  cross-sectionally.
- **Jindra & Walkling (2004)**, and **Branch & Yang (2003)**, on the cross-section of arb
  spreads and the relationship between spread size and break probability — bigger spreads sit
  on shakier deals, so the visible spread oversamples the deals most likely to blow up.

## Why a high win-rate is not an edge — the statistics

- **Negative skew / short-volatility payoffs.** A strategy that wins small most of the time and
  loses big rarely (selling insurance, "picking up pennies in front of a steamroller") has a
  high *win-rate* by construction, regardless of whether its expected value is positive. The
  right question is the **mean net of the tail**, not the hit-rate. Taleb's writing on
  negative-skew payoffs (*Fooled by Randomness*, 2001) is the popular statement; the formal
  one is any treatment of the **certainty-equivalent of a left-skewed payoff**.
- **Small-sample inference with a fat tail.** With ~20 deals and a left-skewed return
  distribution, the sample mean's standard error is dominated by the few break losses. We test
  the per-deal mean against zero with a **one-sample t** (sensitive to skew, so reported with
  caution) **and** a **nonparametric bootstrap** of the mean (Efron & Tibshirani, *An
  Introduction to the Bootstrap*, 1993) — the bootstrap CI is the honest small-sample, fat-tail
  instrument here.
- **Survivorship in event books.** Acquired targets **delist** and disappear from free price
  feeds; a book reconstructed only from names you can still pull would over-represent deals
  that *broke* (the target survived). We name this on the Signal axis and mark entry/exit from
  **documented closes** rather than a survivorship-filtered live feed.

## Method lineage (the desk's shared engine)

- **Per-deal realized arb return + bootstrap.**
  [`strategy.arb_returns`](../merger_arbitrage/strategy.py),
  [`strategy.bootstrap_mean`](../merger_arbitrage/strategy.py) — realized return per deal and
  the bootstrap sampling distribution of the book mean (placebo p = P[mean ≤ 0]).
- **Win-rate vs the break tail.** [`strategy.win_rate`](../merger_arbitrage/strategy.py) and
  [`strategy.skewness`](../merger_arbitrage/strategy.py) — the hit-rate next to the magnitude
  of the loser tail and the return skew.
- **Deterministic synthetic control.**
  [`data.synthetic_book`](../merger_arbitrage/data.py) builds a deal book with a **known break
  probability** and the spread **pinned to fair-insurance value**, plus a planted `edge` knob:
  with `edge = 0` the arb is a fair bet and the test must NOT reach significance; a large
  `edge` must light it up. The offline core runs with no network.
- **Execution lag + costs.** Entry is the close **one day after** the announcement (no
  look-ahead); [`strategy.net_of_costs`](../merger_arbitrage/strategy.py) charges a one-way
  cost on entry and exit (long-only target — no borrow).

## Data sources used here

- **Hardcoded deal book** of 20 real announced all-cash US M&A deals (Microsoft/Activision,
  Musk/Twitter, Pfizer/Seagen, Amgen/Horizon, Cisco/Splunk, … and the breaks JetBlue/Spirit,
  TD/First Horizon, Avangrid/PNM, MaxLinear/Silicon Motion) — offer, announce/resolve dates,
  outcome, and documented entry/pre-deal closes, in [`merger_arbitrage/data.py`](../merger_arbitrage/data.py).
- **yfinance** daily adjusted closes for the still-listed (broken-deal) targets (FHN, SIMO)
  confirm the documented marks; cached under `_cache/target_prices.csv`. All headline numbers
  are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **Event-driven / short-volatility family.** Merger-arb belongs with the bench's other
  "steady coupon that is really sold insurance" studies — covered-call writing, short-vol
  carry — where a high win-rate and a fat left tail are the recurring tell. The thesis is the
  same: a premium that compensates for a rare disaster is not a free lunch.
