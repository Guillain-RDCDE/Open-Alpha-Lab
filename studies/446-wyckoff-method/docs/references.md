# References — Study 446 (Wyckoff Method)

## The claim and its source

- **Richard D. Wyckoff** (1873–1934), *Stock Market Technique* (1933) and the Wyckoff
  course materials — the original statement of the **accumulation → markup → distribution →
  markdown** market cycle, the "Composite Operator", the law of *supply and demand*, the law
  of *cause and effect*, and the law of *effort versus result* (price vs volume).
- **Hank Pruden**, *The Three Skills of Top Trading* (2007) — the modern academic
  systematisation of the Wyckoff method, the phase schematic (A–E), and the canonical events
  we mechanise: the **Spring / Shakeout** (accumulation), the **Upthrust After Distribution
  (UTAD)**, the Selling/Buying Climax, the Sign of Strength/Weakness, and the Last Point of
  Support/Supply.
- **David H. Weis**, *Trades About to Happen* (2013) and **Bruce Fraser** / Stockcharts'
  *Wyckoff Method* primer — contemporary practitioner statements of the Spring and Upthrust
  as the tradable entries inside a trading range, with volume as the confirming "tell".

The method is, by its own proponents' admission, an interpretive *framework*: the "correct"
phase labelling is only clear in hindsight and two analysts annotate the same chart
differently. There is no single falsifiable rule, so we test the **tightest mechanical
version** practitioners accept — a ZigZag-bracketed trading range plus the volume-confirmed
Spring and Upthrust.

## Why this is hard to test (and what the literature finds)

- **Lo, Mamaysky & Wang (2000)**, "Foundations of Technical Analysis", *Journal of Finance* —
  the standard reference for putting a *subjective* chart pattern (head-and-shoulders, etc.)
  through a mechanical, kernel-smoothed detector and a statistical test; the template for
  mechanising a "you know it when you see it" method like Wyckoff.
- **Park & Irwin (2007)**, "What do we know about the profitability of technical analysis?",
  *Journal of Economic Surveys* — the broad survey: most TA rules that survive in-sample fail
  out-of-sample or once data-snooping is corrected.
- **De Bondt & Thaler (1985)**, "Does the Stock Market Overreact?", *Journal of Finance* —
  the over-reaction / mean-reversion literature a working Spring (a "shakeout" that snaps
  back) would have to be an instance of, at the multi-week horizon.

## Shared-method citations (the desk's inference engine)

- **Newey & West (1987)** — heteroskedasticity-and-autocorrelation-consistent (HAC) standard
  errors; the `hac_t` used to certify (or deny) the **t ≥ 2** Signal bar.
- **White (2000)**, "A Reality Check for Data Snooping" — the data-snooping discipline behind
  the label-shuffle placebo: does the *event label* beat a coin placed at the same bars?
- **Politis & Romano (1994)** — the stationary/circular block bootstrap spirit behind testing
  a serially-dependent return series rather than assuming i.i.d. draws.

## Related desk studies

- [`../445-elliott-wave`](../445-elliott-wave/) — the sibling "irreducibly-subjective theory,
  test the tightest mechanical proxy" teardown (ZigZag + Fibonacci wave-3). Same shape, same
  None × Mirage verdict.
- [`../444-dow-theory`](../444-dow-theory/) — Dow Theory's higher-high/higher-low
  Industrials/Transports confirmation, another classic-theory mechanical proxy.
- [`../104-bollinger-reversion`](../104-bollinger-reversion/) — "price always returns to the
  bands": the same false-break / reversion intuition the Spring rests on, tested vs a
  random-day control.
- [`../363-pead-drift`](../363-pead-drift/) — the desk's gold-standard *real* event-drift
  study, for contrast: what a genuine, HAC-significant forward drift actually looks like.
