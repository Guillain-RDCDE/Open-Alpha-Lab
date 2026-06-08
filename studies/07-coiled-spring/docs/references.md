# Sources & literature map — Study 07 (Coiled-Spring)

## The claim under test

- **Jayesh Shah — *Trade the 20 EMA: How to find explosive short-term moves***
  (self-published, Amazon). The book this study tests. It prescribes the three-step
  "20 EMA pivot breakout": (1) a stock below its 20-EMA breaks above and forms a pivot
  high; (2) the pullback **holds the 20-EMA in close**; (3) buy the pivot breakout on
  **≥ 2× the prior month's average volume**, stop one tick under the breakout bar, and
  trail out. Its evidence is a handful of hand-picked Indian-market winners (UCOBANK,
  SKIPPER, FACT, …) — no losers, no costs, no out-of-sample test. That selection is
  precisely what this study replaces with a full-universe, costed backtest.
  *(Source PDF was a Turkish translation circulated via forexgercekleri.xyz / @eseckal;
  the strategy is the author's, market-agnostic by construction.)*

## What the academic literature already says about the ingredients

The rule is a **volume-confirmed price breakout / short-horizon momentum** trade dressed
in a moving-average filter. Each ingredient has a mature literature:

- **Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*, J. Finance.**
  The canonical momentum result (3–12 month formation). The book's horizon is far shorter
  (days), which is closer to short-term reversal territory than to classic momentum.
- **Lo, Mamaysky & Wang (2000), *Foundations of Technical Analysis*, J. Finance.** A formal
  test of chart patterns (including support/resistance and breakouts) with kernel
  smoothing; finds *some* incremental information in a few patterns, modest and fragile.
- **Brock, Lakonishok & LeBaron (1992), *Simple Technical Trading Rules…*, J. Finance** and
  **Sullivan, Timmermann & White (1999), *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap*, J. Finance.** BLL found moving-average rules "worked";
  STW showed that once you correct for the *universe of rules searched* (White's Reality
  Check), the apparent edge largely evaporates. The cautionary tale for any single TA rule.
- **Marshall, Cahan & Cahan (2008), *Does intraday technical analysis in the U.S. equity
  market have value?*** — generally "no" after costs. Representative of the modern verdict
  on retail TA rules.
- **Zakamulin (2014), *The real-life performance of market timing with moving-average and
  time-series momentum rules*.** Moving-average timing looks far weaker once realistic
  costs, look-ahead-free signals and transaction frictions are imposed.

## Desk method

- **White (2000), *A Reality Check for Data Snooping*, Econometrica** and **Newey & West
  (1987)** — the inference backbone (HAC *t*, bootstrap) the desk uses, here applied to the
  *excess* of a breakout entry over a same-stock random entry.
- House methodology: [`../../METHODOLOGY.md`](../../METHODOLOGY.md). Shared engine:
  [`../../quantlab/`](../../quantlab/).

## Related studies in this repo

- **[Study 02 — Falling-Knife](../../02-falling-knife/)** — the mirror trade (buying sharp
  *drops* rather than *breakouts*); same "does a price trigger carry information?" question.
- **[Study 04 — Social-Oracle](../../04-social-oracle/)** — another retail signal sold on
  cherry-picked winners; verdict NONE / MIRAGE. Shares this study's cached universe.
- **[Study 05 — Twin-Spread](../../05-twin-spread/)** — a textbook rule tested for decay,
  same inference and capacity machinery.
