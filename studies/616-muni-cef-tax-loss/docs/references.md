# References & literature map — Study 616 (Muni-CEF-Tax-Loss)

## The claim under test

- **The seminal paper.** Laura T. Starks, Li Yong & Lu Zheng, *Tax-Loss Selling and the January
  Effect: Evidence from Municipal Bond Closed-End Funds* (2006, **Journal of Finance** 61(6),
  3049–3067). https://doi.org/10.1111/j.1540-6261.2006.01011.x — Muni CEFs are the cleanest
  laboratory for the tax-loss-selling hypothesis because their holders are almost exclusively
  **taxable retail investors** (the tax-exempt coupon is worthless to institutions and
  tax-deferred accounts). Starks-Yong-Zheng document abnormally high muni-CEF returns in early
  January, driven by the funds whose *year-end* sellers had losses to harvest — a **flow**
  effect on the discount, not a fundamentals effect on the NAV.
- **The mechanism.** Taxable holders sell losers in December to realise capital losses against
  the tax bill; in a thin, retail-held vehicle that selling pressure lands on the **price**
  while the NAV (a muni-bond portfolio) barely moves — the discount widens. The sellers (or new
  bargain-hunters) come back in January and the discount snaps shut. Ritter (1988, *The Buying
  and Selling Behavior of Individual Investors at the Turn of the Year*, JF 43(3), 701–717) is
  the parking-the-proceeds template; Sias & Starks (1997, JFE) separate the individual-investor
  component of the turn-of-year seasonal.

## The CEF backdrop

- Malkiel (1977, JF), Lee, Shleifer & Thaler (1991, *Investor Sentiment and the Closed-End Fund
  Puzzle*, JF 46(1), 75–109) — why CEF prices detach from NAV at all, and why retail sentiment
  is the marginal price-setter. Pontiff (1996, *Costly Arbitrage: Evidence from Closed-End
  Funds*, QJE 111(4), 1135–1151) — why the mispricing isn't arbitraged flat: no share creation,
  thin borrow, idiosyncratic risk.
- **Post-publication decay.** McLean & Pontiff (2016, *Does Academic Research Destroy Stock
  Return Predictability?*, JF 71(1)) — the anomaly was published in 2006; our sub-period split
  (January excess +327 bps pre-2017 vs +41 bps after, on MUB) is exactly the decay shape they
  predict, and it drives the Tradability stamp.

## What we measure, and the proxy's honesty

- **Discount-motion proxy.** Per-fund daily NAV isn't on yfinance, so we use the fund's monthly
  **total return minus the benchmark's** (MUB, an ETF that prices at NAV; VWLTX, a NAV-priced
  mutual fund, for the pre-2007 extension). A muni CEF's NAV is a (levered) muni portfolio, so
  a negative excess month = the *price* lagging the muni market = discount widening — the same
  proxy logic as study [367-closed-end-fund-discount](../367-closed-end-fund-discount/),
  labelled a **proxy** throughout. Leverage makes the CEF leg noisier than the benchmark but has
  no December/January calendar of its own.
- **Total-return both legs.** yfinance auto-adjusted closes on fund and benchmark alike, so the
  CEFs' fat monthly distributions never pollute the seasonal comparison.
- **Per-winter unit.** All muni CEFs move together in a given January (a common discount
  factor), so pooled fund-months massively overstate n. The primary test equal-weights the
  panel into **one December and one January observation per winter** and runs a one-sample *t*
  across non-overlapping winters (lag-1 autocorrelation reported; Welch 1947 for group splits).
- **Exact seasonal placebo.** Instead of a Monte-Carlo shuffle we enumerate **all 132 ordered
  pairs of distinct calendar months** and rank the Dec→Jan contrast among them — deterministic,
  no RNG, minimum attainable *p* = 1/132 ≈ 0.0076 (Fisher's randomisation logic, exhaustive).

## Method lineage (the desk's shared engine)

- **Excess panel / proxy.** [`data.monthly_excess`](../muni_cef_tax_loss/data.py) — fund TR
  minus NAV-priced benchmark TR, per month.
- **Per-winter basket + one-sample t.** [`strategy.winter_table`](../muni_cef_tax_loss/strategy.py)
  and [`strategy.winter_stats`](../muni_cef_tax_loss/strategy.py).
- **Exact 132-pair placebo.** [`strategy.exact_pair_placebo`](../muni_cef_tax_loss/strategy.py).
- **Costs.** [`strategy.january_swap`](../muni_cef_tax_loss/strategy.py) — 4 one-way legs ×
  cost × NAV per winter.
- **Third axis.** [`strategy.dec15_vs_jan15`](../muni_cef_tax_loss/strategy.py) — paired Dec-15
  vs Jan-15 entries, common end-February exit.
- **Deterministic synthetic control.** [`data.synthetic_excess`](../muni_cef_tax_loss/data.py)
  with plantable December-dump / January-snap knobs; zero knobs must not fire.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes: 12 seasoned national muni CEFs (NEA, NVG,
  NZF, NUV, NXP, BTT, MYI, MQY, MHD, EIM, VMO, MMU — Nuveen, BlackRock, Eaton Vance, Invesco,
  Western Asset sponsors) + **MUB** (iShares National Muni Bond ETF, listed 2007-09) +
  **VWLTX** (Vanguard Long-Term Tax-Exempt), 2000-01-03 → 2026-06-30, cached under
  `_cache/mct_prices.csv`. All headline numbers are pinned in [`docs/results.md`](results.md)
  and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map)

- **[367-closed-end-fund-discount](../367-closed-end-fund-discount/) (Real ·
  Fragile)** — the **level** edge: buy the widest-discount CEFs and wait for mean reversion,
  cross-sectional, any month. This study is its **seasonal-flow cousin**: *when* (December →
  January) the muni-CEF discount predictably widens and snaps back, on a time-series calendar
  test — a different axis of the same discount machinery, kept deliberately distinct.
- [96-new-year-pop](../96-new-year-pop/) — the generic January effect on stocks;
  here the January effect appears in its sharpest documented habitat (taxable-retail-only
  vehicles), which is exactly Starks-Yong-Zheng's identification trick.
- [89-turn-of-the-month](../89-turn-of-the-month/) — the flow-driven calendar
  seasonal template.
