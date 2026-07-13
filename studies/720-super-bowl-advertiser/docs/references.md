# References & literature map — Study 720 (Super-Bowl-Advertiser)

## The claim under test

- **The folklore.** A company that runs a **Super Bowl commercial** buys a burst of national
  attention — ~120 million viewers, a week of social buzz, a spike in brand searches — and its
  stock is said to **drift up in the days after the game**, the "big-ad signal". The believers
  turn it into a tradable February calendar: watch who advertises Sunday, buy the Monday open,
  ride the buzz for a week. The poster children run the other way too — the *cautionary* ones,
  **Pets.com** (the sock-puppet ad, Super Bowl 2000) and the dot-com class of 2000, are
  remembered precisely because they blew their capital on the ad and then went bust.
- **Where it's repeated.** Financial media every January/February, ad-industry round-ups (AdAge,
  USA Today Ad Meter), and — crucially — a **peer-reviewed finance paper** (below) that found a
  real post-game abnormal return. The believers' framing is one-legged: a positive **drift**
  after the game; we test that leg, plus the day-one **reaction**, on a modern representative
  table.

## The academic anchor — Super Bowl ads really did move stocks (in 2000–2004)

- **Fehle, Tsyplakov & Zdorovtsov (2005), *Can Companies Influence Investor Behavior through
  Advertising? Super Bowl Commercials and Stock Returns*, Financial Management 34(4), 31–61.**
  The canonical study: firms that advertised in the Super Bowl earned **significantly positive
  abnormal returns** in the days *following* the game (they interpret it as advertising drawing
  investor attention, à la Merton's investor-recognition hypothesis). This is the steelman —
  and our test asks whether the effect survives two decades later on a modern survivor tape (it
  does not: drift **+0.4%**, Welch *t* = **0.31**, placebo *p* = **0.64**).
- **Merton (1987), *A Simple Model of Capital Market Equilibrium with Incomplete Information*,
  Journal of Finance.** The investor-recognition mechanism the Super Bowl paper leans on — a
  stock investors *notice* commands a higher price / lower expected return. Advertising is a
  recognition shock.
- **Barber & Odean (2008), *All That Glitters: The Effect of Attention and News on the Buying
  Behavior of Individual and Institutional Investors*, Review of Financial Studies.** Retail
  buying concentrates in attention-grabbing stocks — the demand-side channel a Super Bowl ad
  would trigger, briefly, if the folklore held.
- **Grullon, Kanatas & Weston (2004), *Advertising, Breadth of Ownership, and Liquidity*,
  Review of Financial Studies.** Advertising is associated with more shareholders and better
  liquidity — a slow, structural effect, not the sharp tradable pop the calendar-trade needs.
- **Chemmanur & Yan (2019) and the broader "advertising and stock returns" literature** —
  evidence that ad spending predicts returns is mixed and horizon-dependent, and a single-event
  drift is far harder to pin down than an annual-spend cross-section.

## Why "the drift is tradable" is the hard part — and why our tape is biased *for* it

- **Survivorship.** The most spectacular Super Bowl advertisers **went to zero** — **Pets.com**,
  **Computer.com**, **Kozmo.com**, **LifeMinders.com**, the dot-com class of Super Bowl 2000, and
  **Just for Feet** (SB 1999) — and therefore have **no continuing yfinance series**; **Squarespace**
  (a Super Bowl regular) was **taken private** by Permira in Oct 2024 and also left the public
  tape. Our priced sample is the set of advertisers that *survived and stayed listed*, biased
  **toward** names that did not collapse — i.e. **for** the believers' "advertising pays" claim.
  A survivor-only drift near zero is therefore a conservative refutation, and we name the bias on
  the Signal axis (Brown, Goetzmann, Ibbotson & Ross, 1992, *Survivorship Bias in Performance
  Studies*, RFS).
- **Small-sample inference.** With ~32 documented advertiser-years, the cross-section of abnormal
  returns has a large standard error. We test the drift's mean against zero with a **one-sample t**
  (Welch, 1947) and, because the sample is tiny and fat-tailed, with a **placebo / randomisation
  null** — draw the same number of random non-event windows on the same tickers and ask how often
  chance matches the drift (Fisher's randomisation logic; Efron & Tibshirani, 1993, *An
  Introduction to the Bootstrap*). Event-study standard errors: MacKinlay (1997), *Event Studies
  in Economics and Finance*, Journal of Economic Literature.
- **Selection on famous anecdotes.** The sock puppet is remembered *because* it was extreme;
  building a "law" (positive *or* negative) from the loudest cases is the classic data-snooping
  trap (Harvey, Liu & Zhu, 2016, *…and the Cross-Section of Expected Returns*, RFS). A
  representative table of advertisers — not just the legends — is the honest test, and that is
  what the hardcoded table is.

## Method lineage (the desk's shared engine)

- **Abnormal-return event windows.**
  [`strategy.event_window`](../super_bowl_advertiser/strategy.py) computes excess-of-SPY
  cumulative returns on a short **drift** leg `[+1 … +5d]` and a longer **hold** leg
  `[+6 … +25d]` after the Monday, plus the single-day **Monday reaction** `[0,0]`, with a one-day
  entry lag (you act at Monday's close — the Sunday-night ad already public, no weekend-gap
  look-ahead).
- **Welch t + placebo p-value.** [`strategy.welch_t`](../super_bowl_advertiser/strategy.py) and
  [`strategy.placebo_pvalue`](../super_bowl_advertiser/strategy.py) — the drift's mean vs zero,
  and a 20,000-draw randomisation null sized to the event count.
- **Deterministic synthetic control.**
  [`data.synthetic_ads`](../super_bowl_advertiser/data.py) plants a known post-game drift of size
  `edge` into otherwise-random windows; with `edge=0` the inference must NOT manufacture
  significance, and a large `edge` must light up the drift leg. The control runs offline.
- **Costs on the believers' trade.**
  [`strategy.net_of_costs`](../super_bowl_advertiser/strategy.py) charges a one-way large-cap cost
  on the **two** crossings of a long-the-advertisers ad-calendar basket.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + ~26 advertiser tickers, 2013-06-03 → 2026-07-10,
  cached under `_cache/superbowl_prices.csv`. The advertiser table (tickers, game dates, labels)
  is hardcoded in [`data.ADVERTISERS`](../super_bowl_advertiser/data.py); famously-delisted /
  taken-private advertisers are listed in [`data.DELISTED`](../super_bowl_advertiser/data.py) for
  the survivorship caveat. Game dates are exact Super Bowl Sundays (2015–2024). Headline numbers
  are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 389 — Name-Change-Effect](../389-name-change-effect/)**: the adjacent attention effect
  — does renaming toward the hot theme (`.com`/`Blockchain`/`AI`) pop the stock? Same family (an
  attention grab, not a fundamental), same small-sample / survivorship pathology.
- **[Study 343 — Data-Mining-Roulette](../343-data-mining-roulette/)**: the methodological cousin
  — how loud anecdotes manufacture "laws" that don't survive a representative sample.
