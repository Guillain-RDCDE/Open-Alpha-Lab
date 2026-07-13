# References & literature map — Study 751 (Fortune-500-Inclusion)

## The claim under test

- **The folklore.** Being **added to** a famous list draws a wave of attention and buying, and
  being **dropped** sheds it — so a company's **debut on the Fortune 500** (or its **exit**)
  should print an abnormal return around the annual June list reveal. It is pitched as the
  index-inclusion trade (below) applied to a magazine cover: prestige as a catalyst.
- **Where it's repeated.** Business media treat the annual Fortune 500 reveal as a
  market-moving event ("Company X joins the Fortune 500"), and the framing borrows its
  plausibility from the genuine, well-documented **S&P-500 index-inclusion effect**. The
  believers' two-legged version is an *added* pop and a *dropped* penalty; we test both legs.

## The academic anchor — index inclusion is real, but for a reason that does not transfer

- **Shleifer (1986), *Do Demand Curves for Stocks Slope Down?*, Journal of Finance.** The
  canonical result: stocks **added to the S&P 500** earn a positive abnormal return around the
  change — evidence that demand curves for stocks are not flat.
- **Harris & Gurel (1986), *Price and Volume Effects Associated with Changes in the S&P 500*,
  Journal of Finance.** Documents the addition price pop and, crucially, its **partial
  reversal** — consistent with a temporary **price-pressure / forced-buying** mechanism, not a
  permanent re-rating.
- **Wurgler & Zhuravskaya (2002), *Does Arbitrage Flatten Demand Curves for Stocks?*, Journal
  of Business,** and **Chen, Noronha & Singal (2004), *The Price Response to S&P 500 Index
  Additions and Deletions*, Journal of Finance.** Refine the mechanism: the effect is largest
  where arbitrage is hardest and is tied to **index-fund demand**, an asymmetry between
  additions and deletions.
- **Why it does not transfer.** All of the above hinge on a **mechanical demand shock** — index
  funds *must* rebalance to track the S&P 500. The **Fortune 500 is a media ranking by
  prior-year revenue**: no fund tracks it, no one is forced to buy or sell, and the revenue that
  decides membership was public months earlier (so the reveal carries **no new information**).
  Strip out demand and information and only a **pure attention / prestige** effect remains —
  which is exactly the hypothesis this study isolates and tests. See **Study 249 —
  Index-Inclusion** for the real S&P-500 version on this desk.

## The attention hypothesis — the only channel left, and its limits

- **Barber & Odean (2008), *All That Glitters: The Effect of Attention on the Buying Behavior
  of Individual and Institutional Investors*, Review of Financial Studies.** Salient,
  attention-grabbing events reallocate *retail* attention and can move price briefly. The
  Fortune-500 reveal is a plausible attention shock; whether it is large or persistent enough
  to print a tradable abnormal return is the empirical question.
- **Cooper, Dimitrov & Rau (2001), *A Rose.com by Any Other Name*, Journal of Finance.** The
  desk's adjacent evidence that a *label* (a `.com` rename) can pop a stock on attention alone —
  but see **Study 389 — Name-Change-Effect**, where that pop is a coin-flip on a representative
  sample. Attention effects are real in the lab and thin on a fair tape.

## Why the honest test is a placebo null, not a point estimate

- **Small-sample inference.** With ~a dozen events per bucket, a cross-section of abnormal
  returns has a large standard error. We test each bucket's mean against zero with a **one-sample
  t** (Welch, 1947) and, because the sample is tiny and fat-tailed, with a **placebo /
  randomisation null** — draw the same number of random non-event windows on the same tickers
  and ask how often chance matches the observed CAR (Fisher's randomisation logic; Efron &
  Tibshirani, 1993, *An Introduction to the Bootstrap*).
- **Event-study standard errors.** MacKinlay (1997), *Event Studies in Economics and Finance*,
  Journal of Economic Literature — the market-model CAR construction and its estimation-window
  conventions used here.
- **Survivorship.** The worst exits are **bankruptcies** whose series ends (J.C. Penney, Bed
  Bath & Beyond, Sears) and leave no continuing yfinance data; the surviving exit sample is
  biased **against** a negative drop-reaction (Brown, Goetzmann, Ibbotson & Ross, 1992,
  *Survivorship Bias in Performance Studies*, RFS). Named on the Signal axis.
- **Window-mining.** Reporting the most favourable of several event windows is the classic
  data-snooping trap (Harvey, Liu & Zhu, 2016, *…and the Cross-Section of Expected Returns*,
  RFS); we show the full window grid so no single lucky window can be sold as the result.

## Method lineage (the desk's shared engine)

- **Market-model CARs.** [`strategy.event_car`](../fortune_500_inclusion/strategy.py) fits
  `stock = α + β·SPY` on a clean pre-event window and cumulates the abnormal return over the
  event window, with a one-day execution lag for the tradable variant.
- **Placebo null.** [`strategy.placebo_car_dist`](../fortune_500_inclusion/strategy.py) and
  [`strategy.placebo_pvalue`](../fortune_500_inclusion/strategy.py) — a randomisation null
  sized to the event count, drawn from random non-event windows on the same names.
- **Deterministic synthetic control.**
  [`data.synthetic_events`](../fortune_500_inclusion/data.py) plants a known added-bucket CAR
  edge of size `car_bps`; with `car_bps = 0` the inference must NOT manufacture significance,
  and a large edge must light up. Runs offline.
- **Costs.** [`strategy.net_of_costs`](../fortune_500_inclusion/strategy.py) charges a one-way
  round-trip on a "buy on the reveal, hold the window" trade.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + ~26 event tickers, 2015-01-02 → 2026-07-10,
  cached under `_cache/`. The add/drop table (tickers, list-reveal dates, direction) is
  hardcoded in [`data.FORTUNE_EVENTS`](../fortune_500_inclusion/data.py) — a **labelled proxy**
  for a paid point-in-time membership feed — and famously-delisted exits are listed in
  [`data.DELISTED_EXITS`](../fortune_500_inclusion/data.py) for the survivorship caveat.
  Headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 249 — Index-Inclusion](../249-index-inclusion/)**: the *real* S&P-500 version, with
  the demand-shock mechanism this study lacks.
- **[Study 391 — CEO-Turnover](../391-ceo-turnover/)**: the same market-model event-study engine
  applied to another hardcoded corporate-event table (forced/planned CEO changes).
- **[Study 389 — Name-Change-Effect](../389-name-change-effect/)**: the adjacent attention/label
  effect — whether a themed rebrand pops. Same small-sample / survivorship pathology.
