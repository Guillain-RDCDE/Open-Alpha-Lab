# References & literature map — Study 942 (The Inverse Tax)

## The claim under test

- **The "inverse tax" thesis.** The standard warning attached to SH, PSQ and SDS is that an
  inverse ETF is a *structurally* worse way to be short than simply being short: you pay an
  expense ratio the direct shorter does not; you eat the daily-reset path drag; and you hand
  the sponsor the financing that a short seller would otherwise collect as a rebate on the
  short-sale proceeds. On this telling the funds are a tax on people who cannot (or will not)
  open a margin account, and anyone who can borrow the shares should short the index outright.
- **The steelman for the funds.** Three things run the other way and are rarely counted. The
  ProShares inverse S&P 500 / Nasdaq-100 funds track the **price** index, so their holder never
  owes the dividends a share-short is charged for. The funds hold their NAV in collateral and
  in swaps struck against a financing rate, so they credit short-rate interest on the notional
  — and a retail shorter usually receives **no rebate at all** on the proceeds. And the
  daily-reset drag is a property of *maintaining constant leverage*, not of the wrapper: a
  self-managed book rebalanced to −1× every day pays exactly the same drag. Which side wins is
  an arithmetic question, and it is what this study settles on the tape.

## The mechanics being priced

- **Cheng & Madhavan (2009), *The Dynamics of Leveraged and Inverse Exchange-Traded Funds*,
  Journal of Investment Management.** The canonical statement of the constant-leverage daily
  identity and of the path dependence it creates: over any horizon longer than a day the
  realised return of a −k× fund is a function of the index's *path*, not just its endpoints.
  The compounding term this study measures as ``daily-reset − static`` is theirs.
- **Avellaneda & Zhang (2010), *Path-Dependence of Leveraged ETF Returns*, SIAM Journal on
  Financial Mathematics.** The closed-form decomposition of the horizon return into a power of
  the index return and a variance term of order ``(k² − k)/2 · σ²``; the drag on a −1× fund is
  ``σ²`` and on a −2× fund ``3σ²``, which is why the −2× arm's drag is ~3× the −1× arm's here.
- **Lu, Wang & Zhang (2009), *Long Term Performance of Leveraged ETFs*.** Empirical evidence
  that leveraged and inverse funds track their daily benchmarks tightly and that the
  multi-period divergence is compounding, not tracking failure — consistent with the
  near-unit realised exposure (β ≈ 0.99) we measure.
- **Charupat & Miu (2011), *The Pricing and Performance of Leveraged Exchange-Traded Funds*,
  Journal of Banking & Finance.** Decomposes leveraged-ETF returns into the leveraged index
  leg, the financing leg and fees — the same accounting this study runs, extended here to a
  directly-short book.
- **The benchmark is a *price* index.** The S&P 500 and Nasdaq-100 indices that SH/PSQ/SDS
  track are price indices, so a fund holder never owes the constituents' distributions while a
  share-short does. This is a documented feature of the prospectuses rather than a finding, but
  it means the fund and the share-short are **not short the same payoff**, and it is the largest
  single term in the raw gap. The study therefore reports both the product-vs-product race and a
  **same-mandate** race in which the fund is debited the distributions; the two carry opposite
  signs, and both are significant.
- **Survivorship.** SH (2006), PSQ (2006) and SDS (2006) are survivors, and the industry has
  liquidated a long tail of inverse and leveraged-inverse products. No delisted-fund database is
  used here, so the measured wrapper residual is a survivor's residual and the universe is an
  explicit hindsight pick (the largest and longest-lived of their kind). Elton, Gruber & Blake
  (1996), *Survivorship Bias and Mutual Fund Performance*, RFS, is the reference for how much
  that can be worth in a fund cross-section.
- **Trainor & Baryla (2008), *Leveraged ETFs: A Risky Double That Doesn't Multiply by Two*,
  Journal of Financial Planning.** The popular statement of the decay warning that the
  "inverse tax" folklore descends from.

## Short-selling economics — where the assumptions come from

- **D'Avolio (2002), *The Market for Borrowing Stock*, Journal of Financial Economics.** The
  reference on stock-loan fees: the vast majority of names (and index ETFs in particular) are
  *general collateral* and lend for a few tens of basis points. Our 30 bps base case and the
  0-100 bps sweep sit squarely inside that distribution — but it remains a **PROXY**, not tape.
- **Duffie, Gârleanu & Pedersen (2002), *Securities Lending, Shorting, and Pricing*, Journal of
  Financial Economics.** The rebate on short-sale proceeds as the price of the loan, and why
  the lender's share of the short rate varies so much by client type. This is exactly the
  ``credit`` parameter the study sweeps from 0 to 2 units of the bill rate.
- **Regulation T and FINRA margin rules** are why a retail account's short proceeds are held as
  collateral and, at most brokers below institutional balances, earn the customer nothing.
  Nothing in this study assumes a particular broker: the answer is reported as a function of
  the credit, and the **break-even credit** is quoted so the reader can place their own account
  on the axis.

## Related desk studies (dedup)

- **[Study 61 — Slow-Burn](../../61-slow-burn/)** and **[Study 100 — Melting-Ice](../../100-melting-ice/)**
  test whether **3× long** funds (TQQQ, UPRO) "decay to zero". They price the wrapper against
  the *index*; Study 942 prices it against the *honest alternative book* — the question is not
  "does leverage decay" but "is the fund a worse short than doing it yourself".
- **[Study 941 — Short Both Legs](../../941-double-short-leveraged-pair/)** shorts TQQQ *and*
  SQQQ to harvest the decay as a return stream. Study 942 never trades the decay; it asks an
  implementation question about a single directional book.
- **[Study 943 — Reset Frequency](../../943-leverage-reset-frequency/)** asks whether a
  *monthly* reset would beat the daily one. Study 942 holds the reset frequency fixed at daily
  on both arms and isolates the wrapper, the financing and the dividends instead.
- **[Study 945 — The Hidden Financing](../../945-leverage-financing-cost/)** measures the
  financing rate embedded inside **long** leveraged funds. Study 942 measures the mirror
  quantity on the **inverse** side, where the sign of the financing flips and becomes the
  decisive term.
- **[Study 92 — Easy-Money](../../92-easy-money/)** and
  **[Study 375 — VXX-Roll-Decay](../../375-vxx-roll-decay/)** short *volatility* ETPs for their
  bleed — a different instrument, a different bleed, and a return strategy rather than an
  implementation comparison.
- **[Study 914 — Securities-Lending Offset](../../914-sec-lending-offset/)** asks whether the
  lending revenue an index fund earns reaches its holders. Study 942 sits on the borrower's
  side of the same market: what the *shorter* pays and is credited.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../inverse_tax/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_gap_ci`](../inverse_tax/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Data sources

- **SH** (−1× S&P 500), **PSQ** (−1× Nasdaq-100), **SDS** (−2× S&P 500), **SPY**, **QQQ**,
  **BIL** (1-3 month T-bills, the excess-of-cash yardstick) — daily **total-return** closes via
  `yfinance` (`auto_adjust=True`), from the shared desk cache.
- **SPY, QQQ price-only closes** (`auto_adjust=False`, cached as `praw_*`) — needed because the
  dividend distinction is the whole point of one decomposition term: a share-short owes the
  distributions, the funds' price-index benchmark does not.
- **^IRX** — the 13-week Treasury-bill discount rate, a *level* in percent, accrued act/360
  over calendar days at the previous close's quote. It is a discount quote rather than a
  bond-equivalent yield; `strategy.financing_crosscheck` reports it against BIL's realised
  total return so the wedge is visible rather than assumed away.
- **Non-tape inputs, all labelled PROXY / ASSUMPTION and swept:** the rebate credit (0-2 units
  of the bill rate), the stock-loan fee (0-100 bps), the direct book's one-way rebalance cost
  (0-10 bps × NAV), and the prospectus gross expense ratios (SH/SDS 0.89%, PSQ 0.95%) used only
  inside the accounting decomposition.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps. The
  common window starts at BIL's 2007 inception, which is also where the tradable cash leg
  begins; SH/PSQ (June 2006) and SDS (July 2006) pre-date it by under a year.
