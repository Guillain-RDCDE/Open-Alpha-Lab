# References & literature map — Study 960 (The Unstaked ETF)

## The claim under test

- **The unstaked-wrapper thesis.** A US spot-Ethereum ETP holds coins in custody. Over
  most of the window tested here those coins sat idle rather than being posted to the
  Ethereum consensus layer, so the fund's holder forgoes the protocol's staking reward
  that a self-custodying, self-staking holder would earn. The popular framing is that this
  invisible give-up dwarfs the visible sponsor fee, and that it should show up as a
  widening tracking difference against the coin. This study asks how much of the realised
  gap the staking yield accounts for on top of the expense ratio — and, first, whether the
  gap is measurable at all.
- **The steelman.** The arithmetic is not in dispute: if a staked coin earns ~3%/yr and the
  fund's coins do not, the fund's holder is ~3%/yr behind a staking holder before the
  sponsor fee. The empirical questions are (a) whether that shows up in the tape and (b)
  what benchmark it could possibly show up *against*.

## The mechanism — where the yield comes from and why the benchmark cannot contain it

- **Buterin et al., the Ethereum proof-of-stake specification** (`ethereum/consensus-specs`)
  and the *Ethereum.org staking documentation* — the consensus reward is paid in protocol
  issuance plus priority fees to validators who post 32 ETH; the network-average nominal
  rate is a decreasing function of total stake, and drifted roughly 3–4%/yr over 2024–2026.
  Community dashboards (beaconcha.in, the Ethereum Foundation's issuance calculator) publish
  the realised series. Our `STAKING_YIELD_ANN = 3.0%` is a net-of-provider-commission central
  case drawn from that range, declared as an **ASSUMPTION** and swept 2.0–4.5%/yr.
- **The benchmark point, which is the study's spine.** *ETH-USD is the price of an unstaked
  coin.* The consensus reward accrues in additional coins to a validator, not in the market
  price of one coin — so a "total return of ETH including staking" index is a *constructed*
  series, not a traded one. Any fund-minus-ETH-USD tracking difference therefore contains
  the sponsor fee and the custody frictions, and by construction contains **no** staking
  component whatever. The forgone yield is a bookkeeping entry against a hypothetical
  self-staking holder — real, but not a tape observation.
- **EIP-1559 and the burn** (Buterin, Conner, Dietrichs, Doosan, Weinstein, 2019) — the fee
  burn makes the *net* issuance to a non-staker negative in busy periods, which is the usual
  argument for why the non-staker's dilution is smaller than the headline reward rate. We do
  not model it; the sweep is the honest substitute.

## Measuring tracking difference — why the interval matters more than the point

- **Elton, Gruber & Busse (2004), *Are Investors Rational? Choices Among Index Funds*,
  Journal of Finance** — the founding demonstration that index-fund performance differences
  are almost entirely explained by declared costs, and the template for testing a fund
  against its stated fee.
- **Blitz, Huij & Swinkels (2012), *The Performance of European Index Funds and ETFs*,
  European Financial Management** — realised tracking difference decomposes into the expense
  ratio plus securities lending, tax and sampling effects; it is measured over long windows
  precisely because the daily series is dominated by noise.
- **Petajisto (2017), *Inefficiencies in the Pricing of Exchange-Traded Funds*, Financial
  Analysts Journal** — ETF closes deviate from NAV, and the deviation mean-reverts. That is
  the source of the −0.56 lag-1 autocorrelation and the endpoint sensitivity documented in
  `docs/results.md`.
- **The stale/asynchronous-price problem.** Scholes & Williams (1977) and Dimson (1979)
  formalised the bias from non-synchronous closing prices; Lo & MacKinlay (1990) showed how
  much spurious autocorrelation it injects. Here the mismatch is not stale but *late*: the
  24/7 coin bar is stamped hours after the 16:00 fund close, giving a transient error that
  telescopes out of an endpoint comparison but inflates every daily-difference variance.
  The fund-vs-fund estimator is the standard remedy — a same-close benchmark.
- **Newey & West (1987), *A Simple, Positive Semi-Definite Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica** — the HAC *t* in
  [`strategy.newey_west_t`](../unstaked/strategy.py). With *negatively* autocorrelated
  differences the HAC variance falls **below** the i.i.d. one, so HAC is the *flattering*
  direction on this tape, not a conservative one — and the readings we actually quote (block
  bootstrap, non-overlapping monthly *t*) are flattering in the same direction. We are
  therefore **not** taking the most conservative number available; the widest ruler, the
  i.i.d. daily *t*, is −1.32 on the ETHE−ETHA spread and does not clear 2. The case for
  discarding it — an MA(1)-shaped −0.47 daily reversal that the i.i.d. variance assumes away,
  plus 22/24 negative months and a month-jackknife range of −1.79 to −1.51 %/yr — is set out
  in full in `docs/results.md` §2 rather than left implicit.
- **Politis & Romano (1994), *The Stationary Bootstrap*, JASA** — the circular block
  bootstrap behind [`strategy.bootstrap_td_ci`](../unstaked/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py). Their point about
  block-length sensitivity is load-bearing here and not a footnote: on the fund-vs-coin
  difference the half-width runs from ±18.0 %/yr at *b* = 5 to ±5.4 at *b* = 63, so
  [`strategy.resolution_sweep`](../unstaked/strategy.py) publishes the whole sweep and the
  study quotes a *range* rather than one convenient width.

## Related desk studies (dedup)

- **[Study 618 — GBTC Premium Cycle](../../618-gbtc-premium-cycle/)**: the *closed-end*
  premium/discount of a Grayscale trust before conversion — a wrapper **mispricing** story
  on the pre-ETF tape. Study 960 begins on the day that story ends (2024-07-23) and asks
  about **realised cost** inside the ETF wrapper, where creation/redemption has already
  killed the premium.
- **[Study 913 — Tracking-Difference Persistence](../../913-tracking-difference-persistence/)**:
  does last year's best tracker stay the best, across conventional index funds. Same
  estimator family, different question (persistence, not level) and a benchmark that shares
  the fund's closing stamp — which is exactly the contrast this study exploits.
- **[Study 959 — Crypto Fee War](../../959-crypto-etf-fee-war/)**: the *bitcoin* wrapper race
  across the **ten** lines that launched together on 2024-01-11. Study 960 is the **Ethereum**
  case, and its subject is the one cost that has no bitcoin analogue at all — a consensus
  yield the wrapper does not collect. The comparison is also a coverage warning: 959 can
  replicate its fee finding across nine independent cheap wrappers, while 960 has **two**
  funds in cache and they are the cohort's fee extremes, so 960's spread is a single pair at
  the widest available gap, not a cross-section.
- **[Study 958 — Spot ETF Basis](../../958-spot-btc-etf-basis/)**: the cash-and-carry basis
  after the spot ETFs landed — a futures-vs-spot funding story, not a fund-vs-coin cost one.
- **[Study 378 — ETF NAV Premium](../../378-etf-nav-premium/)**: whether a discount to NAV
  mean-reverts. That premium/discount noise is a *nuisance* here, not the subject; it is what
  the block bootstrap is sized to absorb.
- **[Study 209 — ETH/BTC Ratio](../../209-eth-btc-ratio/)** and
  **[Study 582 — ETH Gas Fees](../../582-eth-gas-fees/)**: Ethereum *return* and *on-chain
  activity* studies. Neither touches the wrapper's cost of ownership.

## Data sources

- **ETHA** (iShares Ethereum Trust), **ETHE** (Grayscale Ethereum Trust), **ETH-USD** (the
  coin), **BIL** (1–3 month T-bill, the cash leg) — daily closes via `yfinance`
  (`auto_adjust=True`), ETF era 2024-07-23 → 2026-06-30. **Not measured, and named so that
  the gap is visible:** FETH, ETHW, ETHV, QETH, EZET, CETH and the Grayscale mini trust, the
  rest of the 2024-07-23 cohort, are absent from the desk cache
  (`data.COHORT_NOT_MEASURED`). ETHA and ETHE distribute nothing, so
  their adjusted close is both **price and total return**; **ETH-USD is price-only** and
  **unstaked**; BIL's close is a genuine **total return**.
- **Non-tape inputs, all declared in [`unstaked/data.py`](../unstaked/data.py) and swept:**
  the sponsor fees (`EXPENSE_RATIOS`, from the issuers' published fee schedules — ETHA's
  headline 0.25%/yr with an early partial waiver, hence the 0.12–0.25% range in
  `ETHA_FEE_RANGE`; ETHE's 2.50%/yr) and the net staking yield (`STAKING_YIELD_ANN`, central
  3.0%/yr, swept `STAKING_SWEEP` 2.0–4.5%/yr). None of these is measured by this study, and
  no conclusion is allowed to rest on a single value of any of them.
- **As-of 2026-06-30**; the partial current month is dropped so the sample never creeps. The
  window opens at the funds' first trading day and is never widened backwards — ETHE's
  pre-conversion history belongs to Study 618.
