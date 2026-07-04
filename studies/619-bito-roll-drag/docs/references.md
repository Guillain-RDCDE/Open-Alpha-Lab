# References — Study 619 (BITO Roll Drag)

## The claim's source

- **"BITO doesn't track bitcoin — the roll eats you alive."** A staple of crypto-ETF
  commentary since the fund's first week. BITO listed 2021-10-19 as the first US bitcoin ETF
  (futures-based under the '40 Act, per then-SEC-chair Gensler's preference) and gathered
  ~$1B faster than any ETF in history; the contango-drag critique followed immediately.
  - ProShares Bitcoin Strategy ETF (BITO) — official page, prospectus, roll policy and the
    0.95% expense ratio: https://www.proshares.com/our-etfs/strategic/bito
  - CME Bitcoin futures (BTC) contract specs — cash-settled to the BRR, **termination on the
    last Friday of the contract month**:
    https://www.cmegroup.com/markets/cryptocurrencies/bitcoin/bitcoin.contractSpecs.html

## Key papers & industry analyses

- **Keynes (1930), *A Treatise on Money*** — the original theory of normal backwardation /
  contango: a hedging-pressure premium embedded in the futures curve that a passive long roll
  pays or collects.
- **Gorton & Rouwenhorst (2006), "Facts and Fantasies about Commodity Futures,"** *Financial
  Analysts Journal* 62(2) — decomposes futures-index returns into spot + roll yield +
  collateral; the canonical framework for "the roll is the return."
  https://doi.org/10.2469/faj.v62.n2.4083
- **Bhardwaj, Gorton & Rouwenhorst (2015), "Facts and Fantasies about Commodity Futures Ten
  Years Later,"** NBER WP 21243 — the out-of-sample sequel. https://www.nber.org/papers/w21243
- **Schmeling, Schrimpf & Todorov (2022), "Crypto Carry,"** BIS Working Paper No 1087 — documents
  the persistently positive (and occasionally violently inverted) bitcoin futures basis on CME
  and offshore venues; the carry BITO's roll locks in.
  https://www.bis.org/publ/work1087.htm
- **SEC (Oct-2021) approval context and the Jan-2024 spot bitcoin ETP approvals** — the order
  that ended BITO's monopoly: https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011024
- **IBIT options begin trading (Nov-2024)** — Nasdaq/OCC listing, which migrated the last
  BITO-only rationale (a listed option chain) to the spot ETF:
  https://www.blackrock.com/us/individual/products/333011/ishares-bitcoin-trust

## Named siblings — the mechanical-decay family (dedup guard)

Same *family* — a packaged vehicle that mechanically bleeds versus its reference — but a
**new instrument and a new mechanism test**: here the vehicle is a *long* futures ETF on a
**monthly** CME roll vs a *live spot benchmark that also trades as an ETF*, so the drag is
measurable two ways (vs spot, and matched-close vs IBIT) and the roll-window attribution can
ask *where* the toll is paid:

- [Study 61 — Slow-Burn](../61-slow-burn/) — leveraged-ETF volatility decay (daily-rebalance
  variance drag, not a futures roll).
- [Study 100 — Melting-Ice](../100-melting-ice/) — contango bleed in the commodity-futures ETF
  (USO-style), the same carry arithmetic on a different curve and clientele.
- [Study 375 — VXX-Roll-Decay](../375-vxx-roll-decay/) — the VIX-futures ETP roll bleed and the
  short-carry book against it (crash-insurance premium).

## Data sources

- **yfinance** (Yahoo! Finance, no key) — `BITO`, `IBIT` daily total-return closes
  (`auto_adjust=True`; BITO's large monthly distributions make total-return mandatory),
  `BTC-USD` spot (7-day tape, sampled on BITO trading days), `BTC=F` CME front-month future
  (contango classification only). https://github.com/ranaroussi/yfinance
- Fee levels quoted in results: BITO expense ratio **0.95%** (ProShares), IBIT sponsor fee
  **0.25%** (iShares prospectus).

## Method citations (shared desk kit)

- **Newey & West (1987)** — HAC (Bartlett-kernel) standard errors; the daily/monthly gap series
  carry persistent basis regimes, so i.i.d. *t*-stats would overstate certainty.
- **Welch (1947)** — unequal-variance two-sample *t* for the roll-window and regime splits.
- Desk-wide rules: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (*t* ≥ 2 on
  the real tape for `REAL`), one documented lag, costs one-way × NAV with shorts paying borrow,
  synthetic controls as machinery proofs only.
