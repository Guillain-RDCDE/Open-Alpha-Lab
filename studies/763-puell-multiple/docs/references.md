# References & literature map — Study 763 (Puell-Multiple)

## The claim under test

- **The source.** David Puell, *"The Puell Multiple"* (2019), published on Medium and
  popularised through LookIntoBitcoin, Glassnode and Woobull. The metric is named after him. As
  the believers state it: Bitcoin miners are the market's structural, near-continuous sellers
  (they must sell issued coins to pay for power and hardware). The **daily issuance value in USD**
  — new coins minted that day × price — relative to its own **trailing 365-day average** measures
  how much sell-side pressure miners are exerting *versus normal*. When the multiple is **high
  (> ~4)**, daily issuance revenue is running far above its yearly norm — historically the euphoric
  cycle **top** (sell / take profit). When it is **low (< ~0.5)**, miner revenue is severely
  compressed and the weakest miners capitulate — historically the cycle **bottom** (buy). Puell's
  original charts highlight the 2011, 2013, 2017 tops and the 2015, 2018 bottoms.
- **The economic logic, steelmanned.** Unlike a pure price oscillator, the Puell Multiple has a
  supply-side story: it is anchored to *issuance*, a quantity fixed by protocol and stepped down
  at each halving. The halving imprint is real — for ~1 year after a halving the multiple is
  mechanically suppressed because the trailing denominator still contains pre-halving days at
  double the block reward — which its proponents read as "the metric knows the supply shock."
- **What we are NOT claiming to test.** A fee-inclusive Puell variant (some dashboards add
  transaction fees to issuance). That series is not reconstructable from price + schedule alone
  and is explicitly out of scope; we test the **original, issuance-only** definition, which we can
  rebuild exactly.

## What we measure, and the honesty rails

- **The metric is reconstructed, not proxied.** `Puell(t) = 144·reward(t)·price(t) /
  trailing-365d mean of the same`. The block reward is the **known halving step function** (25 →
  12.5 → 6.25 → 3.125 over the 2014-2026 tape), and the 144 blocks/day constant **cancels** in the
  ratio. The single named approximation — real daily block counts vary a few percent around the
  144 target because difficulty retargets every 2016 blocks — largely cancels in the ratio and
  never moves a threshold crossing. This is a genuine reconstruction of the canonical series, a
  strength over a digitised proxy chart. A direct consequence, stated plainly: within a halving
  epoch the Puell Multiple is `price(t) / trailing-365d-mean(price)` — a 365-day price ratio —
  so most of its "signal" is price relative to its own one-year average.
- **Signal test — HAC predictive regression.** Forward 30/90/180-day BTC log-return on
  `log(Puell)`, Newey-West standard errors with lag ≥ the horizon (daily forward windows overlap
  heavily; the HAC lag makes the *t* honest about the induced autocorrelation). A contrarian gauge
  needs a *negative* slope clearing |t| = 2. A horse race adds BTC's own trailing-180d momentum.
- **Band event study.** Each day mapped to top (Puell ≥ 4) / neutral / bottom (Puell ≤ 0.5); mean
  forward return per band vs the unconditional distribution, with a Welch *t* and a **random-date
  placebo** (20 × 1,000 draws). The placebo, not the raw day-count, is the guard — the band days
  are large in number but heavily autocorrelated (the top band is a *single* 2017-18 episode), so
  the Welch *t* is reported with a loud fake-precision caveat.
- **One documented execution lag.** Puell known at the close of day t; the timer's exposure
  applies to day t+1's return (a single `shift`, applied once).
- **Costs one-way × NAV** (10 bps) on every flip; the timer is long/flat (no borrow); price-only
  (BTC pays no yield), labelled as such. Gross and net both reported; the decisive comparison is
  **excess-of-buy-and-hold** (timer minus HODL), HAC *t*.
- **Single-survivor named on the Signal axis.** BTC is the surviving 100×+ moonshot and Puell is
  derived from its own price; every band is read off ~three cycles of one asset.

## Data sources

- **BTC-USD daily close** — yfinance (no key), cached under `_cache/puell_btc_usd.csv`,
  2014-09-17 → 2026-06-30. The identical desk tape used by sibling study
  [663-hash-ribbons](../../663-hash-ribbons/) (same fingerprint `9529d5277775`), reused so the
  crypto studies agree on the one price series they share.
- **Halving schedule** — hardcoded in [`puell_multiple/data.py`](../puell_multiple/data.py)
  (`HALVINGS`): block-height-derived dates 2012-11-28, 2016-07-09, 2020-05-11, 2024-04-20 and the
  corresponding rewards. Public, deterministic, protocol-fixed.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Shared method

- **Newey, W. & West, K. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix.* Econometrica 55(3) — the HAC standard errors used
  in the predictive regression and the timer's excess-return *t*.
- **White, H. (2000)** / **Politis & Romano (1994).** Reality Check / stationary bootstrap — the
  desk's data-snooping discipline; here the threshold sweep is reported precisely to expose the
  knob-tuning that a single "best" threshold would hide.
- **Welch, B. L. (1947).** The unequal-variance *t*-test used for the band event study (reported
  with the small-effective-n caveat).

## Related desk studies (the dedup map — what this study is NOT)

- [293-mvrv-ratio](../../293-mvrv-ratio/) — **MVRV** (market value / realized value) as a
  contrarian top/bottom gauge, tested monthly (NONE / MIRAGE). Same *family* of on-chain
  valuation-band timer, but a **different metric** (realized cap, a cost-basis average) and a
  monthly cadence; MVRV needs a digitised proxy series, whereas the Puell Multiple is
  reconstructed exactly from price + schedule. This study is Puell's issuance-revenue oscillator,
  daily, with a HAC forward-return regression.
- [663-hash-ribbons](../../663-hash-ribbons/) — a **hashrate** SMA crossover as a rare discrete
  *buy* event (WEAK / FRAGILE). Hashrate (miner *effort*), not issuance *revenue*; a crossover
  event, not a valuation band.
- [221-mayer-multiple](../../221-mayer-multiple/) — price / **200-day** SMA valuation band. The
  Puell Multiple within an epoch is a *365-day* price ratio, so it is a close cousin — but it is
  built from issuance revenue and carries the halving imprint the Mayer Multiple lacks.
- [323-btc-halving](../../323-btc-halving/) — the halving **calendar** as a return event; uses the
  same schedule this study uses to build issuance, but tests the date, not a revenue oscillator.
- [210-crypto-trend](../../210-crypto-trend/) — 200-day price SMA **trend-following** (long/flat),
  a momentum rule, not a contrarian valuation band.

None of the siblings test the literal issuance-only **Puell Multiple** (daily miner issuance
value / trailing-365d average) as a contrarian top/bottom timer — this study's own axis.
