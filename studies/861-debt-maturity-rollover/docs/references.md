# References & literature map — Study 861 (Debt-Maturity Rollover Risk)

## The claim under test

- **The thesis.** A firm's debt has a *maturity structure*. Debt due within a year — short-term
  borrowings, commercial paper, and the current maturities of long-term debt — has to be
  **rolled over** (refinanced) at whatever interest rate and credit spread prevail when it comes
  due. A firm funded with a **high share of short-term debt** therefore carries **rollover
  risk**: if rates rise or credit tightens between issuance and maturity, it refinances at a
  worse price, and in a true credit crunch it may not be able to refinance at all. The claim is
  that the market **under-prices** this risk, so high-short-term-share firms subsequently
  **under-earn**, and that the penalty is **largest when rates are actually rising** (the
  post-2022 Fed hiking cycle).
- **The academic anchor.** Rollover risk is a first-order object in the maturity-structure and
  credit-risk literature. He & Xiong (2012, *Journal of Finance*, "Rollover Risk and Credit
  Risk") formalise how the need to refinance maturing debt raises default risk and required
  returns; Almeida, Campello, Laranjeira & Weisbenner (2012, *Critical Finance Review*) show
  firms with more long-term debt maturing right before the 2007-08 crisis cut investment sharply
  — a real rollover shock. The *return* prediction sits inside the broader **distress-risk /
  leverage anomaly** debate: Campbell, Hilscher & Szilagyi (2008, *Journal of Finance*) document
  that high-distress firms puzzlingly **under**-earn (not over-earn), the same sign this study
  finds; the classic risk-premium view would instead predict risky firms earn *more*.
- **The open question we test.** On a small, honestly-thin panel of large US filers that report a
  clean maturity split on EDGAR, does the short-term-debt share (a) **predict a forward return
  penalty** for the high-share names (the mispricing claim), (b) do so **more strongly in the
  2022+ rising-rate era**, and (c) **survive** realistic long-short costs plus borrow — once you
  rank strictly on point-in-time filed values and hold with one execution lag?

## What we measure, and the honesty rails

- **Signal, point-in-time.** `st_share` = (DebtCurrent + LongTermDebtCurrent) ÷ (DebtCurrent +
  LongTermDebtCurrent + LongTermDebtNoncurrent), known only at the **10-Q/10-K filing date**
  (`filed`), never the period end. The three legs are the us-gaap XBRL concepts of the same name;
  a missing short-term leg is treated as a genuine zero (the firm disclosed no such balance). A
  balance-sheet-scaled variant `(DebtCurrent + LongTermDebtCurrent)/Assets` is carried as a
  robustness cut (the *level* of the maturity wall, not just its share of the debt stack).
- **Primary test — calendar-time long-short (Newey-West).** Each month-end, rank the names
  carrying a fresh signal into terciles (the panel is too thin for finer sorts), go **long the
  low-share / short the high-share** tercile equal-weight, earn the **next** month's return (one
  execution lag). The decisive statistic is the **Newey-West (HAC, Bartlett) t** of the monthly
  long-short series — the autocorrelation-robust bar `REAL` is written against (METHODOLOGY →
  *The inference bar*). A one-sample t, a monthly hit-rate, a drop-one-name jackknife, and
  half/quartile/quintile sorts accompany it.
- **Cross-check — pooled event drift.** Bucket all (ticker, filing) events by the signal, measure
  low-minus-high forward drift over ≈1m/1q/2q horizons, with a one-sample t and a **label-shuffle
  placebo** (permute signals, re-form random terciles). The tercile monotonicity picture is read
  honestly — the effect is concentrated in the safe (low-share) leg, not a clean monotone ladder.
- **Third axis — the rate-era cut.** The claim's own prediction is that rollover risk bites when
  rates rise, so we split the calendar long-short at **2022-01-01** (the hiking cycle) and compare
  the two halves; alternate splits (2014/2017/2019) are reported as robustness.
- **Costs & borrow.** The tradability timer charges one-way cost × NAV × monthly turnover on
  **both** legs and makes the short leg pay an annualised borrow, stressed across four cost/borrow
  pairs from 10 bps + 50 bps to 50 bps + 300 bps — the standard desk long-short friction model.
- **Coverage is a first-class caveat, not a footnote.** Only 32 of the 48 basket names tag the
  three maturity legs cleanly; the cross-section is thinner in 2008-09 and the concept start dates
  differ by name. Terciles on a partial large-cap panel are noisy by construction, and much of the
  spread rides the safe leg — every number here should be read in that light.

## Survivorship — named on the Signal axis

The basket is **current survivors** (all still listed): a fixed roster of large debt-carrying
names that report the maturity split today. It **cannot** include firms whose rollover wall
actually pushed them into default or distressed acquisition — the exact left tail the rollover
story is about. For a long-*low*-share / short-*high*-share claim portfolio, that omission biases
the measured penalty **conservatively** (the worst high-share names, which would have made the
short leg pay even more, are missing). We therefore reason about the bias direction explicitly
rather than claiming it away, and never cite the survivor panel to inflate magnitude.

## Data sources

- **Short-term debt, current maturities of long-term debt, noncurrent long-term debt, total
  assets** — SEC EDGAR XBRL `companyconcept` API (`data.sec.gov`): `DebtCurrent`,
  `LongTermDebtCurrent`, `LongTermDebtNoncurrent`, `Assets`; 10-Q/10-K instant facts,
  de-duplicated on period end (earliest filing wins), keeping the filing date so the signal is
  strictly point-in-time. Cached under `_cache/rollover_events.csv`.
- **Daily adjusted closes** — yfinance (no key), cached under `_cache/rollover_prices.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [540-distress-risk-anomaly](../../540-distress-risk-anomaly/) — the Campbell-Hilscher-Szilagyi
  **failure-probability** distress score and its (anomalous, under-earning) return. That ranks on
  a *composite hazard model* of many inputs; this study ranks on a single, mechanical
  **maturity-structure** ratio — *when* the debt comes due, not *how likely* default is overall.
- [123-altman-z](../../123-altman-z/) — the **Altman Z-score** bankruptcy classifier (working
  capital, retained earnings, EBIT, equity, sales — all scaled by assets). A distress *level*
  built from profitability and liquidity; it contains no maturity-timing term at all.
- [230-ohlson-o-score](../../230-ohlson-o-score/) — the **Ohlson O-score** logit of bankruptcy
  (size, leverage, working capital, net-income dynamics). Again a distress *probability*, not the
  short-vs-long **maturity mix** of the debt stack.
- [154-leverage-anomaly](../../154-leverage-anomaly/) — ranks on the **amount** of leverage
  (LongTermDebtNoncurrent/Assets and Liabilities/Equity): *how much* debt a firm carries. This
  study holds the debt level aside and asks about its **maturity composition** — a high-leverage
  firm that has termed everything out scores *low* here, a low-leverage firm rolling commercial
  paper scores *high*.

None of the siblings rank on the **short-term-debt share of total debt** — the maturity-rollover
axis this study owns.
