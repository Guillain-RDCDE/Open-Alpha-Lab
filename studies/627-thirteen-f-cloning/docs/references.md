# References & literature map — Study 627 (13F Cloning)

## The claim under test

- **The folk claim.** "Copy Warren Buffett's 13F 45 days late and you still beat the market."
  A staple of retail investing media, guru-tracking sites (Dataroma, WhaleWisdom, GuruFocus)
  and at least one ETF built on the idea of cloning gurus' filings. The intuition: Berkshire's
  holdings are public, huge, and turn over slowly — so the 45-day disclosure delay should cost
  almost nothing, and the free rider keeps the stock-picking skill without the fee.
- **The academic source of the claim.** Gerald S. Martin & John Puthenpurackal,
  *Imitation is the Sincerest Form of Flattery: Warren Buffett and Berkshire Hathaway* (2008,
  SSRN working paper). On **1976–2006** tape they find a portfolio mimicking Berkshire's
  disclosed holdings — bought **after** the public disclosure — still outperformed the market
  by ~10%/yr. This is the strongest published version of exactly our claim, on an earlier
  window.
- **Buffett's skill is not in dispute.** Andrea Frazzini, David Kabiller & Lasse H. Pedersen,
  *Buffett's Alpha* (2018, **Financial Analysts Journal** 74(4)): Berkshire's long-run alpha
  is real and traceable to leverage on cheap, safe, quality stocks. The question here is
  narrower and operational — did the **lagged public copy** of the 13F beat the market on the
  era when you could actually script it (the XML filings, 2013+)?

## Why the result can differ from the literature

- **Post-publication / post-crowding decay.** R. David McLean & Jeffrey Pontiff, *Does
  Academic Research Destroy Stock Return Predictability?* (2016, **Journal of Finance** 71(1)):
  documented anomalies shrink once public. The Buffett-cloning claim was published in 2008;
  our tape starts 2013.
- **A regime, not just decay.** 2013–2026 is a mega-cap growth bull market in which Berkshire
  itself (BRK-B, +12.4%/yr) lagged SPY (+14.5%/yr). A value-tilted, financials-heavy top-10 —
  minus the AAPL position's early years for the EW leg — lagged further. We say the window
  out loud rather than extrapolate either way.
- **Best-ideas concentration cuts both ways.** Randolph B. Cohen, Christopher Polk & Bernhard
  Silli, *Best Ideas* (2010, SSRN; 2021 update): managers' highest-conviction positions tend
  to outperform. The top-10-by-value clone is exactly a best-ideas extraction — here the best
  ideas (KHC, IBM, WFC, OXY vintages…) are what dragged.

## The mechanics of 13F

- **SEC Form 13F.** Institutional investment managers with ≥ $100M in 13(f) securities must
  file quarterly within **45 days** of quarter-end (Rule 13f-1; Securities Exchange Act
  §13(f)). The disclosure lag is therefore *inside* the filing date — rebalancing at the
  filing date needs no extra embargo. https://www.sec.gov/divisions/investment/13ffaq
- **XML information tables** (machine-readable holdings) begin with the 2013Q2 filings — the
  start of our sample. Values are reported in $thousands before 2023 and $ after; only
  within-filing ranks and weights are used, so the unit change is harmless.
- **Confidential treatment.** The SEC can grant Berkshire confidential treatment on positions
  being built; those appear only in later amendments (13F-HR/A). We use **original filings
  only** — exactly the free rider's real-time information set.

## Method lineage (the desk's shared engine)

- **Filing parser.** [`data.fetch_13f`](../thirteen_f_cloning/data.py) — EDGAR submissions
  API + XML information tables, aggregated by CUSIP across Berkshire's reporting managers,
  originals only, top-15 cached.
- **The clone.** [`strategy.build_clone`](../thirteen_f_cloning/strategy.py) — rebalance at
  the close of the first trading day after each filing date (the study's single execution
  lag), drifting weights between filings, one-way costs × traded NAV.
- **HAC inference.** [`strategy.nw_tstat`](../thirteen_f_cloning/strategy.py) and
  [`strategy.capm_alpha_nw`](../thirteen_f_cloning/strategy.py) — Newey & West (1987,
  **Econometrica** 55(3)) standard errors on monthly active returns and CAPM alpha,
  excess-vs-excess (^IRX risk-free leg).
- **Random-manager placebo.** [`strategy.random_manager_baseline`](../thirteen_f_cloning/strategy.py)
  — 200 seeded random top-10 selections from the same universe/calendar/lag (house rule:
  every random baseline averages ≥ 20 seeds).
- **Synthetic control.** [`data.synthetic_world`](../thirteen_f_cloning/data.py) — a planted,
  tunable manager alpha disclosed through lagged quarterly filings; the null must stay flat.
  Machinery proof only.

## Data sources used here

- **SEC EDGAR** — submissions JSON + 13F-HR information tables, CIK 0001067983
  (Berkshire Hathaway): https://data.sec.gov/submissions/CIK0001067983.json and
  https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001067983&type=13F-HR .
  Cached under `_cache/brk_13f_top.csv`.
- **yfinance** daily auto-adjusted (total-return) closes for the 28 priceable clone names +
  SPY + BRK-B + ^IRX, 2013-01-02 → 2026-06-30, cached under `_cache/clone_prices.csv`.
- All headline numbers pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (dedup guard)

- [263-insider-buying](../263-insider-buying/) — the **Form 4** cousin: corporate insiders'
  own open-market buys, an *insider-information* signal with a 2-business-day disclosure
  clock. This study is the **Form 13F** claim — quarterly *institutional holdings* cloning
  with a 45-day lag, and Berkshire-specific because that is how the folk claim is always
  told. Different form, different clock, different mechanism (skill free-riding, not
  information timing).
- The desk's factor-zoo teardowns (e.g. [88-dogs-of-the-dow](../88-dogs-of-the-dow/)) share
  the "famous public recipe vs the tape" framing; none touches 13F cloning.
