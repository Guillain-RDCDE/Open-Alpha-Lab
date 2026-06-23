# References & literature map — Study 400 (Patent-Intensity)

## The claim under test

- **The innovation premium (believers' version).** The firms that invest most heavily in
  invention — the highest **R&D intensity** (R&D expense / revenue) and the most patent-prolific —
  are compounding intangible moats the market is slow to price, so a portfolio **long the
  high-intensity innovators and short the low-intensity incumbents** should harvest a persistent
  premium. The pitch travels through factor-investing marketing ("innovation factor", "intangible
  capital", thematic "disruption" ETFs) and a real academic literature (below).
- **The academic backbone.** Louis Chan, Josef Lakonishok & Theodore Sougiannis (2001), *The Stock
  Market Valuation of Research and Development Expenditures*, **Journal of Finance** 56(6): firms
  with high **R&D-to-market-equity** earn higher subsequent returns — but the headline result is
  largely a *value/mispricing* effect (R&D scaled by **price**, not by sales), and R&D scaled by
  *sales* or *assets* shows far weaker return predictability. David Hirshleifer, Po-Hsuan Hsu & Dongmei
  Li (2013), *Innovative Efficiency and Stock Returns*, **Journal of Financial Economics** 107(3):
  it is **innovative *efficiency*** (patents or citations *per dollar of R&D*), not gross R&D
  intensity, that predicts returns — gross intensity alone is a weak, confounded signal. Both papers
  are the steelman *and* the warning: the premium that survives is about *efficiency / cheapness*, not
  about *spending a lot on R&D*.

## Why R&D intensity is a *proxy* for patents — and what we do

- **No free issued-patent feed.** Daily/annual issued-patent counts and forward-citation weights
  (USPTO PatentsView, Kogan-Papanikolaou-Seru-Stoffman patent-value data) are not on a free,
  keyless endpoint. Following the literature's own operationalisation, we use **reported R&D expense
  / revenue** from audited filings as the transparent stand-in for "how patent-and-invention-intensive
  is this firm." We label it a **proxy** everywhere. It captures *input* intensity (how much a firm
  spends on inventing) but not *output quality* (whether the spend yields valuable patents) — exactly
  the distinction Hirshleifer-Hsu-Li show is decisive, and a reason to expect gross intensity to be a
  *weak* signal.
- **SEC EDGAR companyfacts.** R&D and revenue come from `us-gaap:ResearchAndDevelopmentExpense`
  (and the software-R&D and revenue concept variants) via the public XBRL `companyfacts` API
  (`data.sec.gov`, no key). We keep only 10-K, full-fiscal-year facts. A name with no R&D line
  (banks, energy, staples retail) is floored to intensity ~0 — economically correct (those firms
  genuinely don't patent), and it is what populates the short leg.

## Why a sector/style control is the crux

- **R&D intensity is a sector map.** High-R&D ≈ semis, software, pharma, biotech; low-R&D ≈ banks,
  insurers, utilities, energy, staples retail, telco. So "long high-intensity / short low-intensity"
  is, mechanically, **long growth/tech-pharma / short value-financials-staples** — the canonical
  style axis (Fama & French, 1993, *Common Risk Factors in the Returns on Stocks and Bonds*, **JFE**
  33; the HML value factor and its growth-tech mirror). Any apparent "innovation premium" must be
  shown to be *distinct* from this style/sector beta. We test it with a **random blind long/short**
  control (the sampling distribution of any split of the same field) and report where the intensity
  split sits in it.
- **Intangibles and accounting.** Baruch Lev (2001), *Intangibles: Management, Measurement, and
  Reporting* (Brookings); Lev & Sougiannis (1996), *The capitalization, amortization, and
  value-relevance of R&D*, **Journal of Accounting and Economics** 21 — why expensed R&D distorts
  book values and creates the very value/growth mismeasurement the premium rides on.

## Inference & honesty (the desk's shared method)

- **HAC (Newey-West) t-stat.** [`strategy.hac_tstat`](../patent_intensity/strategy.py) — the
  Signal-axis test on the monthly long-short and long-minus-SPY spreads. Newey & West (1987), *A
  Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix*, **Econometrica** 55. `REAL` requires `t ≥ 2` on the real tape; a sub-2 *t* with literature
  support reads `WEAK`.
- **Multiple testing on a famous factor.** Harvey, Liu & Zhu (2016), *…and the Cross-Section of
  Expected Returns*, **Review of Financial Studies** 29; McLean & Pontiff (2016), *Does Academic
  Research Destroy Stock Return Predictability?*, **Journal of Finance** 71 — why a published anomaly
  needs a higher bar and tends to shrink out of sample (the R&D-intensity premium has been public
  since 2001).
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../patent_intensity/data.py) plants a *known* annual long-high-minus-
  short-low premium via the `edge` knob; the offline core runs with no network. At `edge = 0` the
  long-short must stay insignificant (no false positive); a large planted edge must light up.
- **One reporting lag, costs, borrow.**
  [`strategy.intensity_books`](../patent_intensity/strategy.py) forms the year-Y book from
  fiscal-year-(Y-1) intensity (no look-ahead), charges one-way turnover × NAV at a stated bps, and
  **charges borrow on the short leg** (a long/short pays to be short).

## Data sources used here

- **SEC EDGAR companyfacts** — annual R&D / revenue for 40 large-caps, 2007-2026 fiscal years,
  cached under `_cache/intensity.parquet`.
- **yfinance** — monthly total returns (auto-adjusted) for the 40 names + SPY, 2005-02 → 2026-05,
  cached under `_cache/returns.parquet`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 393 — AI-Datacenter-Basket](../../393-ai-datacenter-basket/)**: a thematic basket whose
  outperformance is look-ahead *selection*; here the basket is sector-chosen (not return-chosen), so
  the critique is *style attribution* rather than winner-selection — the two ways a "theme" inflates.
- **[Study 395 — Quantum-Computing-Basket](../../395-quantum-computing-basket/)** and
  **[Study 396 — Reshoring-Basket](../../396-reshoring-basket/)**: sibling thematic-factor teardowns —
  whether a named "innovation/theme" tilt is a distinct, forward-harvestable edge or repackaged beta.
