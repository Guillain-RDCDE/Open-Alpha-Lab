# References & literature map — Study 225 (Sector-Rotation)

## The claim under test

- **Business-cycle sector rotation.** A widely-cited practitioner framework (Stovall 1996;
  Fidelity Sector Investing) holds that different sectors outperform at different phases of
  the business cycle: Technology and Consumer Discretionary lead expansions; Energy and
  Materials peak near cycle tops; Utilities and Consumer Staples defensively hold up during
  recessions. A disciplined rotation into the "right" sector for each phase should therefore
  beat a passive index.
- **Momentum proxy for cycle phase.** Since cycle phases are not observable in real time,
  practitioners often use trailing price momentum as a signal — rank the 11 SPDR sector ETFs
  by their 6- or 12-month total return, buy the winners. This is also the cross-sectional
  momentum strategy (Jegadeesh & Titman 1993) applied at the sector level.

## Foundational momentum literature

- **Jegadeesh, N. & Titman, S. (1993).** *Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency.* Journal of Finance 48(1), 65–91. The canonical
  momentum paper: US stocks ranked on 3–12 month past returns continue outperforming for up
  to 12 months. All cross-sectional momentum work — including sector rotation — descends
  from this finding.
- **Asness, C. S., Liew, J. M. & Stevens, R. L. (1997).** *Parallels Between the Cross-
  Sectional Predictability of Stock and Country Returns.* Journal of Portfolio Management
  23(3), 79–87. Extends Jegadeesh-Titman momentum to country indices; the sector analogue
  is the direct descendant.
- **Moskowitz, T. J. & Grinblatt, M. (1999).** *Do Industries Explain Momentum?*
  Journal of Finance 54(4), 1249–1290. Key finding: much of individual-stock momentum is
  explained by industry (sector) momentum — but industry momentum itself is weak and
  unreliable after controlling for common factors.

## Sector rotation frameworks

- **Stovall, S. A. (1996).** *Standard & Poor's Guide to Sector Investing.* McGraw-Hill.
  The practitioner classic that maps the 11 S&P sectors to business-cycle phases. Widely
  cited but largely untested out of sample.
- **Fidelity Investments Sector Investing.** *The Business Cycle Approach to Equity
  Sector Investing.* Research note (various years). Claims that rotating through sectors
  aligned with business-cycle phases adds ~2-4%/yr. Our real-tape test does not replicate
  this active return.
- **Huang, D., Li, J., Wang, L. & Zhou, G. (2020).** *Time Series Momentum: Is It There?*
  Journal of Financial Economics 135(3), 774–794. Shows that many time-series momentum
  effects are data-mined or conditional on specific sample periods — relevant context for
  the sector rotation claim.

## Why the premium is thin on the real tape

- **Equal-weight drag.** The equal-weight sector basket consistently underperforms the
  cap-weighted SPY by ~3-4%/yr in the data. The sector-rotation strategy (also equal-weight,
  concentrated into a subset of sectors) inherits this structural disadvantage.
- **Post-publication decay.** McLean, R. D. & Pontiff, J. (2016). *Does Academic Research
  Destroy Stock Return Predictability?* Journal of Finance 71(1), 5–32. Documented anomaly
  returns drop ~32% post-publication on average; sector momentum has been widely known since
  the late 1990s.
- **No exploitable cycle timing.** Faber, M. T. (2007). *A Quantitative Approach to
  Tactical Asset Allocation.* Journal of Wealth Management, Spring 2007. Sector timing
  strategies that look compelling in backtests often fail out-of-sample because cycle
  phases are identified only in retrospect.
- **Cost sensitivity.** Novy-Marx, R. & Velikov, M. (2016). *A Taxonomy of Anomalies and
  Their Trading Costs.* Review of Financial Studies 29(1), 104–147. Monthly rebalancing
  incurs turnover costs that erase thin alpha layers — our data confirms the active layer
  (vs EW) is only +0.1%/yr gross.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica — used
  via [`strategy.summarize`](../sector_rotation/strategy.py) for the per-period return mean.
- **12-1 and 6-1 momentum score conventions.** Skip-1 month follows Jegadeesh & Titman
  (1993) to avoid contamination from the 1-month reversal effect (De Bondt & Thaler 1985).

## Related desk studies

- **[Study 146 — Country-Momentum](../../146-country-momentum/)**: the same 6/12-month
  momentum rotation applied to 23 country ETFs — same methodology, same verdict (WEAK/FRAGILE).
  Sector rotation shares the equity-beta trap: the absolute return is real but the active
  return vs equal-weight is near zero.
- **[Study 24 — Stampede](../../24-stampede/)**: cross-sectional 12-1 momentum on individual
  US stocks — the purest form of the Jegadeesh-Titman anomaly; WEAK/FRAGILE same as here.
- **[Study 113 — Faber-Timing](../../113-faber-timing/)**: a sector-timing overlay using
  moving-average trend signals rather than cross-sectional rank — a structurally different
  approach to sector allocation.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: the 60/40 portfolio using SPY
  directly — the relevant cap-weight benchmark that sector rotation systematically
  underperforms in our test.
