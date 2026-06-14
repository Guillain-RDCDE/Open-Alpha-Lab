# References & literature map — Study 150 (SAD-Effect)

## The claim under test

- **Kamstra, Kramer & Levi (2003)** — *Winter Blues: A SAD Stock Market Cycle*, American Economic Review
  93(1), 324–343. The canonical paper: equity returns are driven by Seasonal Affective Disorder (SAD),
  a clinically recognised depressive condition triggered by reduced daylight in autumn. As daylight
  shortens (Sep-Nov), investors become seasonally more risk-averse and demand higher risk premia,
  depressing prices; as daylight returns (Dec-Mar), risk aversion normalises and prices recover. The
  regression specification uses astronomical day-length change as the independent variable.

## The data-mining critique

- **Kelly & Meschke (2010)** — *Sentiment and Stock Returns: The SAD Anomaly Revisited*, Journal of
  Banking & Finance 34(6), 1308–1326. Documents that the KKL result is not robust: it was weak in
  the original sample (|t| < 2 on recovery in most specifications), absent before the sample, and
  reversed or flat after. The effect is shown to be entangled with the January seasonal and the
  Halloween (Sell-in-May) anomaly — not cleanly attributable to a psychological daylight mechanism.
  Our sub-period analysis reproduces their main finding: KKL sample (1950-2002) shows t ~ 1.8,
  post-publication (2003-2026) shows t ~ -0.85.

- **Jacobsen & Marquering (2008)** — *Is it the Weather?*, Journal of Banking & Finance 32(4),
  526–540. Broader critique of weather-based finance explanations: the weather variables (including
  daylight hours) are highly collinear with calendar effects (January, Halloween) and the incremental
  explanatory power is not statistically significant once those are controlled for.

## The seasonal effects KKL subsumes

- **Bouman & Jacobsen (2002)** — *The Halloween Indicator, "Sell in May and Go Away": Another Puzzle*,
  American Economic Review 92(5), 1618–1635. The winter-half (Nov-Apr) equity premium was documented
  here; KKL claim SAD explains this. Our study 55-summer-lull tests this overlap.

- **Rozeff & Kinney (1976)** — *Capital Market Seasonality: The Case of Stock Returns*, Journal of
  Financial Economics 3(4), 379–402. The original January effect paper; returns in January are
  anomalously high. KKL subsume January into their "recovery" window, so their recovery signal
  is at least partly explained by the pre-existing January seasonal rather than SAD per se.

## Behavioural finance and seasonal risk aversion

- **Hirshleifer & Shumway (2003)** — *Good Day Sunshine: Stock Returns and the Weather*, Journal
  of Finance 58(3), 1009–1032. Documents a positive correlation between sunshine and daily equity
  returns across 26 countries — a related weather/mood effect at higher frequency. This is the
  broader empirical context for mood-based finance; KKL's SAD claim is a lower-frequency version.

- **Saunders (1993)** — *Stock Prices and Wall Street Weather*, American Economic Review 83(5),
  1337–1345. Early evidence that cloud cover correlates negatively with NYSE returns — one of the
  first weather-finance papers; provides the conceptual context for KKL.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica 55(3) —
  [`strategy.daylight_regression`](../sad_effect/strategy.py) implements Newey-West standard errors
  for the OLS slope; [`strategy.summary`](../sad_effect/strategy.py) uses it for the mean t-stat.

- **Block-bootstrap Sharpe CI.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA 89(428)
  — the quantlab engine; available for the quants notebook.

- **Post-publication decay.** McLean & Pontiff (2016), *Does Academic Research Destroy Stock Return
  Predictability?*, Journal of Finance 71(1) — the expected pattern of a data-mined anomaly is
  precisely what we observe: apparent in-sample, absent post-publication.

## Data sources used here

- **Shiller S&P 500 monthly dataset** — staged at `_cache/shiller_sp500.parquet`. Monthly price
  and dividend data for the S&P 500 and its predecessor indices from 1871 to present, compiled by
  Robert Shiller (Yale), originally published in *Irrational Exuberance* (2000). Columns used:
  SP500 (monthly index level) and Dividend (annual dividend, divided by 12 for monthly yield).
  Total return = price change + lagged monthly dividend yield. This is the same primary source
  used by Kamstra, Kramer & Levi (2003).

## Related desk studies

- **[Study 55 — Summer-Lull](../../55-summer-lull/)** — the Halloween / Sell-in-May seasonal, which
  KKL claim is a special case of SAD. Our test finds the winter-summer gap has some statistical
  support but is not clearly attributable to SAD vs other calendar effects.
- **[Study 48 — Groundhog](../../48-groundhog/)** — another folk seasonal, for comparison.
- **[Study 67 — Fed-Drift](../../67-fed-drift/)** — a monthly return seasonality study based on
  the FOMC calendar; illustrates the desk's calendar-pattern methodology.
