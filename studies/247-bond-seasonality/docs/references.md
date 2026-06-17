# References & literature map - Study 247 (Bond-Seasonality)

## The claim under test

A **turn-of-month (TOM) or turn-of-year (TOY) calendar effect in US Treasury bonds**:
bond prices (and yields in reverse) systematically move at predictable monthly or annual
windows, driven by institutional rebalancing, portfolio flows, or risk-off demand. The
tradable version: buy TLT or IEF on the last 2 trading days of the month and exit on the
first 2 trading days of the next month.

## Literature on calendar effects in bonds

- McConnell, J. J. & Xu, W. (2008), *"Equity Returns at the Turn of the Month,"*
  **Financial Analysts Journal** 64(2), 49-64. Documents the turn-of-month premium in
  equities and discusses the institutional-rebalancing mechanism (portfolio managers
  reallocating into bonds at month-end as a risk-off move), which simultaneously provides
  a theoretical channel for a *negative* yield movement at month-end.
- Ogden, J. P. (1990), *"Turn-of-Month Evaluations of Liquid Profits and Stock Returns:
  A Common Explanation for the Monthly and January Effects,"* **Journal of Finance**
  45(4), 1259-1272. Proposes a payment-cycle explanation for the TOM effect in equities;
  by symmetry, if equity inflows concentrate at month-end, bond holdings may be trimmed,
  or the reverse.
- Lakonishok, J. & Smidt, S. (1988), *"Are Seasonal Anomalies Real? A Ninety-Year
  Perspective,"* **Review of Financial Studies** 1(4), 403-425. The canonical warning
  about calendar anomaly data-snooping and the difficulty of out-of-sample replication.
- Andersen, T. G. & Bollerslev, T. (1998), *"Deutsche Mark-Dollar Volatility: Intraday
  Activity Patterns, Macroeconomic Announcements, and Longer Run Dependencies,"*
  **Journal of Finance** 53(1), 219-265. Documents intraday and calendar patterns in
  fixed-income related markets, providing context for the month-end institutional
  rebalancing channel.
- Swinkels, L. & van Vliet, P. (2012), *"An Anatomy of Calendar Effects,"* **Journal of
  Asset Management** 13(4), 279-292. Surveys calendar anomalies across asset classes and
  notes that bond markets show weaker, less consistent calendar patterns than equities.
- Schwert, G. W. (2003), *"Anomalies and Market Efficiency,"* **Handbook of the
  Economics of Finance** (Eds. Constantinides, Harris, Stulz), Ch. 15. Reviews the
  post-publication decay of calendar anomalies generally; the TOM equity effect survived
  longer than most but has also attenuated.

## The plausible mechanism for a TOM bond effect

1. **Equity TOM flow -> bond demand**: institutional investors buy equities at
   month-end/start (the documented equity TOM effect). To rebalance, they must
   simultaneously sell or hold fixed income, producing asymmetric flows.
2. **Month-end mark-to-market rebalancing**: pension and insurance funds rebalance at
   month-end, increasing bond demand to hit duration targets after equity price movements.
3. **Coupon and principal payment flows**: large coupon payments cluster at standard
   coupon dates (1st and 15th of the month), and recipients tend to reinvest in bonds,
   providing a mechanical bid at certain turn-of-month windows.

None of these channels is strong enough to generate large, reliable returns once costs
are included, which is consistent with the **Weak/Mirage** result this study lands on.

## Why it likely fails as stated

- **Bond markets are deeper and more institutionally dominated than equities.** Predictable
  calendar patterns are arbitraged faster by sophisticated fixed-income desks with
  lower transaction costs.
- **The instrument (TLT) carries duration risk (~17 yr modified duration)**. Large daily
  return swings from rate moves dwarf any calendar microstructure signal.
- **The TOM window covers ~19% of trading days** - unlike extreme calendar rules (~4%),
  this is large enough that a TOM-only book captures most of the term premium, but it
  still misses the other 81% of B&H return.

## Method lineage

- **Newey-West HAC standard errors** for the mean of an autocorrelated return series:
  Newey, W. K. & West, K. D. (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix,"* **Econometrica**
  55(3), 703-708.
- **Wilson score interval** for the win-rate proportions: Wilson, E. B. (1927), *"Probable
  Inference, the Law of Succession, and Statistical Inference,"* JASA 22, 209-212.

## Data sources

- **TLT** (iShares 20+ Year Treasury Bond ETF) - daily, **total-return adjusted**, via
  `quantlab.data` (Yahoo Finance), from 2002-07-30.
- **IEF** (iShares 7-10 Year Treasury Bond ETF) - daily, **total-return adjusted**, via
  `quantlab.data` (Yahoo Finance), from 2002-07-30.

## Related desk studies

- [Study 96 - New-Year-Pop](../../96-new-year-pop/) - a calendar seasonality study in
  equities; same HAC + Wilson framework, similar capacity arithmetic.
- [Study 95 - Holiday-Cheer](../../95-holiday-cheer/) - the desk's other calendar study;
  a pre-holiday effect that was real and then faded - the bond TOM is the analogous test
  for fixed income.
- [Study 89 - Turn-of-Month](../../89-turn-of-month/) - the equity TOM effect; the bond
  TOM effect tested here is the fixed-income counterpart.
