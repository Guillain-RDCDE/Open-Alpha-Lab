# References & literature map -- Study 501 (Idiosyncratic-Volatility)

## The primary claim under test

- **Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006).** "The Cross-Section of Volatility
  and Expected Returns." *Journal of Finance*, 61(1), 259--299. The founding paper for the
  **IVOL puzzle**. Sorting stocks on idiosyncratic volatility -- the std of the residual from a
  Fama-French three-factor regression -- the authors find that high-IVOL stocks earn
  *abysmally low* average returns: the highest-IVOL quintile underperforms the lowest by ~1%
  per month. The sign is the puzzle: standard theory says only systematic risk is priced, so
  idiosyncratic risk should command zero premium, yet it is priced with a *negative* sign.
- **Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2009).** "High Idiosyncratic Volatility
  and Low Returns: International and Further U.S. Evidence." *Journal of Financial Economics*,
  91(1), 1--23. The follow-up: the negative IVOL-return relation appears in 23 developed
  markets, ruling out a US-specific data artifact.

## Why the puzzle should (or should not) exist -- competing explanations

- **Bali, T. G., Cakici, N., & Whitelaw, R. F. (2011).** "Maxing Out: Stocks as Lotteries and
  the Cross-Section of Expected Returns." *Journal of Financial Economics*, 99(2), 427--446.
  Argues the IVOL effect is a proxy for a **lottery-demand** (MAX) effect: investors overpay
  for stocks with extreme positive returns; once you control for MAX, IVOL's predictive power
  weakens or reverses. (See Study 365 -- Lottery-MAX-Effect on this desk.)
- **Fu, F. (2009).** "Idiosyncratic Risk and the Cross-Section of Expected Stock Returns."
  *Journal of Financial Economics*, 91(1), 24--37. Using *conditional* (EGARCH) IVOL rather
  than the lagged realised IVOL of AHXZ, Fu finds a **positive** IVOL-return relation -- the
  sign of the effect is highly sensitive to how IVOL is measured and to look-ahead choices.
- **Stambaugh, R. F., Yu, J., & Yuan, Y. (2015).** "Arbitrage Asymmetry and the Idiosyncratic
  Volatility Puzzle." *Journal of Finance*, 70(5), 1903--1948. The leading modern explanation:
  IVOL deters arbitrage, so mispricing is largest among high-IVOL names; because short-selling
  is harder than buying, *overpriced* high-IVOL stocks dominate, producing the negative
  average relation. The sign flips among underpriced stocks.

## Distinguishing IVOL from total-volatility / low-beta

- **Blitz, D. & van Vliet, P. (2007).** "The Volatility Effect: Lower Risk Without Lower
  Return." *Journal of Portfolio Management*, 34(1), 102--113. The total-volatility low-vol
  anomaly (Study 330 on this desk). IVOL is the *residual* after removing market beta, so a
  pure IVOL test is not just a re-labelled low-beta / low-total-vol bet.
- **Frazzini, A. & Pedersen, L. H. (2014).** "Betting Against Beta." *Journal of Financial
  Economics*, 111(1), 1--31. The systematic-risk cousin (Study 238). Beta is the market-risk
  loading; IVOL is what is left after stripping it out -- the two are conceptually orthogonal,
  which is exactly why this study residualises against SPY before measuring vol.

## Survivorship and the sign of the effect

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings are correlated with poor performance and are concentrated among
  high-IVOL names (bankruptcy, distress). Removing them biases the high-IVOL leg's measured
  return UPWARD -- the precise mechanism that can flip the puzzle's sign on a survivor basket,
  as we observe.
- **Brown, S. J., Goetzmann, W. N., Ibbotson, R. G., & Ross, S. A. (1992).** "Survivorship
  Bias in Performance Studies." *Review of Financial Studies*, 5(4), 553--580. The classic
  statement of how conditioning on survival manufactures spurious return patterns -- here, an
  apparent *reward* to idiosyncratic risk.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703--708. The HAC long-run-variance estimator in
  [`strategy.hac_tstat`](../idiosyncratic_volatility/strategy.py).

## Related desk studies

- **[Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: the total-vol
  cousin (SPLV/SPHB) -- IVOL strips the market beta that 330 leaves in.
- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: the systematic-risk
  sibling -- same rolling-sort, equal-weight, HAC-inference machinery.
- **[Study 365 -- Lottery-MAX-Effect](../../365-lottery-max-effect/)**: Bali-Cakici-Whitelaw's
  competing explanation for the IVOL puzzle.
- **[Study 332 -- Downside-Beta](../../332-downside-beta/)**: another residual-risk cut of the
  cross-section.
