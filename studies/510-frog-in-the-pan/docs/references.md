# References & literature map -- Study 510 (Frog-In-The-Pan)

## The primary claim under test

- **Da, Z., Gurun, U. G., & Warachka, M. (2014).** "Frog in the Pan: Continuous Information and
  Momentum." *Review of Financial Studies*, 27(7), 2171--2218. The founding paper. The market
  under-reacts more to information that arrives in many small *continuous* steps than to the same
  total move delivered in a few *discrete* jumps -- because gradual signals attract less
  attention. They proxy the continuity of information with an **information-discreteness (ID)**
  measure, `ID = sign(PRET) · (%neg − %pos)`, built from the sign-consistency of daily returns
  over the formation window, and show momentum profits are concentrated in the LOW-ID (gradual)
  names and largely absent in the HIGH-ID (jumpy) names. The metaphor: a frog dropped in boiling
  water leaps out (a discrete shock the market prices instantly); a frog warmed slowly never
  reacts (continuous information the market under-reacts to). Their LOW-ID minus HIGH-ID momentum
  spread is large and significant on the broad CRSP universe (1927--2007).

## The momentum factor it conditions

- **Jegadeesh, N. & Titman, S. (1993).** "Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65--91. The canonical
  3-12 month relative-strength momentum that FIP conditions on. The base effect this study finds
  flat on a large-cap survivor basket (see [507](../../507-cross-sectional-momentum/)).
- **Jegadeesh, N. & Titman, S. (2001).** "Profitability of Momentum Strategies: An Evaluation of
  Alternative Explanations." *Journal of Finance*, 56(2), 699--720. Confirms momentum persists
  out-of-sample and links it to delayed reaction -- the mechanism FIP sharpens.

## Why under-reaction would track information continuity

- **Hong, H. & Stein, H. C. (1999).** "A Unified Theory of Underreaction, Momentum Trading, and
  Overreaction in Asset Markets." *Journal of Finance*, 54(6), 2143--2184. Gradual information
  diffusion across investors generates momentum; FIP is a direct empirical proxy for the *speed*
  of that diffusion.
- **Barberis, N., Shleifer, A., & Vishny, R. (1998).** "A Model of Investor Sentiment." *Journal
  of Financial Economics*, 49(3), 307--343. Conservatism / under-reaction to a stream of small
  signals -- the behavioural backbone of the frog-in-the-pan effect.
- **Da, Z., Engelberg, J., & Gao, P. (2011).** "In Search of Attention." *Journal of Finance*,
  66(5), 1461--1499. (Same lead author.) Limited investor attention -- gradual moves attract
  less of it, the precise channel FIP exploits.

## Subsequent evidence, replication, and attenuation

- **Hou, K., Xue, C., & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial
  Studies*, 33(5), 2019--2133. Many momentum-conditioning anomalies replicate with weaker
  magnitudes on broad, recent samples; large-cap-only universes are the harshest test.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. ~32% post-publication attenuation; FIP
  (published 2014) is squarely in the post-publication window.

## Survivorship bias and universe construction

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings correlate with poor performance; a *gradual* slide into delisting is
  exactly a LOW-ID loser, so survivorship bias inflates the FIP slice's short leg the most.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703--708. The HAC
  *t*-stat in [`strategy.hac_tstat`](../frog_in_the_pan/strategy.py).
- **Politis, D. N. & Romano, J. P. (1994).** Permutation / resampling inference -- the
  label-shuffle placebo p-value that certifies (or fails to certify) the LOW-ID mean.

## Related desk studies (the dedup map)

- **[Study 507 -- Cross-Sectional-Momentum](../../507-cross-sectional-momentum/)** -- the plain
  12-1 winners-minus-losers on the same basket family. Study 510 *conditions* that book on
  information-discreteness; the plain WML is the baseline here.
- **[Study 24 -- Stampede](../../24-stampede/)** -- cross-sectional momentum on the full S&P 500.
  Study 510 is distinct: the FIP *interaction*, not the base effect.
- **[Study 25 -- Clean-Slate](../../25-clean-slate/)** / **[237 -- Residual-Momentum](../../237-residual-momentum/)**
  -- residual momentum (strip out factor exposure). FIP instead conditions on the *path shape*
  (continuity) of the same total return, an orthogonal refinement.
- **[Study 509 -- Intermediate-Momentum](../../509-intermediate-momentum/)** -- conditions
  momentum on the *timing* of past returns; Study 510 conditions on their *continuity*.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) -- the |t| >= 2 inference bar,
  the excess-of-cash Sharpe rule, one execution lag, costs one-way × NAV with shorts paying borrow.
