# References — Study 597 (Guyton-Klinger Guardrails)

## The claim's source

- **Guyton, J. (2004).** *Decision Rules and Portfolio Management for Retirees: Is the "Safe"
  Initial Withdrawal Rate Too Safe?* Journal of Financial Planning, 17(10), 54–62.
  The original decision-rules paper: with an inflation-cap and a freeze-after-loss rule, initial
  withdrawal rates of ~5.8–6.2% were "safe" over his 1973–2003 window.
- **Guyton, J. & Klinger, W. (2006).** *Decision Rules and Maximum Initial Withdrawal Rates.*
  Journal of Financial Planning, 19(3), 48–58.
  Adds the two guardrails tested here — the **capital-preservation rule** (cut 10% when the
  current withdrawal rate exceeds 1.2× the initial, while >15 years remain) and the
  **prosperity rule** (raise 10% below 0.8×) — and reports 5.2–5.6% initial rates at ~99%
  Monte-Carlo success with 65% equity.
- **Bengen, W. (1994).** *Determining Withdrawal Rates Using Historical Data.* Journal of
  Financial Planning, 7(4), 171–180. The fixed constant-real rule (the 4% rule) the guardrails
  are pitched against; our simulator with all rules disabled is exactly this.

## The adversarial literature

- **Kitces, M. (2015).** *What Is A Safe Withdrawal Rate And How Do Guyton-Klinger Decision
  Rules Change It?* Nerd's Eye View. <https://www.kitces.com/blog/url-upside-potential-and-downside-risk-of-guyton-klinger-decision-rules/>.
  Points out the standard critique this study quantifies: the decision rules "work" by cutting
  real spending substantially — sometimes for long stretches — in exactly the scenarios the
  fixed rule fails.
- **Pfau, W. (2015).** *Making Sense Out of Variable Spending Strategies for Retirees.*
  Journal of Financial Planning, 28(10). SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2579123>.
  A unified comparison of dynamic-withdrawal rules: every rule that raises the initial rate
  pays for it with downside spending volatility — the trade-off is universal, not GK-specific.
- **Clare, A., Seaton, J., Smith, P. & Thomas, S. (2017).** *Reducing Sequence Risk Using
  Trend Following and the CAPE Ratio.* Financial Analysts Journal, 73(4) — the broader
  sequence-risk context for withdrawal-rate research on long historical tapes.

## Sibling studies on this desk (the dedup guard)

- [Study 173 — Four-Percent-Rule](../../173-four-percent-rule/): the **fixed** Bengen rule on
  the same Shiller tape (Real; SAFEMAX 4.14% in its annual-cohort, nominal-bond variant). This
  study tests the **DYNAMIC** variant — the claim that decision rules beat the fixed ceiling.
  Framed as: 173 asked *"does 4% survive?"*; 597 asks *"do guardrails let you start at 5%+?"*.
- [Study 596 — Bond Tent Glidepath](../../596-bond-tent-glidepath/): same monthly-start
  30-year cohort framework, same Shiller extract; there the dynamic lever is the **allocation**
  (and it fails), here the dynamic lever is the **withdrawal** (and it mechanically works —
  at the price this study measures). Our fixed-4% success (96.26%) cross-checks 596's 96.33%.
- [Study 172 — Hundred-Minus-Age](../../172-hundred-minus-age/): accumulation glidepaths on
  the same tape.

## Data

- **Shiller, R.** *Irrational Exuberance* long-run US dataset (S&P composite price, dividends,
  CPI, 10-year yield, CAPE), monthly 1871+. Homepage: <http://www.econ.yale.edu/~shiller/data.htm>.
  Fetched via the GitHub raw mirror <https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv>,
  cached at `_cache/shiller_sp500.parquet` (cache-first; same extract staged repo-wide).
- Bond returns: first-order 10-year approximation `y_{t-1}/12 − D·Δy` with modified duration
  D = 7, left **nominal** (the GK rules live in nominal space); all outcomes deflated by
  realised CPI. Stated as a decision in [results.md](results.md).

## Method

- **Newey, W. & West, K. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix.* Econometrica 55(3) — HAC t on overlapping
  cohort differences, bandwidth forced to the full 360-month overlap.
- **Politis, D. & Romano, J. (1992).** Circular block bootstrap — 120-month blocks on the
  joint EQ/BD/CPI monthly tape for distribution-statistic CIs.
