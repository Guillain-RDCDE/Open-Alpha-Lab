# Sources & literature map — Study 38 (Chorus)

## The claim's source

- **Z. Kakushadze & J. A. Serur (2018), *151 Trading Strategies*, §3.20 — "Combining alphas (alpha
  combo)."** The catalogue entry for blending several individual signals into one portfolio, and the
  reason the desk's last study is a *combination* rather than a new anomaly. SSRN `3247865` · arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). *(Copyrighted; not redistributed.)*

## Why combining decorrelated signals works — the Fundamental Law

- **Grinold, R. (1989), "The Fundamental Law of Active Management," *Journal of Portfolio Management*
  15(3)**, and **Grinold, R. & Kahn, R. (1999), *Active Portfolio Management* (2nd ed.).** The result
  this whole study tests: the information ratio of a strategy scales as **IR ≈ IC · √breadth** — skill
  per bet times the square root of the number of *independent* bets. Combining several weak,
  *decorrelated* signals raises breadth, so the blend's IR exceeds any component's. The study's headline
  ("the edge is diversification, not prediction") is this law in one sentence — and its `MIXED` real-data
  result is the law's fine print: breadth multiplies IC, so a *negative*-IC bet, however decorrelated,
  subtracts.
- **Clarke, de Silva & Thorley (2002), "Portfolio Constraints and the Fundamental Law of Active
  Management," *Financial Analysts Journal* 58(5).** The "transfer coefficient" refinement — real
  constraints (here: dollar-neutral, gross-1, daily turnover) erode how much of the theoretical IR you
  actually capture, which is exactly where the combo dies net of cost.
- **Markowitz, H. (1952), "Portfolio Selection," *Journal of Finance* 7(1).** The original case that
  diversification across imperfectly-correlated return streams is the only "free lunch" in finance — the
  intuition underneath the alpha-combo.

## The component anomalies (each its own prior study)

- **Study 24 — Stampede (§4.1, cross-sectional momentum)**, [`../../24-stampede/`](../../24-stampede/).
  The 12-1 month winners-minus-losers leg of the chorus (Jegadeesh & Titman 1993).
- **Study 33 — Slingshot (§3.9, short-term reversal)**, [`../../33-slingshot/`](../../33-slingshot/).
  The 5-day contrarian-fade leg (Jegadeesh 1990, Lehmann 1990).
- **Study 18 — Dull-Roar** and the **low-volatility anomaly** — the long-low-vol leg (Ang, Hodrick, Xing
  & Zhang 2006, "The Cross-Section of Volatility and Expected Returns," *Journal of Finance* 61(1);
  Baker, Bradley & Wurgler 2011, "Benchmarks as Limits to Arbitrage," *FAJ* 67(1)). Its **inversion** on
  this current-membership 2010–2026 panel — where high-beta names led the rally — is the study's central
  surprise.

## The shared method

- **Newey-West (1987)** HAC SEs · **Lo (2002)** Sharpe inference · **White (2000)** Reality Check —
  the shared [`quantlab/`](../../../quantlab/) engine; see [`METHODOLOGY.md`](../../../METHODOLOGY.md).
  Reproducibility stamp (as-of + fingerprint) via [`quantlab/repro.py`](../../../quantlab/repro.py).
