# References & literature map — Study 25 (Clean-Slate)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entry is **strategy §3.7 (residual
  momentum)** — the same construction as price momentum (§3.1), but run on the residuals of a serial
  factor regression rather than total returns.

## The claim under test — the steelman

- **Residual momentum.** David Blitz, Joop Huij & Martin Martens, *"Residual Momentum"*, **Journal of
  Empirical Finance** 18(3), 2011: estimate each stock's exposure to the Fama-French factors over a
  trailing window, take the residual returns, and run 12-1 momentum on them. The result delivers a
  similar return to conventional momentum with **roughly half the volatility** and far smaller crashes,
  because the residual is purged of the time-varying factor exposures that drive momentum's tail.
- **The factor model.** Eugene Fama & Kenneth French, *"Common Risk Factors in the Returns on Stocks and
  Bonds"*, **Journal of Financial Economics** 1993 — the MKT/SMB/HML factors the source residualises
  against (we use a 1-factor market residual; see caveats).

## Why the crash lives in the systematic part — `WEAK` / `FRAGILE` / `Confirmed`

- **Momentum crashes are a systematic-exposure phenomenon.** Kent Daniel & Tobias Moskowitz, *"Momentum
  Crashes"*, **Journal of Financial Economics** 2016: after a market crash, past losers are high-beta and
  past winners low-beta, so a momentum book becomes implicitly short the market just as it rebounds —
  the crash is in the *beta* (and value) exposure. Residualising removes exactly that, which is why
  residual momentum has a milder tail.
- **Risk-managed momentum.** Pedro Barroso & Pedro Santa-Clara, *"Momentum Has Its Moments"*, **Journal
  of Financial Economics** 2015 — vol-scaling the factor; the desk's [Study 16](../../16-storm-shy/)
  overlay, stacked with residualisation in the beat-7 complement.
- **Decay.** As in [Study 24](../../24-stampede/), the premium is faint on the modern, survivorship-
  biased large-cap sample regardless of residualisation — about *this* sample, not the long-run effect.

## The desk's own method — engine and reproducibility

- **Causal residualisation.** A trailing-window rolling-beta regression (no full-sample look-ahead — the
  trap [Study 22](../../22-crystal-ball/) dissects), lagged one day.
- **HAC / Newey–West inference** (Newey & West, *Econometrica* 1987) on the residual-WML alpha;
  reproducibility via [`quantlab.repro`](../../../quantlab/repro.py) and [`quantlab.universe`](../../../quantlab/universe.py).

## Caveats stated in the open (house rule)

- **1-factor (market) residual, not FF3.** We lack SMB/HML here, so we residualise against the
  equal-weight market only. This captures the beta-driven crash but *not* the value-driven part, which is
  why the standalone drawdown reduction is incremental; an FF3 residual (the source's recipe) is the
  first beat-7 fork.
- **Survivorship bias; total-return closes; current large-cap universe.** As in [Study 24](../../24-stampede/) —
  the ranking is robust, magnitudes are not, and the modern window is one regime.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
