# Study 208 — Gold-Miners

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Beta=1.73× (HAC *t*=35) is real but well-known and not exploitable; alpha=−10.6%/yr (*t*=−2.14) erases the leverage; asymmetry claim busted (*t*=−2.18, opposite direction). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | GDX CAGR 5.1% vs GLD 9.4%; Sharpe 0.12 vs 0.49; max drawdown −80.6%; miners-vs-bullion timing rule Sharpe 0.13 (*t*=0.62). |
| **Asymmetry** | ![Busted](https://img.shields.io/badge/Asymmetry_claim%3F-Busted-8b949e?style=flat-square) | beta_up=1.84 > beta_dn=1.62 (*t*=−2.18) — miners actually amplify gold upside *more* than downside, opposite to the popular narrative. |

> **In one sentence:** GDX amplifies gold at ~1.73× but a ~10.6%/yr operational alpha drain (significant), a ~26.7%/yr idiosyncratic vol tax, and GLD outperforming by ~4%/yr make miners-as-"leveraged-gold" a losing trade; the "more downside than upside" asymmetry is empirically rejected; and a GDX/GLD relative-strength timing rule (Sharpe 0.13) is a Mirage versus simply holding GLD (Sharpe 0.49).

## What we tested

The pitch: gold miners (GDX) are "leveraged gold" — up more than gold in bull markets and supposedly offering asymmetric upside. We test three precise claims against 20 years of daily GLD/GDX data (2006-2026, n=5,047 days):

1. **Is GDX actually leveraged gold?** OLS beta of GDX on GLD, with the alpha (operational drag vs bullion).
2. **Is the leverage symmetric?** Upside beta vs downside beta on gold-up vs gold-down days, with a HAC t-stat on the difference.
3. **Can a timing rule exploit the relationship?** GDX/GLD relative-strength signal (200-day SMA) vs holding GLD.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the leveraged-gold pitch, the real Sharpe comparison, the downside myth, why timing doesn't help |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | OLS beta/alpha with HAC inference, asymmetric beta sandwich estimator, timing rule HAC t-stat, positive synthetic controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`gold_miners/`](gold_miners/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
