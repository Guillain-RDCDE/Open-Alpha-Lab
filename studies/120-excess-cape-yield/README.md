# Study 120 -- Excess-CAPE-Yield

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | ECY -> 10yr excess return: R² = **0.70**, OLS t = **5.27** on 14 non-overlapping 10-year windows (1882--2013); OOS R² = +0.42. Plain 1/CAPE alone: t = −0.27, R² = 0.006 -- the bond-yield adjustment is load-bearing. |
| **Tradability** -- does it survive costs, capacity and scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Signal requires a 10-year holding period; it is a valuation thermometer, not a stopwatch. No practical entry/exit rule, no short-term timing ability (1yr R² = 0.05). |
| **Forecaster vs timer?** | ![Confirmed](https://img.shields.io/badge/Forecaster_vs_timer%3F-Confirmed-8b949e?style=flat-square) | ECY is a REAL long-horizon forecaster (confirmed) and a MIRAGE short-horizon timer (rejected). These are two different claims; only the first holds. |

> **In one sentence:** Shiller's Excess CAPE Yield is a genuinely real 10-year equity risk premium gauge -- R² 0.70, t 5.27 on non-overlapping decades -- but the 10-year holding period makes it a tide table, not a trading signal.

## What we tested

Shiller and Bunn (2014) define the Excess CAPE Yield as ECY = 1/CAPE minus the real 10-year
bond yield (nominal yield minus trailing 12-month inflation). The idea: by netting out the
risk-free hurdle, ECY becomes a direct estimate of the forward equity risk premium and should
predict the next 10-year equity-minus-bond excess return better than 1/CAPE alone. We test
this on the full Shiller monthly dataset (1882--2013 regression sample, 14 non-overlapping
10-year windows), compare it to the plain earnings yield, compute an out-of-sample R², and
examine whether the 1-year horizon carries any of the signal -- it does not.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what ECY is, why the bond yield matters, the tide-table vs stopwatch distinction, the bucket ladder in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | overlapping-window R² vs non-overlapping t, ECY vs 1/CAPE comparison, OOS split, horizon decay, Valkanov inference caveat |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`excess_cape_yield/`](excess_cape_yield/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
