# Study 226 — Crude-Seasonality

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does crude reliably rally in spring? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The spring-vs-autumn spread clears |t| ≥ 2 on the full sample (**t = 2.47**, Apr–Jun mean +2.76% vs Aug–Nov mean −1.62%), but no individual month survives Bonferroni correction, and the effect is insignificant in the 2000–2012 sub-period (t = 1.42). Regime-dependent, not a law. |
| **Tradability** — does the calendar timer add value? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Against the investable proxy (XLE), the long-spring/short-autumn timer earns **Sharpe 0.04** vs buy-and-hold's **0.35** (excess of T-bill, both legs) — an 88% reduction in risk-adjusted return. The crude timer beats buying CL=F, but crude itself is a zero-real-return commodity. |
| **"Every year"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The pattern is not there unconditionally: insignificant in 2000–2012 (t = 1.42), only marginally significant 2013-on (t = 2.02). Year-by-year counts show large variation with no reliable per-year edge. The "every year" framing is false. |

> **Does crude really rally into summer driving season every year?** It rallies more often than it falls in spring — but "more often" and "reliably enough to trade" are not the same thing. The spread is real but fragile and regime-dependent; the calendar timer wrecks your energy-equity performance.

> **In one sentence:** a spring-positive / autumn-negative pattern in crude passes the basic t-test on the full 2000–2026 sample but fails Bonferroni correction, falls apart in the pre-shale decade, and produces a near-zero Sharpe when applied to the investable energy-equity market.

## What we tested

The claim: April–June, refineries switch to pricier summer gasoline blends and driving demand peaks,
so crude should rally; August–November, driving season ends and refinery maintenance peaks, so crude
should weaken. We test it on **every calendar month of WTI crude futures (CL=F)** data available
(2000-09 → 2026-05, 309 months on a verified hole-free monthly grid built from daily closes), plus
energy equities (XLE, 1999–2026): (1) per-month t-stats with Bonferroni adjustment for 12 tests,
(2) a spring-vs-autumn Welch t-test, (3) a calendar timer (long spring, short autumn, T-bill
otherwise) vs buy-and-hold on both crude and energy equities, and (4) a 2000–2012 / 2013-on
sub-period split. The offline control is a synthetic world with a tunable spring premium and a null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a compelling seasonal narrative can be real sometimes and unreliable always |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-month t-stats, Bonferroni correction, spring-vs-autumn spread, timer race, sub-period split |

The fingerprinted real-data run (CL=F + XLE + ^IRX, 2000–2026, fp `04d20473b062`) is in
[docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py)
(`--fetch` to download); the offline machinery proof runs on the synthetic world in
[crude_seasonality/data.py](crude_seasonality/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
