# Study 86 — Tail-Radar

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | SKEW quintile spread t = −1.40 (h=1d), −1.17 (h=5d), −2.12 (h=21d) — no horizon clears \|t\| ≥ 2 in the claimed direction; crash-frequency Fisher p = 0.655. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No positive gross edge at any horizon; SKEW adds nothing over VIX in regression (t_SKEW ≤ 1.6 everywhere). |
| **Radar works?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | High-SKEW regimes have a crash rate of 8.8% vs 9.1% in low-SKEW — marginally *lower*, not higher; the radar does not see the swans. |

> **In one sentence:** the CBOE SKEW index is not a black-swan radar — high SKEW readings carry no significant predictive power for forward SPY crashes or below-average returns, add nothing over VIX in a joint regression, and the crash rate is actually slightly *lower* after high-SKEW episodes.

## What we tested

The famous claim: when SKEW is high, sophisticated investors are pricing in crash risk via
expensive OTM put options, so the smart investor should reduce equity exposure or buy
protection. We take this literally and run three honest tests on 8,324 daily observations
(1993–2026): (1) sort days into SKEW quintiles and measure forward SPY returns at 1, 5, and
21 days — the claim predicts the top quintile should deliver the worst returns; (2) test
whether crash frequency (SPY < −5% over 21 days) is elevated in high-SKEW regimes via Fisher
exact test; (3) run a joint OLS regression of forward returns on standardised SKEW and VIX to
see if SKEW carries any incremental information beyond the simpler fear gauge. A deterministic
synthetic tape with a tunable SKEW-signal knob serves as a positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | SKEW explained plainly, the quintile chart in plain English, why the crash-frequency test is the real test, the 'smart money' story debunked |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats per quintile/horizon, Fisher test details, SKEW vs VIX regression with HAC-robust SEs, synthetic positive control sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`tail_radar/`](tail_radar/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
