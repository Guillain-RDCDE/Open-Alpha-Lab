# Study 289 — Diwali-Muhurat

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Post-Diwali INDA window return **−0.21%** excess vs a random window of the same length; t = **−0.87** (Newey-West −0.96), permutation p = **0.84**. n ≈ 14–23 events is far too small to resolve a sub-percent seasonal. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The tradeable proxy (INDA) does not even trade in the Indian evening Muhurat window; the closest version — a once-a-year long-only rotation — is dominated by simply holding the ETF, and costs only deepen the (negative) excess. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The "Muhurat returns" tallies you see each autumn test against a 50% coin and cherry-pick the one-hour INR session; against the honest up-drift baseline and a tradeable foreign proxy, the omen vanishes. |

> **In one sentence:** the Muhurat session is a lovely ritual but a non-event for a portfolio — the day after Diwali is, if anything, slightly *below* a random day, and there is no instrument a foreign investor can hold to capture the auspicious window anyway.

## What we tested

Indian folklore says holding equities across the Diwali (Laxmi Pujan) **Muhurat trading
session** brings an auspicious year. We make that tradeable for a non-Indian investor by
hardcoding all 23 Diwali dates (2003–2025) in `data.py` and joining the **iShares MSCI
India ETF (INDA)** as the proxy. For each Diwali we buy at the first INDA close strictly
*after* the event (a one-session execution lag — the date is public) and hold a fixed
window, then test the window return against the **honest baseline**: the unconditional
same-length forward-window mean, via a 10,000-draw block-permutation test plus plain and
Newey-West t-stats. The synthetic positive control (a planted +150 bps Diwali premium)
confirms the engine finds a real effect when one exists; the proxy tape has none. The
proxy/USD mismatch and the tiny n are named on the Signal axis. **This study ships no real
INDA cache**, so the headline numbers come from the deterministic synthetic NULL tape at
the study seed — call `data.fetch_inda(fetch=True)` once to verify on the live proxy.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the ritual, the base-rate trap, what "the day after Diwali" actually earns, in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | event study, block-permutation distribution, Newey-West t, costs, the n≈14 power calculation, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`diwali_muhurat/`](diwali_muhurat/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
