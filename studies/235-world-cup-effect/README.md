# Study 235 — World-Cup-Effect

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Per-edition t vs unconditional mean = **-2.52**, p = **0.022** on n = 19 tournaments; but the result is dominated by known macro crises (Korean War 1950, oil shock 1974, dot-com 2002) that coincide with WC windows — football sentiment is unidentifiable as a separate driver. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | One 5-week window every 4 years is not a systematic strategy; the ~3.5 bps/day unconditional market drift you forgo during positive WC windows cancels any timing edge. |
| **Confirmed?** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | Edmans et al. (2007) find real country-specific elimination effects; the S&P 500 global version tested here is confounded and not robustly separable from macro crises. |

> **In one sentence:** the S&P 500 does tend to drift lower during FIFA World Cup summers, but the driver is macro crises that happened to coincide with 3–4 key tournaments — not football sentiment — and with n = 19, the signal is too fragile to trade.

## What we tested

Does the S&P 500 systematically underperform during FIFA World Cup tournament
windows (1950–2022)? We hardcode all 19 World Cup start/end dates in `data.py`,
fetch S&P 500 daily data via yfinance (^GSPC, 1950–2023), and compare returns
during the ~32-trading-day WC window against (1) all other trading days and
(2) same-duration matched control windows 2 years prior.

The per-edition t-test (the correct unit of inference — 19 independent
tournaments, not 340 autocorrelated daily observations) gives t = **-2.52**,
p = **0.022** vs the unconditional daily mean. But 11/19 windows are negative
(binom p = 0.32) and the biggest losers are the **Korean War summer (1950)**,
the **oil crisis (1974)**, and the **dot-com bust (2002)** — macro crises, not
football attention. The signal is directional, fragile, and not investable.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the timeline of WC windows vs S&P crashes, the confound story in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-edition t-test vs unconditional, permutation distribution, matched-control comparison, power calculation |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`world_cup_effect/`](world_cup_effect/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
