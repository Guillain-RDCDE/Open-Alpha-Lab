# Study 193 — Window-Dressing

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Pump contrast = **−2.60 bps/day** (wrong sign, Welch p = 0.635); reversal contrast = **+6.01 bps** (wrong sign); spread HAC t = **−1.39**. All Bonferroni-corrected window-sweep p-values exceed 0.19. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross spread is negative (−3.96 bps/day); no break-even cost exists. ~40 active days per year means any round-trip cost adds to a loser. |
| **Quarter-end pattern** | ![Inverted](https://img.shields.io/badge/Quarter--end_pattern-Inverted-8b949e?style=flat-square) | On SPY 1993–2026 the pump window is slightly *below* baseline and early-quarter is slightly *above* — the opposite of the Carhart textbook story. |

> **In one sentence:** Carhart et al. (2002) documented quarter-end portfolio pumping in mutual funds in 1985–1994; on SPY from 1993 to 2026 the pump window is slightly below baseline and the reversal window above — neither contrast is statistically significant, and the long-pump/short-reversal spread earns −4 bps/day with HAC t = −1.4.

## What we tested

The academic claim: mutual fund managers inflate holdings prices in the last few days of each
calendar quarter ("window dressing") and prices mechanically reverse in the first days of
the next quarter. We test this on **SPY daily total-return prices from 1993-02-01 to 2026-06-12**
(8,399 days, 670 pump days, 669 reversal days) with:

1. **Pump vs baseline** — last 5 trading days of each calendar quarter vs all other days.
2. **Reversal vs baseline** — first 5 trading days of the next quarter vs all other days.
3. **Long-pump / short-reversal spread** vs a random-day placebo (same number of long/short days,
   random selection) — isolates the timing signal from the equity premium.
4. **Window sweep** 1–10 days with Bonferroni correction — guards against a cherry-picked window.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the intuition, the quarter-end calendar, the textbook claim, and what the data actually shows — in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, Welch tests, Bonferroni window sweep, random-day control, sub-period breakdown, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`window_dressing/`](window_dressing/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
