# Study 360 -- NAAIM-Exposure

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The contrarian *direction* is right and stable across every sub-period (weeks following manager pessimism +16.2%/yr vs all-in +8.6%/yr next week, gap +7.6 pp/yr), but the headline long-short HAC *t* = **+0.81** and the predictive-regression slope *t* = **-0.40** both fall far short of the *t* >= 2 bar; R-squared ~**0.03%**. The published series is current-vintage, not point-in-time. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A long/flat contrarian overlay nets **+8.7%/yr**, *below* buy-and-hold's **+12.0%/yr** (total return, 5 bps) -- managers go all-in during the meat of bull markets, so "fade the pros" sits out the weeks you most need. The edge never pays for the missed upside. |
| **Pros smarter than retail?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The pitch is that NAAIM is *professional* positioning, so fading it should beat fading a retail poll -- but it gives the **same** weak, untradeable tilt as the individual-investor AAII survey ([Study 257](../../257-aaii-sentiment/)). |

> **In one sentence:** the NAAIM Exposure Index -- active managers' actual weekly equity exposure -- does lean the contrarian way (weeks after the pros bail to cash beat weeks after they go all-in), but the effect never clears *t* >= 2, explains ~0.03% of next-week variance, loses outright to staying invested, and is no more bankable than the retail crowd's opinion.

## The claim

> *When the pros are all-in, should you sell? Is the NAAIM manager-exposure index a contrarian timing tool?*

## What we tested

Join the **weekly NAAIM Exposure Index** (NAAIM's free since-inception spreadsheet:
active managers' reported equity exposure, 0-200%) to **SPY total-return** closes,
2006-2026 (1,040 weeks); at each survey date the observed exposure predicts the SPY
return earned the *following* week (one-week execution lag). We run three honest
tests: a **regime sort** of next-week return by prior-exposure tercile, a
**predictive HAC regression** of next-week return on standardised exposure, and a
**long/flat contrarian overlay** pinned head-to-head against buy-and-hold net of
costs (total return on both legs). We add a sub-period breakdown and a deterministic
synthetic positive control that confirms the engine recovers a planted contrarian
edge (slope *t* = -6.12) and reads ~zero on the null (*t* = -0.37). The angle is
**professional positioning** -- the smart-money twin of the retail
([257](../../257-aaii-sentiment/)), options ([261](../../261-put-call-ratio/)) and
leverage ([260](../../260-margin-debt/)) gauges.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the index in plain language, the right-direction regime chart, and why a directionally-true "fade the pros" still loses to buy-and-hold |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | synthetic positive control, regime HAC t-stats, predictive HAC regression, timing overlay vs buy-and-hold, sub-period decomposition, the retail head-to-head |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`naaim_exposure/`](naaim_exposure/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
