# Study 239 — Spinoffs

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

Do corporate spin-offs really outrun the market after they are cut loose?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | 6m and 12m alpha just clear \|*t*\| = 2 on the hardcoded 14-event table (+11.9%, *t* = +2.09 at 6m; +16.9%, *t* = +2.07 at 12m), but n = 14 with extreme outliers and a curated table makes this fragile; 18m and 24m are below the bar. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Cannot be translated into a systematic strategy from this study alone: no selection rules, too-small n, illiquid early days, and outlier-driven alpha; a broader universe is needed. |
| **Greenblatt's forced-seller thesis?** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | Some large-cap spin-offs dramatically outperformed (CARR +89%, CEG +79%, GEV +108% alpha at 12m); others were disasters (BHF -49%, KVUE -56%, FOXA -48%). The average is positive, the dispersion is enormous, and the median at 18m is negative. |

> **In one sentence:** spin-offs show a positive average alpha in the first year on this curated 14-event sample (Greenblatt's "forced sellers" thesis), but the t-stats barely clear the bar, the table is cherry-picked, and three spectacular outliers explain most of the gain — this is a Weak signal at best.

## What we tested

Cusatis, Miles & Woolridge (1993) and Greenblatt (1997) argue that spin-off children
outperform the market in the first 1–3 years post-distribution: institutional sellers
with mandate constraints dump shares regardless of valuation, creating a buying
opportunity.  We test this on a hardcoded table of 14 notable US spin-offs (2011–2024),
computing buy-and-hold returns for the child ticker vs SPY over 6/12/18/24-month
horizons starting the day after the ex-distribution date.  A Newey-West HAC t-stat
on the alpha (child minus SPY) is the inference bar.

**Key limitations stated upfront:**
1. The 14-event table is curated — not a systematic universe of all spin-offs.
2. Spin-off children may be illiquid immediately post-distribution.
3. SPY is not a size-or-sector-matched benchmark; factor-adjusted alpha would be smaller.
4. The 2020 UTX-to-CARR/OTIS spin benefitted from the COVID energy sector recovery.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the horizon sweep, the individual event stories, why some win and others lose, the forced-seller hypothesis |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-horizon HAC t-stats, individual event table, synthetic positive control, cost sweep, factor caveat |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`spinoffs/`](spinoffs/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
