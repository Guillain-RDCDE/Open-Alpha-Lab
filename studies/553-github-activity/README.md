# Study 553 — GitHub-Activity 🐙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Do surges in a tech company's open-source commit activity foreshadow its stock — engineering as an alt-data signal?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does GitHub velocity nowcast returns? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | On a synthetic tape with a **realistic** planted effect (`ic = 0.03`) the mean cross-sectional IC is **+0.045** — the *right sign*, but at *t* = **1.64** it **fails the ≥ 2 bar** (placebo *p* **0.122**), and its significance **flickers** across sub-samples (IC-*t* 0.89 / 1.37 by half; long-short *t* 0.71 → 2.25 by split). And it can **never** be `REAL`: no point-in-time GitHub→ticker tape exists on a free stack, so the ceiling is `WEAK` by construction. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | You cannot build the signal a backtest needs — GitHub feeds are rate-limited *current* snapshots (renamed / archived / privatised repos vanish), so the real velocity is un-reconstructable and any free backtest is survivorship-/look-ahead-contaminated. On top of that the trade shorts the laggards (borrow) and the gross spread (**+14.1%/yr**, *t* 1.51 → net **+12.3%/yr**) never clears significance. |

> **In one sentence:** reading a tech firm's open-source commit/star velocity as an innovation nowcast is a *plausible, right-signed* idea — the engine recovers a planted effect cleanly — but at a realistic alt-data strength it lands at IC-*t* **1.64** (placebo *p* 0.12), is sign-stable yet significance-fragile across cuts, and can never be certified `REAL` because the point-in-time GitHub→ticker tape it would require does not exist on a free stack.

## What we tested

The alt-data dream: a listed tech firm's **public GitHub velocity** — the blended run-rate of
commits, merged pull-requests and new stars across its open-source org, z-scored across the field —
is a live *innovation-intensity* nowcast, so a **surge in velocity should foreshadow forward
returns** (the software-native cousin of R&D-intensity return studies:
[Chan-Lakonishok-Sougiannis 2001](docs/references.md), Hirshleifer-Hsu-Li 2013). Because **no free,
point-in-time, survivorship-clean mapping from GitHub orgs to tickers exists** (renames, archives,
private moves, rate limits, snapshot-only feeds), this study is **synthetic-only** — a deterministic
30-firm × 40-quarter panel with a single knob `ic` that plants the believers' effect. We compute the
per-period cross-sectional **Spearman IC** and its *t*-stat (the headline nowcast test), a
**label-shuffle placebo** null, a long-top/short-bottom decile book with costs + a short borrow, a
tail-fraction / sub-sample **robustness sweep**, and a seed-robust synthetic positive control (25
seeds) proving the engine catches a planted nowcast and stays flat at the null. The
data-availability limitation is named on the Signal axis — and a synthetic-only study can never earn
`REAL`. *Distinct from [400 Patent-Intensity](../400-patent-intensity/) (audited R&D-intensity
tertile trade) — this is the **public open-source telemetry** cross-sectional IC nowcast.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "read the factory floor of software" means, why a small-but-real signal still isn't a trade, and why we can't even get honest data — in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the per-period Spearman IC and its *t*, the within-period label-shuffle placebo, the decile long-short with costs + borrow, the tail/sub-sample robustness sweep, and the seed-robust synthetic positive control |

The fingerprinted headline run (synthetic panel, `ic = 0.03`, seed 553, panel fp `33d0732cb08f`,
as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery runs entirely on
the deterministic synthetic world in [`github_activity/data.py`](github_activity/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`github_activity/`](github_activity/). Synthetic-only — the point-in-time GitHub→ticker tape does not exist on a free stack. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
