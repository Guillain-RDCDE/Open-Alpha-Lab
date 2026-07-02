# Study 543 — Western-Zodiac-CEO ♌

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a CEO's sun sign predict returns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | One-way ANOVA across the 11 signs present: **F = 0.69** (analytic *p* 0.72, label-shuffle permutation *p* **0.70**). The folklore's best shot — long the best sign, short the rest — picks Aquarius (spread **+264 pts**) but Welch *t* **+0.73**, and the max-statistic placebo (re-selecting the best sign each shuffle) puts *p* at **0.83**. Curated 31-name tape — structurally can't certify REAL. |
| **Tradability** — does the sort pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A 12-way sort on **2-3 names per sign** is a multiple-comparisons mirage. The winning sign "wins" on one mega-cap (NVDA in Aquarius); gross **+264 pts** → net **+259 pts**, but the pick is *p* 0.83 and the winner changes every window. Nothing to trade. |
| **Lucky sign?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Across five forward windows the "best" sign wanders (Cancer → Cancer → Leo → Aquarius → Aquarius) and the ANOVA never clears *p* < 0.23. Eleven signs over 31 names is hopelessly underpowered — the horoscope's C-suite edition. |

> **In one sentence:** the western-astrology cousin of the Chinese-zodiac study is a textbook small-sample mirage — a hand-curated 32-name CEO table split twelve ways leaves 2-3 names per sign, the sign means scatter like noise (ANOVA F 0.69, permutation *p* 0.70), and the folklore's "best sign" is a post-hoc pick riding one AI mega-winner that dies the moment you correct for selection (*p* 0.83) and changes identity every window.

## What we tested

The folklore: a company run by a CEO of the "right" **western sun sign** outperforms — the
horoscope applied to the C-suite, and the sibling of [Study 165 — Chinese-Zodiac](../../165-chinese-zodiac/)
(which tests the *zodiac year*). We hand-curate a **32-name large-cap CEO table** mapping each chief
to their sun sign from a **public birth date** (standard western tropical cutoffs), pull **real**
yfinance forward returns, and run the honest, underpowered test the claim deserves: a one-way
**ANOVA** across the signs, a **label-shuffle permutation** null, the folklore's tradable
best-sign-vs-rest spread with a **max-statistic** correction for post-hoc selection, costs + a short
borrow, a **five-window** robustness sweep, and a deterministic, seed-robust **synthetic positive
control** that plants a sign effect and proves the engine catches it. Nothing survives. Because a
sun sign is fixed at birth (no time series to average) and the table is tiny, the Signal axis is
capped below `REAL` by construction — this study exists to *disprove* the folklore cleanly.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the claim is, the CEO birth-date table, why 12 signs over 30 names is not enough, and why "Aquarius won" is a mirage |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the ANOVA F, the label-shuffle + max-statistic placebos, the best-vs-rest Welch *t*, the five-window sign-wander, costs + borrow, and the seed-robust synthetic positive control |

The fingerprinted real-data run (31 CEOs, scored 2022-06, forward to 2026-06, panel fp
`67cb7992be0c`) is in [docs/results.md](docs/results.md); the offline machinery proof runs on the
deterministic synthetic world in [`western_zodiac_ceo/data.py`](western_zodiac_ceo/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`western_zodiac_ceo/`](western_zodiac_ceo/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
