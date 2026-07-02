# Results — Study 543 (Western-Zodiac-CEO): does a CEO's sun sign predict returns?

*Generated from [`western_zodiac_ceo/`](../western_zodiac_ceo/) over this study's cached yfinance
tape: daily adjusted close for a **hand-curated 32-name large-cap CEO table** (31 with a full price
history; SQ/Block delisted from the ticker over the window), fingerprint `c391b6621701`,
2018-01-02 → 2026-06-26. Each name's **sun sign** is derived from its CEO's public birth date
(standard western tropical cutoffs); the **forward return** is real. Scored as-of **2022-06-30**,
forward holding window **2022-06-30 → 2026-06-26** (4 years). Cross-section fingerprint
`67cb7992be0c`. As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Lucky sign?" `BUSTED`

The folklore: a company run by a CEO of the "right" star sign outperforms — the western-astrology
cousin of the Chinese zodiac-year claim ([Study 165](../../165-chinese-zodiac/)). We hand-curate a
32-name large-cap table mapping each CEO to their **sun sign** (from a public birth date), sort the
cross-section by sign, and test whether the sign means differ.

They do not. A one-way **ANOVA across the eleven signs present** gives **F = 0.69** (analytic
*p* = 0.72), and a **label-shuffle placebo** puts the permutation *p* at **0.70** — the sign means
scatter exactly like noise. The folklore's best shot — long the *best* sign, short the rest — picks
**Aquarius** (mean forward return **+466%** vs the rest's **+202%**, spread **+264 pts**), but its
Welch *t* is only **+0.73**, and once you correct for having *selected* the best sign (a
max-statistic shuffle null) the spread's *p* is **0.83** — utterly unremarkable. Across five forward
windows the winning sign wanders (Cancer → Cancer → Leo → Aquarius → Aquarius) and the ANOVA never
clears *p* < 0.23. So `NONE` on the signal axis (nothing survives, and a hand-curated ~30-name table
can never certify a `REAL` signal anyway), `MIRAGE` on tradability (a 12-way sort on 2-3 names per
cell is a multiple-comparisons mirage that dies at any cost), and `BUSTED` on the folklore itself.

## Data stamp

- **Prices**: 31 large-cap survivors, daily adjusted close, 2018-01-02 → 2026-06-26, fingerprint
  `c391b6621701`
- **Curated CEO → sign table**: 32 names (public birth dates → western tropical sun sign); one
  name (SQ/Block) lacks a full price history over the window
- **Cross-section** (scored 2022-06-30, forward to 2026-06-26): 31 names, fingerprint `67cb7992be0c`

## The sign sort — noise, dressed as astrology

| | value |
|---|---|
| Signs present in the panel | **11** (Scorpio has 1 name; others 2-4) |
| ANOVA F across signs | **0.69** |
| ANOVA analytic *p* | **0.72** |
| Label-shuffle permutation *p* (on F) | **0.70** |

An F below 1 means the between-sign variation is *smaller* than the within-sign variation — the
signs carry no information about forward returns. The permutation *p* of 0.70 confirms it: seven in
ten random re-labellings produce an F at least this large.

## The folklore's best shot — long the best sign, short the rest

| | value |
|---|---|
| Best sign (selected post-hoc) | **Aquarius** (NVDA, COIN, DIS) |
| Best-sign mean forward return | **+466%** |
| Rest mean forward return | **+202%** |
| Best − rest spread | **+264 pts** |
| Welch *t* (best vs rest) | **+0.73** |
| Max-statistic placebo *p* (re-selecting best sign each shuffle) | **0.83** |

Aquarius "wins" only because it happens to contain NVDA — a single mega-winner in a 4-year AI
melt-up. Correct for the fact that we *chose* the best of twelve signs after seeing the data, and
the spread is comfortably inside the null (*p* = 0.83). This is the small-sample, multiple-comparisons
trap, not a signal.

## Costs — a footnote to a non-signal

| | value |
|---|---|
| Gross best-vs-rest spread (headline window) | **+264 pts** |
| Net (5 bps/leg round-trip + 100 bps/yr borrow on the short leg, 4y hold) | **+259 pts** |

Costs barely move the number — but there is nothing to trade: the "spread" is a post-hoc pick with
*p* = 0.83, and the winning sign changes every window.

## Robustness — the winning sign wanders, F never clears

| Forward window | ANOVA F | Perm *p* | Winning sign | Best − rest spread |
|---|---|---|---|---|
| 2019-06 → 2021-06 | 1.34 | 0.230 | Cancer | +6.55 |
| 2020-06 → 2022-06 | 0.95 | 0.502 | Cancer | +0.96 |
| 2021-06 → 2023-06 | 0.72 | 0.679 | Leo | +0.45 |
| 2022-06 → 2024-06 | 1.21 | 0.293 | Aquarius | +2.80 |
| 2022-06 → 2026-06 (headline) | 0.69 | 0.687 | Aquarius | +2.64 |

The ANOVA never approaches significance (best *p* = 0.23), and the "lucky sign" is unstable —
Cancer, then Leo, then Aquarius. A signal that changes identity every window is not a signal.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `sign_alpha` (Aries bump) | Mean ANOVA F (25 seeds) | Mean permutation *p* | |
|---|---|---|---|
| 0.00 (null) | **1.16** | **0.444** | flat — no false signal |
| 0.15 | 1.72 | 0.202 | effect emerging |
| 0.30 | 3.27 | 0.039 | clears the bar |
| 0.50 | **6.88** | **0.002** | unmistakable |

At the null (`sign_alpha = 0`) the ANOVA F sits near 1 and the permutation *p* near 0.5 — no false
signal. Planting a genuine sign effect (one sign given a real alpha bump) drives F up and the
permutation *p* toward 0. The detector works — so the flat real-tape result is a statement about
**the folklore**, not a broken engine. (Control only; never cited for the real-tape stamp.)

## Why this can never be REAL

1. **No mechanism, fixed label.** A sun sign is set at birth and never changes over a CEO's tenure.
   There is no time series to average — only one cross-section of ~30 names — so the study is
   structurally capped below `REAL` on the SIGNAL axis whatever the tape prints.
2. **A 12-way test on 2-3 names per cell.** Eleven signs over 31 names is hopelessly underpowered,
   exactly as [Study 165](../../165-chinese-zodiac/)'s "12 animals × ~3 years" is. The best-sign
   pick is a multiple-comparisons artifact that the max-statistic null erases.
3. **Curated, tiny tape.** A hand-curated table of well-known CEOs is neither survivorship-free nor
   powered; it exists to *disprove* the folklore cleanly, not to certify a premium.

## The honest takeaway

CEO sun sign predicts nothing. The ANOVA across signs is F = 0.69 (*p* = 0.72, permutation
*p* = 0.70); the folklore's best shot — long Aquarius, short the rest — is a post-hoc pick that
dies at *p* = 0.83 and whose winning sign changes every window. `NONE` × `MIRAGE`, folklore
`BUSTED`. The synthetic control confirms the engine would light up for a real sign effect — so this
is the horoscope talking, not the code.
