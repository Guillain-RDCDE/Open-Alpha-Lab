# Results — Study 548 (Happiness-Index-Country): do the happiest nations' markets win?

*Generated from [`happiness_index_country/`](../happiness_index_country/). The **World Happiness
Report 2024** country rank (1 = happiest) is a hardcoded public snapshot restricted to the
**investable** set — 24 countries with a liquid, USD-denominated single-country ETF (in
[`data.py`](../happiness_index_country/data.py)). Each rank is joined to the ETF's forward
**total return** (yfinance daily adjusted close, cached under `_cache/`, fingerprint `483868490082`,
2015-01-02 → 2026-06-26). Happiness is scored as-of the WHR-2024 publication (**2024-03-20**); the
forward holding window is **2024-03-20 → 2026-06-26**. Cross-section fingerprint `90b4d9c25e41`.
As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Spurious?" `BUSTED`

The folklore: sort the world's equity markets by their **World Happiness Report** rank, go long the
happiest nations and short the gloomiest, and collect a spread ("optimistic, well-run societies must
have better markets"). We build the investable cross-section — 24 countries with a liquid
single-country ETF — sort into terciles by WHR rank, and test whether the happy markets beat the
gloomy ones.

On the **headline 2024-03 → 2026-06 window the folklore is absent, if anything inverted.** The
*gloomy* tercile earned **+87.0%** versus the *happy* tercile's **+44.6%** — a long-happy /
short-gloomy spread of **−42.4%** (Welch two-sample *t* = **−1.65**, placebo *p* = 0.066). The
Spearman rank correlation between happiness and forward return is **−0.19** (*t* −0.90) — a nothing.
And the sign **flips across windows**: it was *positive* (happy wins, spread +24.4%, slope-*t* +1.80)
over the 2019-21 window, faded to noise through 2021-24, and turned *negative* recently. No stable
*t* ≥ 2 in either direction — the textbook signature of a **spurious cross-country correlation** on a
tiny (n = 24) cross-section. So `NONE` on the signal axis, `MIRAGE` on tradability, `BUSTED` as a
spurious-correlation myth-check.

## Data stamp

- **Country ETFs**: 24 liquid single-country ETFs (total return), daily adjusted close, 2015-01-02
  → 2026-06-26, fingerprint `483868490082`
- **WHR-2024 ranks**: hardcoded public snapshot, 24 investable countries (1 = happiest)
- **Cross-section** (scored 2024-03-20, forward to 2026-06-26): 24 countries, fingerprint `90b4d9c25e41`

## The happiness sort — the folklore is the WRONG WAY ROUND here

| Tercile (7 countries) | Forward return 2024-03 → 2026-06 |
|---|---|
| **Happy** (best WHR ranks: Finland, Denmark, Sweden, Israel, Netherlands, Norway, Switzerland) | **+44.6%** |
| **Gloomy** (worst WHR ranks: Singapore, Spain, Italy, Brazil, Japan, Korea, South Africa) | **+87.0%** |
| **Spread (happy − gloomy)** | **−42.4%** (Welch *t* −1.65, placebo *p* 0.066) |

The folklore predicts happy > gloomy (a *positive* spread). The tape delivers the opposite: the
gloomy bucket — Korea (+209%), Spain, Italy, South Africa — roughly *doubled* the happy Nordic
bucket. The placebo *p* = **0.066** says even this inversion is not quite distinguishable from noise
on n = 24: it is a *non-result*, not an anti-signal.

## The rank correlation & the country-level relation

| | value |
|---|---|
| Spearman rho(happiness rank, forward return) | **−0.19** (*t* −0.90) |
| Pearson corr(happiness score, forward return) | **−0.38** |
| Slope (forward_ret on happiness score) | **−69.1%** per unit |
| Slope *t* | **−1.92** (a *positive* slope would be the folklore) |

The single number the folklore lives on — the rank correlation — is **−0.19 and insignificant**. The
OLS slope leans *negative* (*t* −1.92, i.e. flirting with an *anti*-folklore reading) but does not
clear |*t*| = 2, and its sign is unstable across windows (below).

## Robustness — the sign is not stable

| WHR edition → hold | Happy − gloomy spread | Slope *t* | Spearman rho | Reads as |
|---|---|---|---|---|
| 2019 (2019-03 → 2021-03) | **+24.4%** | **+1.80** | +0.45 | folklore present (weak) |
| 2021 (2021-03 → 2023-03) | **+2.7%** | +0.90 | +0.03 | noise |
| 2022 (2022-03 → 2024-03) | **−7.1%** | +0.26 | −0.23 | noise |
| 2024 (2024-03 → 2026-06, headline) | **−42.4%** | **−1.92** | −0.19 | inverted (weak) |

The sign is *positive* early, drifts to zero, then turns *negative* — a signal whose direction
depends on which WHR edition you pick is not a signal at all. It is exactly what a spurious
correlation on 24 data points looks like.

## Costs

| | value |
|---|---|
| Gross spread (happy − gloomy, headline window) | **−42.4%** |
| Net (5 bps/leg round-trip + 60 bps/yr borrow, 2y hold) | **−43.8%** |

Costs are a footnote: the trade is the *wrong sign* before you pay for it. Country ETFs are cheap to
borrow, so the borrow barely moves the number — there is simply nothing to harvest.

## Synthetic positive control — the engine is faithful (seed-robust, 40 seeds)

| Planted `rank_alpha` | Mean slope-*t* (40 seeds) | Mean spread-*t* (40 seeds) | |
|---|---|---|---|
| 0.00 (null) | **−0.11** | −0.13 | flat — no false signal |
| +0.03 | +0.33 | +0.29 | effect emerging |
| +0.06 | +0.78 | +0.71 | effect visible |
| +0.10 | +1.38 | +1.27 | approaching the bar |
| +0.15 | **+2.13** | +1.96 | clears the bar |

At the null both statistics sit at ≈ 0; planting a genuine happiness effect (`rank_alpha > 0`)
drives them positive and past +2 as it grows. The detector works on n = 24 — so the real-tape
non-result is a statement about **this cross-section**, not a broken engine. (Control only; never
cited for the real-tape stamp.)

## Why this can never be REAL here (named on the SIGNAL axis)

1. **Tiny investable cross-section.** Only ~24 countries have a liquid single-country ETF, so the
   whole test rests on ~24 data points and ~7 per bucket. Standard errors are enormous; a robust
   *t* ≥ 2 across countries is essentially unreachable. Synthetic-only ⇒ capped at WEAK/NONE by the
   desk rule.
2. **No clean point-in-time panel.** WHR methodology has shifted across editions, and ETF total
   returns are currency- and dividend-mangled. There is no free, long, aligned happiness × tradable
   country-index panel — so the "real" tape here is *illustrative*, never certifying.
3. **Spurious by construction.** Happiness rank correlates with a dozen macro variables (GDP,
   governance, sector mix, currency regime). Any cross-country return sort will pick up whatever
   macro factor happened to lead — this window it was EM/Asia, so gloomy "won".

## The honest takeaway

Do the happiest nations' stock markets beat the gloomiest? On the investable cross-section, **no** —
and not even reliably the *other* way. The rank correlation is −0.19 (insignificant), the
tercile spread is the wrong sign (−42%, *t* −1.65, placebo *p* 0.066), and the sign flips across WHR
editions. `NONE` × `MIRAGE`, `BUSTED` as a spurious-correlation myth. The synthetic control confirms
the engine would catch a real happiness effect if one existed — so this is 24 noisy data points
talking, not the code.
