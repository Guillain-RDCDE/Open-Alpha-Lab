# Study 709 — World-Series-Effect ⚾📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | NL-champion → bullish-next-year: Welch *t* = **+1.07**, permutation *p* = **0.30**, hit rate **52.7%** — *below* the 73.0% unconditional up-rate that's the correct baseline. The New York-hometown variant runs **backwards** (*t* = **−0.33**, hit rate **39.2%**). n = 74 seasons, 1950→2024. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | An omen-timing strategy (hold only after a "bullish" title, else cash) underperforms buy-and-hold by **−3.1 pp/yr** (league) or **−6.0 pp/yr** (city) — sitting out a mostly-up market is expensive, and there's no signal underneath to pay for it. |
| **Beats a coin?** | ![Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square) | League-omen coin-test *p* = 0.73; city-omen coin-test *p* = 0.08 — and its 39.2% hit rate is numerically *worse* than 50%. Neither variant reliably beats a flat coin flip, let alone the market's own bias. |

> **In one sentence:** the World Series has a league winner every year and the S&P goes up
> most years anyway — dress the first fact up as an omen for the second and you get a
> Welch *t* under 1.1, a permutation *p* around 0.3, and a "hometown team" variant that
> doesn't even point the right way; this is a coincidence with a mascot, not an edge.

## What we tested

The claim, stated the way baseball-omen folklore states it: *the League (AL vs NL) of the
World Series champion — or, in its "hometown of Wall Street" cousin, whether a New York
franchise wins — predicts next year's stock market*, the same species of story as the
[Super Bowl Indicator](../158-super-bowl/). We hardcode all **76 World Series champions
1950→2025** (league, host city, New-York flag; 1994's players'-strike gap is named, not
imputed) in `data.py`, join them to yfinance **^GSPC** calendar-year returns, and test
both variants against the **correct baseline** — the sample's own 73.0% unconditional
up-rate, never a 50% coin — with a Welch *t*, a 20,000-draw permutation test on the mean
contrast, a Wilson-bounded hit rate, and an omen-timing strategy for the tradability
check. A 20-seed synthetic positive control confirms the machinery can find a signal when
one is planted; the real tape confirms there is none here. One documented,
zero-look-ahead execution convention throughout (enter at the season's December 31
close). **Dedup:** siblings [158-super-bowl](../158-super-bowl/) (the football original,
same debunk shape), [235-world-cup-effect](../235-world-cup-effect/) (a different,
during-tournament drift claim — no winner effect), [234-olympic-year](../234-olympic-year/)
(the same ^GSPC/permutation machinery, applied to a symmetric calendar marker) and
[708-eurovision-effect](../708-eurovision-effect/) (the same omen family, a song contest
instead of a ballgame) never test the World Series specifically — this is that corner of
the sports-omen family.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the base-rate trap, why "hometown team wins" doesn't even point the right way, the coin-flip myth-check in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/permutation/binomial anatomy for both signal variants, the omen-timing strategy vs buy-and-hold, and the 20-seed synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`world_series_effect/`](world_series_effect/). The champion table is
hardcoded from MLB's official postseason history; ^GSPC is a price index (no
survivorship). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
