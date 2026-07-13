# Study 734 — NBA-Finals-Effect 🏀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the losing city's market dip (champion's pop)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Not one of six cuts clears \|*t*\| ≥ 2. The EGN loser dip is a placebo-consistent whisper (next-day AR **−0.149%**, *t* = **−0.47**, placebo *p* = **0.325**) gone within a week; the champion "feel-good pop" runs the *wrong way* (**−0.354%**, *t* = **−1.22**); the broad-`SPY` cross-check is a flat **+0.139%** (*t* = **+0.81**) — exactly the within-country cancellation the mechanism predicts. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No net-of-cost, zero-look-ahead capture clears \|*t*\| ≥ 2. The one cut that grazes it (champion/1-week net @ 10 bps, *t* = **−1.99**) is wrong-signed for the folklore and evaporates under its own placebo (*p* = **0.115**). |
| **Does the loser's city dip?** | ![Busted](https://img.shields.io/badge/Does_the_loser's_city_dip%3F-Busted-8b949e?style=flat-square) | 26 Finals, an event study, a random-window placebo, a jackknife and a champion-vs-loser Welch contrast (*t* = **−0.48** to **−0.72**, backwards) all agree: no detectable home-market dip, no pop — the cross-country design collapses because both teams share one US tape. |

> **In one sentence:** Edmans-García-Norli's real "eliminated-country market dips next day"
> effect needs the two teams to trade on *different* national markets — but 25 of 26 NBA
> Finals since 2000 are USA-vs-USA on one shared tape, so the mood nets out, the loser's
> home-city proxy barely twitches (*t* = −0.47, placebo *p* = 0.33), and the champion's city
> actually drifts the wrong way.

## What we tested

The sports-radio folklore, steelmanned via a real finding: Edmans, García & Norli (2007,
*[Sports Sentiment and Stock Returns](https://doi.org/10.1111/j.1540-6261.2007.01262.x)*, JF)
showed a country's stock market really does fall ~49 bps the day after it's *eliminated* from
the soccer World Cup — a genuine mood-to-market channel. The NBA version: when a team loses
the Finals its home city should dip, and the champion's should pop. We hardcode all **26 NBA
Finals 2000→2025** (champion, runner-up, exact clinching-game date) and, since no US city has
a stock index, map each metro to a single **real, tradable, deliberately-coarse hometown
large-cap proxy** (Lakers→`DIS`, Celtics→`STT`, Spurs→`CFR`, … Raptors→`EWC`), then measure
its abnormal return vs the `SPY` US market from the last pre-result close through the next day
and one week — with a random-window placebo, a jackknife, a broad-market cross-check, and a
zero-look-ahead tradable-capture test. **As-of 2026-06-30.**

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the mechanism *should* work, why the shared US market kills it before you start, the loser dip that isn't, the champion pop that points backwards |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-*t* battery, the broad-`SPY` cancellation check, the random-window placebo, the jackknife, the event anatomy, the champion-vs-loser Welch split, the costed capture, the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`nba_finals_effect/`](nba_finals_effect/). The Finals calendar is hardcoded from
Basketball-Reference; proxies + `SPY` are yfinance total-return closes. **Labelled proxy:**
each metro is a single hometown large-cap standing in — coarsely — for a nonexistent city
index; the noisiness is named on the Signal axis. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
