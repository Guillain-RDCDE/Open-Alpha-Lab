# Study 739 — Wildfire-Season 🔥

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a California wildfire hit the utility/insurer basket? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Ignition day is flat: basket **−22.77 bps**, *t* = **−0.92**, placebo *p* = **0.24**. The real craters arrive in the days *after* ignition (liability news), but [+1..+5] doesn't clear the bar either (*t* = −1.41, bootstrap CI crosses zero, **14/14** leave-one-out below 2). The best cut — utility leg, 7 utility-caused fires — reaches only *t* = **−1.96**, fragile (5/7 drops below 2, Camp/Eaton-driven). Insurers barely react (*t* = −0.60). |
| **Tradability** — can you short the fire headline? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The short has a positive *mean* (+228 bps at 5d) but a tiny *median* (+31 bps) — a 2-jackpot lottery inside 14 bets, never significant net of borrow (best *t* = +1.19), decaying to a **43% win rate** by 21 days. Negative-carry bet on a rare, un-timeable liability crater. |
| **A sell-in-July fire-season seasonal?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The Jul→Dec fire window is marginally *positive* for the basket (**wrong sign**, Welch *t* = +1.0, random-window *p* = 0.30). The calendar doesn't price California fire risk in advance. |

> **In one sentence:** across 14 major California wildfires 2003→2025, the exposed
> utility+insurer basket does nothing on the ignition day (*t* = −0.92) and the real
> post-fire crater is a *delayed, utility-specific, inverse-condemnation liability
> repricing concentrated in two megafires* (Camp, Eaton) — not the same-day,
> basket-wide, seasonal, tradable pattern the folklore sells; insurers barely move and
> the "sell in July" seasonal points the wrong way.

## What we tested

Every California fire season, sector commentary revives a tradable story: a major
wildfire should hammer the state's investor-owned utilities (Edison / `EIX`, PG&E /
`PCG` — genuinely exposed via California's strict **inverse-condemnation** liability)
and property insurers (`ALL`/`TRV`/`MCY`/`CB`), and the whole Jul→Dec fire window
should carry a worse basket return than the rest of the year. We steelman it on a
hand-curated table of **14 major California wildfires 2003→2025** (each flagged by
whether a utility's equipment was the cause), run an event study on the utility+insurer
basket around each ignition (abnormal returns, the [−1..+5] anatomy, a random-calendar
placebo), split the utility leg from the insurer leg, pair the basket against SPY, test
the fire-season seasonal against a random-window null, and put a costed short-the-fire
timer through the wringer — with a deterministic synthetic planted-dip tape as the
positive control. **As-of 2026-06-30.**

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a fire *should* crush the exposed names, why the ignition day is a dud, why the real damage is a slow utility-liability crater in a couple of megafires, and why insurers and the calendar shrug |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the event-study anatomy, the [+1..+5] liability window, the bootstrap/jackknife outlier autopsy, the utility-vs-insurer split, the basket-vs-SPY extra drop, the seasonal random-window placebo, the costed short timer, and the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`wildfire_season/`](wildfire_season/). The fire calendar is hand-curated from
the Cal Fire incident archive and utility liability disclosures; EIX/PCG/ALL/TRV/MCY/CB
and SPY are fetched via yfinance (total-return). **PG&E's 2019 bankruptcy and dilution
are inside the `PCG` tape, not survivored out.** **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
