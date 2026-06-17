# Study 298 -- Swiftonomics

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Did the Eras Tour move Live Nation (LYV) or the tape?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Mean abnormal return over [-1, +3] days is **-181 bps** (*negative*), cross-event t = **-1.59**, p = **0.13**, hit-rate **43.8%**, placebo p = **0.07**. Single-name (LYV only). |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A long-LYV-around-events sleeve loses money: **-82 bps/event** gross, **-122 bps** net of 10bps/side. No edge to harvest. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The loudest Eras Tour event (the Ticketmaster meltdown, -1304 bps) was a regulatory *negative* for LYV -- the opposite of the folklore. |

> **In one sentence:** Swiftonomics filled stadiums but did not move Live Nation's stock -- a market-model event study over 16 tour events shows a *negative*, insignificant abnormal return, dragged by the antitrust fallout from the ticketing crisis.

## What we tested

The folklore: "when Taylor announces a tour, buy LYV." We hardcode 16 canonical Eras Tour
events (2022-2024) in `data.py`, pull LYV + ^GSPC daily closes, and run a textbook
**market-model event study** (MacKinlay 1997): estimate LYV's beta on a clean pre-event
window, compute the cumulative abnormal return (CAR) over [-1, +3] days for each event,
and test the cross-event mean against zero with a t-test, a HAC t-stat, and a random-date
placebo distribution. We then price a costed, execution-lagged tradable sleeve. The
synthetic positive control confirms the engine fires when a real abnormal return is
planted; the real LYV tape shows the opposite sign.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the event table, abnormal returns event-by-event, the placebo, the costed trade in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | CAAR path, cross-event t-test, window robustness, by-kind split, HAC, placebo, costed sleeve, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`swiftonomics/`](swiftonomics/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
