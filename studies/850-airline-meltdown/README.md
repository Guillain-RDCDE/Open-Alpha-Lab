# Study 850 — Airline Operational Meltdown ✈️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a public operational meltdown dent the implicated carrier's stock? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The 9-event aggregate is directionally right and nominally significant at one month (mean CAR **−844 bps**, one-sample *t* = **−2.24**, random-date placebo *p* = **0.014**) — but **not robust**: it hinges entirely on the two Boeing 737-MAX groundings (a *fundamental* product-line shock). **Airlines-only** it's insignificant everywhere (day-0 *t* = −0.79, month *t* = −1.30); several individual meltdowns even had *positive* one-month CARs. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Shorting the meltdown stock is positive net of costs+borrow at every horizon in-sample (best *t* = **+1.09**, n=9) but never significant and again entirely Boeing-concentrated — airlines-only it's ≈0. A nine-event edge on two events, needing a hard-to-borrow short into a crowded trade. |
| **Reputational shock vs quick fade?** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | A grounded *product line* (Boeing MAX) genuinely dents the stock — it's fundamental. A cancellation collapse / IT outage / PR firestorm at a still-flying carrier mostly **fades**. The reputational-shock story is not detectable at n=9. |

> **In one sentence:** across 10 of the most infamous airline/Boeing operational meltdowns since 2016 (9 with price coverage — Spirit's is gone, SAVE delisted), the implicated stock's one-month abnormal return averages **−844 bps** (*t* = −2.24, placebo *p* = 0.014) — but strip the two Boeing MAX groundings, which are fundamental earnings shocks rather than the claimed *reputational* one, and the pure operational-meltdown effect collapses to an insignificant −393 bps (*t* = −1.30): a Weak, composition-driven tug, not a real reputational edge.

## What we tested

The folklore says a very public operational meltdown — a multi-day grounding, a
mass-cancellation collapse, a viral PR disaster — inflicts *reputational* damage that
dents the airline's own stock and lingers for a month. We steelman it with a single-name
**market-model event study** (MacKinlay 1997: OLS α/β on SPY over a pre-event estimation
window, abnormal-return CAR at day 0, the event week, the event-month, and the pure
one-month drift), a **same-ticker random-date permutation placebo** (5,000 draws), a
robustness decomposition (airlines-only vs. Boeing-only, leave-one-out, sub-eras), and a
costed **short-the-meltdown** timer — on a hand-curated table of 10 meltdowns (Delta 2016
& 2024, United 2017 Dao, Boeing MAX 2019 & 2024, Southwest 2021 & 2022, American 2021,
Spirit 2021). A deterministic synthetic tape with a *planted* drop is the positive
control. **As-of 2026-06-30.** **Dedup:** distinct from
[707-plane-crash-effect](../707-plane-crash-effect/) (fatal *crashes* → broad-market mood,
not self-inflicted operational failures → the single implicated stock),
[554-airline-bookings](../554-airline-bookings/) (an alt-data demand signal, not a discrete
event) and [313-geopolitical-shock](../313-geopolitical-shock/) (wars/attacks → market-wide).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a meltdown *should* dent the stock, what the tape shows, and why the one number that "works" is really about Boeing, not reputation |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the market-model anatomy, the random-date placebo, the airlines-vs-Boeing decomposition and leave-one-out, the costed short timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`airline_meltdown/`](airline_meltdown/). The meltdown calendar is hand-curated
from public 8-K/DOT/FAA/NTSB filings and contemporary wire coverage; SPY and
LUV/DAL/UAL/AAL/BA are fetched via yfinance (Spirit/SAVE delisted → dropped, named
honestly). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
