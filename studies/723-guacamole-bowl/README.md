# Study 723 — Guacamole-Bowl 🥑

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the Super-Bowl binge print a Jan–Feb seasonal? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The guac window (Jan–Feb) **under**-performs the rest of the year by **−1.19%/month, t = −1.67** (the *wrong* sign). No thesis month clears \|t\| ≥ 2 (Jan t_HAC = −0.36, Feb = +0.12); a placebo across all 66 month-pairs ranks Jan–Feb **9/66** — bottom decile. |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A long-Jan–Feb timer earns **Sharpe −0.09** (−0.11 net) vs buy-and-hold SPY's **+0.61** — it holds the market only in its worst window. No proxy alpha (NW t = 1.27), and the pure-play avocado name (`CVGW`) isn't even reliably on the tape. |
| **Guacamole surge?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The avocado price is **5.9 index points below** its annual mean in the window (winter = peak Mexican supply), and the snack tape's window is the year's weakest. A date-certain, pre-supplied spike leaves no price for a trader. |

> **In one sentence:** the most-cited food-calendar trade — buy the avocado/produce complex ahead of the Super-Bowl guacamole binge — has the guac window landing as the year's *weakest* stretch (−1.19%/mo, t = −1.67, placebo rank 9/66), a timer at negative Sharpe against SPY's +0.61, and an avocado price that's actually *soft* in winter: a real demand event with no tradable price attached.

## What we tested

The folklore: America eats ~**100 million pounds** of avocados for the Super Bowl (the biggest
guacamole day of the year, per the [Hass Avocado Board](https://hassavocadoboard.com/)), so the
avocado/produce trade should carry a **January–February seasonal** you can position ahead of. We test
the strongest tradable version on **`PEP`** (PepsiCo/Frito-Lay — the Super-Bowl chip-and-dip complex of
Tostitos + dips; a **labelled proxy**, since the pure-play avocado name Calavo `CVGW` is currently
untradable on the Yahoo feed) back to 1993 vs **`SPY`**: per-month HAC *t*-stats, a Jan–Feb window
spread, a **placebo across all 66 month-pairs**, a block-bootstrap CI, and a long-window timer net of
costs. A cited, approximate wholesale-Hass seasonal index checks the *price* premise. (Same
labelled-proxy shape as [Study 358](../../358-watch-index/); same calendar-folklore family as
[Study 307](../../307-coffee-seasonality/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a real, date-certain demand spike makes a *terrible* calendar trade — the avocado price's winter softness, the placebo, the timer that loses to doing nothing |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-month HAC *t*, the Jan–Feb Welch spread + 66-pair placebo (z-score), block-bootstrap CI, timer race + Newey-West alpha, a planted-seasonal positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run
(PEP/SPY + ^IRX, 1993–2026, fp `14fdb930823d`): [docs/results.md](docs/results.md).

---

*Engine: [`guacamole_bowl/`](guacamole_bowl/). The tradable leg is a **labelled proxy** (PEP = the Super-Bowl snack complex, not an avocado; CVGW unavailable on the feed) and the avocado seasonal is a **hardcoded, cited, approximate proxy** — shape only, not a live feed. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
