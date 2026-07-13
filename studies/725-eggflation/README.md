# Study 725 — Eggflation 🥚

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the egg price predict CALM? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Last month's egg-price change forecasts CALM's next-month return with HAC *t* = **+0.93** (placebo *p* = **0.48**); +market control *t* = 1.00. The only lag that nominally clears 2 (*t*₊₂ = 2.02) **fails its own placebo** (*p* = 0.11). The eye-catching **+0.76** correlation is two co-trending *levels*. |
| **Tradability** — can you harvest it net of costs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | An egg-momentum timer nets Sharpe **0.63** — it beats buy-hold CALM (0.41) by *parking in cash*, not by signal, and still **loses to plain SPY (0.80)**. Single-name, three-event, one-commodity concentration for sub-index returns. |
| **Already priced in?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The *only* significant lead-lag runs backwards: CALM's return **leads** the next retail egg print (*t* = **2.03**, lag-2 = 2.27). By the time eggflation is a headline, it's already in the chart. |

> **In one sentence:** "bird flu spikes eggs, so buy Cal-Maine" is a `NONE` × `MIRAGE` — the public retail egg print doesn't forecast CALM (next-month *t* = 0.93, placebo *p* = 0.48), the mined lag-2 *t* = 2.02 dies under its placebo, the egg-timer that "works" just sidesteps CALM's slumps and still trails SPY, and the *only* real relationship is the stock **leading** the government egg number — a receipt, not a signal.

## What we tested

The retail-trader folklore: egg prices go vertical on every avian-flu cull (2015, 2022–23,
2024–25 — up to a record **$6.23/dozen**), and **Cal-Maine Foods (`CALM`)**, the largest US
shell-egg producer, is a near-pure bet on the egg price, so you can *front-run* the spike.
We test the strongest tradable version against a **cited, approximate USDA/BLS retail
egg-price series** (a *labelled proxy* — BLS `APU0000708111`, not a live feed) and real
month-end `CALM`/`SPY` tape (yfinance): a **predictive HAC regression** (does last month's
egg change forecast CALM's next-month return, with the publication lag honoured?), a
**circular-shift placebo**, a **reverse lead-lag** test (does the *stock* lead the print?),
and an **egg-momentum timer** net of costs vs buy-hold and SPY — plus a seed-robust
synthetic planted-lead control. *(Same labelled-proxy shape as
[358-watch-index](../358-watch-index/); same "trade the headline" family as
[550-box-office-momentum](../550-box-office-momentum/).)*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the egg/CALM story *feels* obvious, the level-correlation trap, and the punchline that the stock moves first — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | contemporaneous vs predictive HAC regressions, the circular-shift placebo (incl. the mined lag-2), the reverse lead-lag, the timer net of costs, and a synthetic planted-lead positive control |

The fingerprinted real-data run (egg proxy + CALM + SPY, 2015–2026, fp `0c92a7548d09`,
as-of 2026-06-30) is in [docs/results.md](docs/results.md); reproduce via
[examples/verify.py](examples/verify.py) (`--fetch` to download). The offline machinery
proof runs on the synthetic world in [`eggflation/data.py`](eggflation/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`eggflation/`](eggflation/) (with [`quantlab/`](../../quantlab/) for the repro stamp). Egg price is a **hardcoded, cited, approximate proxy** — not a live feed. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
