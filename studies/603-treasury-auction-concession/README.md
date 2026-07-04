# Study 603 — Treasury Auction Concession 🏛️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do yields cheapen into 10Y/30Y auctions and richen after? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the cheapen-into leg · Weak on the richen-after leg.* On 825 official auctions (1979→2026), yields back up **+1.55 bps** into the 10Y (HAC *t* = **+2.15**) and **+2.18 bps** into the 30Y (HAC *t* = **+3.27**) — strongest in the post-2020 supply flood (10Y *t* = 3.39). But the post-auction **richening misses the bar** (10Y *t* = −1.99, 30Y *t* = −0.52) and flipped sign after 2020. No survivorship: full official record, constant-maturity indices. |
| **Tradability** — can you harvest the post-auction bounce via TLT? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Long TLT over the 5 post-auction days: net **+3.69%/yr** at 2 bps/leg (HAC *t* = **+2.11**; carry-clean alpha *t* = +2.26) at 38.5% market time — but the edge is a **2002–2019 phenomenon** (*t* = 2.96) that is **negative since 2020** (−1.32%/yr) and drops below the bar at 5 bps/leg. Real once, decayed and regime-dependent. |
| **"Bigger auctions ⇒ bigger concession"?** | ![Busted](https://img.shields.io/badge/Bigger_auction_bigger_concession%3F-Busted-8b949e?style=flat-square) | The dealer story's own dose-response prediction fails: size-detrended big vs small auctions give Welch *t* = **−0.28** (10Y, wrong sign!) and **+0.92** (30Y, insignificant). The concession follows the auction *calendar*, not the size of the print. |

> **In one sentence:** the auction-week concession is real — yields genuinely back up 1.5–3 bps
> into 10Y/30Y auctions (HAC *t* = 2.2–3.3, biggest exactly when supply flooded) — but the
> fabled snap-back after the auction never clears the bar on 47 years of tape, the TLT round
> trip that harvested it died around 2020, and bigger auctions do **not** get bigger
> concessions — so **Mixed, Fragile, and the size story Busted**.

## What we tested

Every **10-Year Note and 30-Year Bond auction since 1979** — 825 of them, reopenings included —
from the official TreasuryDirect/FiscalData record, against daily constant-maturity yields
(^TNX/^TYX) and TLT. The concession is measured in event windows (pre = the 5 trading days
ending on the auction-day close, post = the 5 after) and, as the **primary test**, a daily
dummy regression Δy on `[1, D_pre, D_post]` with **Newey-West** *t* (overlapping refunding-week
windows autocorrelate by construction). Splits: era (pre-2008 / 2008–2019 / 2020+ supply flood)
and **size-detrended** big-vs-small auctions (Welch). Tradability holds TLT (total-return) over
the post-window with **one execution lag** (entry at the auction-day close — the calendar is
public weeks ahead), 2/5 bps one-way per leg, cash earning bills, **excess-vs-excess**. A
deterministic synthetic control (planted concession/richening V vs a random-walk null) proves
the machinery. Siblings on this bench test different plumbing:
[382-treasury-basis-trade](../382-treasury-basis-trade/) (cash-futures basis),
[383-sofr-repo-stress](../383-sofr-repo-stress/) (repo spikes),
[380-curve-roll-down](../380-curve-roll-down/) (static roll-down). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a "concession" is, the V-shape everyone draws on whiteboards, what 47 years of auctions actually show — and why the snap-back half of the legend doesn't survive |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC dummy regression, era and size splits, the carry-clean TLT alpha test, costs × round trips, and the planted-V synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`treasury_auction_concession/`](treasury_auction_concession/). The signal is the
public auction calendar; the myth-check is the size dose-response. **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
