# Study 546 — Nobel-Announcement-Drift 🏅

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do related sectors drift after the Nobel science prizes? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Across **144** mapped (prize, sector) events, 2001-2024, the mean 10-day post-announcement CAR (vs SPY) is **−0.33%** (one-sample *t* **−1.36**) — insignificant and the *wrong* sign. A random-October **placebo** beats it 98.4% of the time (*p* 0.984; random October drifts **+0.14%**). The only |*t*| ≥ 2 cut is **Medicine → health at *t* −2.33** — pharma drifted *down*. Sign flips pre/post-2013. Small event count + thematic-map assumption named on this axis. |
| **Tradability** — does buying the sector after the prize pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Long-only the mapped sector ETF from t0 to t0+10: gross **−0.33%** per event, net **−0.37%** (2 bps/side, no borrow). The trade loses before costs, at every horizon (3/5/10/21), and its sign is era-unstable. Nothing to harvest. |

> **In one sentence:** the news-attention folklore that pharma drifts after the Medicine Nobel and tech/semis after Physics/Chemistry is **not on the tape** — the average post-announcement sector CAR is a slightly *negative* −0.33% (*t* −1.36), *worse* than a random October week (placebo *p* 0.984), with the only significant cut (Medicine → health, *t* −2.33) pointing the wrong way; a Nobel prize honours decades-old basic science and carries no tradable surprise.

## What we tested

The folklore: when the Nobel Prizes are announced each October, the sectors *thematically* tied to
the science prizes catch a news-attention bid and **drift** — pharma/biotech (**XLV**, **IBB**)
after **Medicine**, tech/semiconductors (**XLK**, **SMH**) after **Physics** & **Chemistry**. We
run a textbook **event study**: hardcode the public announcement dates of the three science prizes
(2001-2024), estimate each sector ETF's market-model beta on a trailing pre-event window, and sum
its **abnormal returns vs SPY** over the post-announcement window (t0, t0+H] where t0 is the first
session on/after the announcement (the one documented execution lag). The headline is a one-sample
*t* on the mean CAR, checked against a **random-October placebo** (relocate each event to a random
trading day in the same October), a **horizon/entry-lag/era robustness sweep**, long-only costs,
and a deterministic **seed-robust synthetic positive control** that plants a drift and proves the
engine catches it. *Distinct from the desk's real announcement-drift family —
[299 Keynote-Drift](../299-keynote-drift/), [363 PEAD](../363-pead-drift/),
[515 Earnings-Announcement-Premium](../515-earnings-announcement-premium/) (announcements with a
cash-flow surprise) and the scheduled-macro [517 Pre-FOMC-Drift](../517-pre-fomc-drift/); here the
"announcement" carries no tradable surprise at all.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the folklore claims, why a Nobel prize isn't a market catalyst, and why the sectors *didn't* drift (pharma actually fell after Medicine) |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the market-model CARs with a one-sample *t*, the random-October placebo, per-prize/per-sector cuts, the horizon/entry-lag/era sweep, costs, and the seed-robust synthetic positive control |

The fingerprinted real-data run (72 announcements, 2001-2024, prices fp `ea4f6fc0a551`, CAR table
fp `e7c04bc8ac6d`) is in [docs/results.md](docs/results.md); the offline machinery proof runs on
the deterministic synthetic world in
[`nobel_announcement_drift/data.py`](nobel_announcement_drift/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`nobel_announcement_drift/`](nobel_announcement_drift/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
