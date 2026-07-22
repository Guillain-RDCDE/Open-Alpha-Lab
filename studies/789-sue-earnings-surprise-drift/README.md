# Study 789 — SUE Earnings-Surprise Drift 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the drift line up with the SUE sort? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Sorted on **standardized unexpected earnings (SUE)**, the top-minus-bottom forward drift is **the wrong sign** at every horizon (**−1.16% / −1.77% / −0.95%** at 1 / 2 / 3 months — the *top*-SUE names drift *less* than the bottom). The naive one-sample *t* is a spurious **−2.3**, but the autocorrelation-robust **calendar-time Newey-West *t* never clears 2** in the predicted direction (max **+0.41** at 63d). The cross-section is **non-monotone**, the sign **flips between terciles (−0.95%) and deciles (+1.53%)**, and a within-quarter block placebo sits at **p = 0.63–0.91**. The faithful synthetic control (edge 0 → 0/20 seeds fire; planted edge → *t* = +21.5) proves the detector works — there is simply no SUE drift to find on this **30-name large-cap survivor** basket. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Gross is already negative; net of one-way costs × per-event turnover (+ borrow) the 63-day long-short is **−1.27% at 5 bps, −1.47% at 10 bps**. There is no gross edge to erode and nothing to deploy. |
| **Does the textbook PEAD replicate here?** | ![Busted](https://img.shields.io/badge/Textbook_PEAD_replicates%3F-Busted-8b949e?style=flat-square) | Non-monotone quintiles, a sign that flips with the bucket count, and a robust *t* pinned near zero: the Bernard-Thomas SUE drift is **absent on liquid large-cap survivors** — exactly where the literature (Chordia et al. 2009; McLean-Pontiff 2016) says it should be weakest or gone. |

> **In one sentence:** the textbook says stocks keep drifting toward their earnings surprise for weeks — but rebuilt honestly as a SUE-sorted event study on a 1,261-event, 30-name large-cap survivor basket (EPS surprises from EDGAR, standardized by their own rolling volatility), the top-minus-bottom drift comes out **the wrong sign** (−0.9% to −1.8%), the robust calendar-time Newey-West *t* **never clears 2** (max +0.41), the sort is **non-monotone and flips sign between terciles and deciles**, and every placebo confirms noise — a clean **None × Mirage**, with the revenue sibling [534](../534-revenue-surprise-drift) coming up equally empty on the same basket.

## What we tested

We rebuild the classic **SUE post-earnings drift** (Bernard & Thomas 1989) as a clean event study
on a fixed **30-name large-cap basket**: per name we pull every quarterly **diluted EPS** figure
from **EDGAR**'s frame-tagged calendar quarters (with the 10-Q/10-K filing date), form the
**standardized unexpected earnings** (SUE = the seasonal-random-walk surprise `EPS_q − EPS_{q−4}`
scaled by the rolling volatility of the **last ~8** such surprises), sort filings into SUE
**terciles**, and measure the forward **1 / 2 / 3-month** drift of a top-minus-bottom long-short —
entering the close **one day after the filing is public** (no look-ahead). Because filings cluster
in earnings seasons, the Signal axis leads with an **autocorrelation-robust calendar-time
Newey-West *t*** (naive one-sample *t* is reported but known to overstate), plus a within-quarter
block placebo, a label-shuffle placebo, and a Wilson-bounded win-rate; Tradability charges one-way
costs × per-event turnover plus short-leg borrow. A deterministic synthetic control with a
*planted* drift confirms the engine is faithful and well-powered (so the flat result is a real
absence of signal, not a broken detector). Survivorship (the basket is names still trading in 2026)
is named on the Signal axis. **Dedup:** siblings [363-pead-drift](../363-pead-drift) (sorts on the
**price-gap CAR** around the announcement, no fundamentals),
[369-earnings-revision-momentum](../369-earnings-revision-momentum) (**analyst revisions**), and
[534-revenue-surprise-drift](../534-revenue-surprise-drift) (**revenue** SUR, not EPS) — none is the
**SUE-sorted fundamental-EPS** portfolio drift, which is this study's own axis. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a "SUE" is, why textbooks say winners keep drifting up after a beat, and why — on big liquid names — that drift simply isn't there (and even leans the other way), in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the SUE tercile long-short, forward 1/2/3-month drift, the naive-vs-robust *t* gap (calendar-time Newey-West), non-monotone quintiles, the tercile-vs-decile sign flip, label-shuffle & within-quarter block placebos, costs × turnover, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sue_drift/`](sue_drift/). Surprise = standardized unexpected earnings (SUE) from EDGAR frame-tagged quarterly diluted EPS. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
