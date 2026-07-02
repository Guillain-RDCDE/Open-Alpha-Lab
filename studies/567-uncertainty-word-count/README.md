# Study 567 — Uncertainty-Word-Count 🤔

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**When executives pile on "uncertain", "maybe", "could" in filings and calls, does volatility spike next?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do hedging words predict future vol/returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The literature (Loughran-McDonald 2011; Campbell et al. 2014) supports it, but **there is no free feed** of scored filings joined to forward vol — so no real-tape *t* to earn `REAL` (capped at `WEAK`). On the synthetic engine the *naive* uncertainty→forward-vol slope is a blockbuster (*t* **13.4**) — but it's mostly a **confound**: hedgy firms are *already* jumpy. Control for trailing vol and it collapses to *t* **3.3**. |
| **Tradability** — could you trade the hedging density? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | In the planted world the long-plain/short-hedgy book earns **+1.79%/event** (*t* 2.46), **+5.85%/yr** net of costs + hedgy-leg borrow — but the per-window slope-*t* wanders (−0.46 → +2.89 → +1.20 → +2.73), a once-a-quarter signal has little data per period, and none of it is verified on a real tape. |

> **In one sentence:** uncertainty-word density really does track future volatility in the literature — but the naive version is *mostly* a vol-persistence confound (uncertain firms are already jumpy: *t* 13.4 → 3.3 once you control for trailing vol), and with no free feed of scored filings the desk can only demonstrate the machinery, not certify the edge — `WEAK × FRAGILE`.

## What we tested

The claim (Loughran & McDonald 2011, *When Is a Liability Not a Liability?*; Campbell, Chen, Dhaliwal,
Lu & Steele 2014; Kravet & Muslu 2013): the **density of uncertainty / hedging words** in a firm's
disclosure — its token share in the LM **Uncertainty** word list ("uncertain", "maybe", "could",
"risk", "approximately", "depend") — predicts **higher future realised volatility** and (the softer
half) **lower future returns**. This study is **synthetic-only by design** — there is no free,
no-key retail feed of parsed EDGAR filings scored against a licensed LM uncertainty lexicon and
joined to *forward* vol (that lives in paid NLP vendors), so a `REAL` stamp (robust *t* ≥ 2 on a
**real** tape) is out of reach and the Signal is capped at `WEAK`. On a deterministic, seeded
firm-event panel (single knob `uncert_beta` plants the text→vol link) we run the honest test: the
**naive** vs **trailing-vol-controlled** forward-vol regression (the central confound check —
uncertain firms are already jumpy and vol is autocorrelated), a **label-shuffle placebo** null, the
return-drag regression and a **long-plain/short-hedgy** book with costs + borrow, a **four-window**
robustness sweep, and a **seed-robust synthetic positive control** (25 seeds) proving the engine
banks a planted effect and reads flat at the null. *Distinct from [566 earnings-call-tone](../566-earnings-call-tone/)
(net **sentiment** → **drift**); this is the **uncertainty** dictionary → **volatility**.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the uncertainty dictionary is, why hedgy words *look* like they predict vol, and the trick — those firms were already jumpy — in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the naive vs trailing-vol-controlled slope (13.4 → 3.3), the placebo null, the return-drag regression, the long-short with costs & borrow, the four-window wander, and the seed-robust synthetic control |

The fingerprinted synthetic headline run (480 firm-events, panel fp `fa07e8ea11c3`, as-of 2026-06-30)
is in [docs/results.md](docs/results.md); the offline machinery lives in
[`uncertainty_word_count/`](uncertainty_word_count/). **No real tape** — the data-availability
limitation is named on the Signal axis.

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`uncertainty_word_count/`](uncertainty_word_count/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
