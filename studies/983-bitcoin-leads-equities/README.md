# Study 983 — The Weekend Oracle 🔮

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does Bitcoin's weekend return predict Monday's equity return? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Over 646 closed-market windows (2014-09-22 → 2026-06-29), a 1% Bitcoin move across the weekend was followed by a **+4.019%** move in SPY on the next session (*t* = **+3.01**, R² 3.55%). The sign agreed 58.8% of the time against a 50% coin flip (*t* = +4.49), and the average next-session return was +0.242% after an up weekend against -0.187% after a down one (*t* = +4.64). Split at March 2020 the slope was +0.001 before and +0.086 after — the era in which Bitcoin became a high-beta Nasdaq position is doing most of the work. This is the **clean** design: zero hours of overlap, unlike the ordinary daily lead-lag correlation of -0.01, which shares 3 hours of clock with its own target. One thing this stamp must not be read as: Bitcoin's weekend move has a standard deviation of 5.3%, most of it Bitcoin's own business, so even a weekend whose news determined Monday perfectly could only produce a *t* of about **2.8** over 646 windows — a null here is weak evidence of absence, not strong evidence of nothing. |
| **Tradability** — is there a Monday trade in it after costs? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Buying SPY only on the session after an up crypto weekend — 363 of 646 windows, 56% of them — returned **+6.28%/yr** on those same sessions against **+2.64%** for holding every one of them regardless (+3.64%/yr, *t* = +1.72), after 2 bps a side. The rule is in the market about one session a week, so this is a comparison of like with like, not a Sharpe you can put next to a buy-and-hold Sharpe. |

> **In one sentence:** Bitcoin's weekend move predicts the next equity session with *t* = +3.01 and an R² of 3.55% — the cleanest lead-lag design available between the two assets, and it explains 3.5% of Monday.

## What we tested

"Crypto leads risk assets" is one of the most repeated claims on financial
Twitter, and almost every version of it is measured wrongly. Yahoo's Bitcoin daily bar closes at
**00:00 UTC** — three hours *after* the New York equity close — so a "same-day" crypto bar
already contains news the equity close could not have seen, and the ordinary daily lead-lag
correlation is contaminated by the clock rather than driven by information.

There is one alignment that escapes this entirely. The stock market is shut for about
**sixty-five hours** every weekend; Bitcoin is not. Whatever happens in that window is priced by
one asset and cannot be priced by the other, so *weekend crypto move → next equity session* is a
lead-lag design with **exactly zero overlap**. This study builds that panel over every gap in
the equity calendar (holiday closures included, as longer versions of the same experiment),
regresses with HC1 errors, splits at March 2020 — when Bitcoin stopped being a diversifier —
buckets by closure length, prices the resulting Monday rule against holding the very same
sessions unconditionally, and runs **gold** through the identical machine as a falsification
control, since GLD is shut over the weekend too and must therefore show nothing.
**Dedup:** distinct from **142-bitcoin-correlation** and **604-crypto-equity-beta**
(contemporaneous co-movement), **067-monday-effect** and **552-weekend-effect** (calendar
seasonality in equities alone, with no external predictor), **481-overnight-vs-intraday** (the
close-to-open decomposition of a single asset) and **330-lead-lag-etfs** (lead-lag between two
assets that share a trading calendar — the case this study cannot use).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the ordinary crypto-leads-stocks chart is a clock artefact, and the one weekend-shaped experiment that isn't |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | overlap accounting per design, the closed-window panel with HC1 errors, regime and closure-length splits, a gold falsification control, the Monday rule against a like-for-like benchmark, and a planted-news simulation |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`weekendoracle/`](weekendoracle/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
