# Study 400 — Patent-Intensity 💡

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there an innovation premium? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The high-R&D-intensity tertile beats the low-intensity tertile by **+3.60%/yr**, and the split loads on a *real* axis (97.9th pct of random splits). But on a clean 21-year tape it **fails t ≥ 2** (HAC **t = 1.50**), clears 2 *only* at the barely-sorted "halves" split, and decays at every concentrated split. Positive, published, but specification-fragile and insignificant ⇒ **Weak**, not Real. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Turnover cost is trivial (annual rebalance), but a long/short **pays short-borrow**: a 100 bps/yr borrow on the short leg drags the spread to **+2.60%/yr at t = 1.08** — a net edge indistinguishable from zero. The long-only leg beats SPY by only +2.25%/yr (t = 1.21), most of it a growth tilt you can buy cheaper as a style ETF. |
| **An innovation premium?** | ![Misattributed](https://img.shields.io/badge/Innovation_premium%3F-Misattributed-8b949e?style=flat-square) | The spread is **not noise** — it is a persistent *style/sector* axis (R&D-heavy = growth/tech/pharma; R&D-light = value/banks/staples). That makes the "innovation premium" a **relabelled growth-vs-value tilt**, not a distinct patent-specific alpha — and even the tilt doesn't clear significance. |

> **In one sentence:** sort large-caps by audited R&D intensity and the high-intensity tertile *does* beat the low-intensity tertile by +3.60%/yr — but at HAC **t = 1.50** that "innovation premium" never clears the significance bar (it reaches t = 2 only when you barely sort, and short-borrow drags the net spread to t = 1.08), and the spread is really the growth-vs-value style axis in a patent costume, not a distinct innovation alpha.

## What we tested

True issued-patent counts aren't on a free feed, so we use **reported R&D intensity** (SEC EDGAR `ResearchAndDevelopmentExpense` / revenue) as the audited proxy for "how patent-and-invention-intensive is this firm" — exactly as the academic literature operationalises it (Chan-Lakonishok-Sougiannis 2001; Hirshleifer-Hsu-Li 2013). From a fixed **40-name** large-cap field chosen *by sector* (not by returns) we rank each year by intensity, go **long the top tertile / short the bottom**, rebalance annually with a **1-year reporting lag**, and over **21.3 years** (2005–2026, 256 months) race the long-short against SPY and against a *blind* random long/short of the same field. We charge one-way costs **and short-borrow**, run a Newey-West HAC test, and confirm with a deterministic synthetic control whose `edge` knob plants — or doesn't — a true innovation premium. Survivorship is named on the Signal axis (the basket is current-membership, but the bias is largely common to both legs of the long/short).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy the inventors, short the dinosaurs" is really "buy growth, short value," why +3.6%/yr isn't an edge if it could be luck, and why short-borrow finishes it off — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | intensity ranking with a reporting lag, the long-short and long-minus-SPY HAC t-stats, the random-blind-split sector control, a fraction-robustness sweep, costs + borrow, and a synthetic planted-premium / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`patent_intensity/`](patent_intensity/). Intensity here is an explicit **proxy** (reported R&D / revenue), not issued patents. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
