# Study 01 — The Overnight Anomaly

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the shared engine
> (`quantlab/`), start at the [root](../../README.md).*

**The idea.** Across world markets, almost all the long-run gain has accrued
*overnight* (close → next open); the trading day is flat. Bruce Knuteson reads
this as the fingerprint of large-scale market manipulation. This study takes the
fact seriously, then audits the inference end-to-end.

### Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** | `REAL` | SPY overnight mean carries a Newey-West *t* ≈ 5; confirmed across ~441 S&P 500 stocks (≈69% have overnight > intraday Sharpe). |
| **Tradability** | `MIRAGE` | Magnitude inflated by a calendar-time illusion; what's left is mostly gap-risk **beta** below trading costs; it **decays** (5y Sharpe ~2→~0.5) and **doesn't scale** (capacity ~\$10M). |
| **Manipulation?** | `NOT SUPPORTED` | Bayesian posterior ≈ 2–3%; the discriminating evidence (foreign-ETF & China inversions, negative P&L at scale) favours microstructure. |

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

---

## 📄 Read the article first

This study is a *response*, so start with the source. One command fetches every
openly available paper from its official source (we don't redistribute PDFs —
[here's why](papers/README.md)):

```bash
python studies/01-overnight-anomaly/papers/download_papers.py
```

- **Celebrating Three Decades of Worldwide Stock Market Manipulation** (2019) — [arXiv:1912.01708](https://arxiv.org/abs/1912.01708)
- **Strikingly Suspicious Overnight and Intraday Returns** (2020) — [arXiv:2010.01727](https://arxiv.org/abs/2010.01727)
- **They Still Haven't Told You** (2022) — [arXiv:2201.00223](https://arxiv.org/abs/2201.00223)
- **Nothing to See Here: How to Say It When You Need to** (2023) — [SSRN 4619084](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619084) *(login wall)*

See [`papers/README.md`](papers/README.md) for the full reading list and
[`docs/references.md`](docs/references.md) for a map of which explanation each
paper argues.

## 📓 Two notebooks, two audiences

Both render inline on GitHub (pre-executed — no need to run anything):

| | For whom | Inside |
|---|---|---|
| **[`notebooks/01_for_the_curious.ipynb`](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes, in plain language: why it matters, why markets move at night, and the three traps |
| **[`notebooks/02_for_the_quants.ipynb`](notebooks/02_for_the_quants.ipynb)** | quants | the full teardown: HAC/Lo inference, the clock illusion, firm-level breadth, alpha-vs-beta, decay, capacity, the steelman manipulator, a Reality Check and a Bayesian posterior |

## 📑 The deliverables

- **Working paper** — [`paper/overnight_alpha.pdf`](paper/overnight_alpha.pdf):
  *"Overnight, or Overhyped?"* (LaTeX source + reproducible figure script in
  [`paper/`](paper/)).
- **Point-by-point rebuttal** — [`RESPONSE.md`](RESPONSE.md): each of Knuteson's
  claims mapped to a quantified counter-measure.

---

## Findings (all reproduced on real data)

- **It's real.** SPY overnight mean: Newey-West (HAC) *t* ≈ 5; intraday *t* ≈ 1.
- **Firm-level breadth.** Across ~441 S&P 500 members, ~69% have a higher
  overnight than intraday Sharpe; equal-weight overnight +372% vs +233% intraday.
- **The clock illusion.** The overnight window averages ~28 calendar hours vs the
  6.5-hour day. Per *session* the night beats the day ~4×; **per calendar hour,
  ~1.3×.** Most of the gap is a unit error.
- **Mostly beta, and fading.** ~40% of the overnight return is gap-risk beta; the
  residual alpha (~1.9 bps) is below break-even cost (~3.3 bps); the 5-year
  overnight Sharpe fell from ~2 (1998) to ~0.5 (2026).
- **Unscalable.** Square-root market-impact capacity is single-digit \$millions.
- **The steelman fails.** Granting a "manipulator" the entire overnight drift, it
  still loses ~\$0.75B/yr at \$1B and ~\$25B/yr at \$10B — self-defeating at the
  scale the accusation requires.
- **Not manipulation.** Foreign-ETF and Chinese T+1 inversions are predicted by
  microstructure; the Bayesian posterior on manipulation is ≈ 2–3%.

## A few quant gotchas this study is careful about

- **Adjustment mode is a decision, not a detail** — it moves return between night
  and day (ex-dividend happens at the open). Default `split_only`; document yours.
- **Sharpe > raw return**, and **per-hour > per-session** — normalise before you
  marvel.
- **The factor 2** — you cross the spread to buy *and* sell, ~252×/year.
- **Execution ≠ academic prints** — the anomaly is measured on close/open
  auctions retail can't touch.

## References

Author–date (Chicago / *JFE*). Full list + literature map:
[`docs/references.md`](docs/references.md); BibTeX in [`references.bib`](references.bib).
The engine that produced every number lives at [`../../quantlab/`](../../quantlab/).
