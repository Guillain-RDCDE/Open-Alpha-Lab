# Study 01 — The Overnight Anomaly

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk, the shared engine
> (`quantlab/`) and the house style, start at the [root](../../README.md) and the
> [methodology](../../METHODOLOGY.md). This page follows the desk's standard seven
> beats.*

## Verdict — read this first

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | `REAL` | SPY overnight mean carries a Newey-West *t* ≈ 5; confirmed across ~441 S&P 500 stocks (≈69% have overnight > intraday Sharpe). |
| **Tradability** — does it survive costs, capacity, scale? | `MIRAGE` | Magnitude inflated by a calendar-time illusion; what's left is mostly gap-risk **beta** below trading costs; it **decays** (5y Sharpe ~2→~0.5) and **doesn't scale** (capacity ~\$10M). |
| **Manipulation?** — does the pattern prove fraud? | `NOT SUPPORTED` | Bayesian posterior ≈ 2–3%; the discriminating evidence (foreign-ETF & China inversions, negative P&L at scale) favours microstructure. |

> **In one sentence:** the overnight effect is *real and broad*, but its size is a
> clock illusion, what remains is mostly beta below trading costs, it doesn't
> scale, and the "market-manipulation" reading isn't supported.

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

---

## 1 · The Claim

Across world markets, almost all the long-run gain has accrued **overnight**
(close → next open); the trading day is flat. **Bruce Knuteson** reads this as the
fingerprint of large-scale market manipulation — one enormous book quietly nudging
prices up in illiquid pre-open windows, worldwide, for decades.

This study is a *response*, so it starts with the source. One command fetches every
openly available paper from its official home (we don't redistribute PDFs —
[here's why](papers/README.md)):

```bash
python studies/01-overnight-anomaly/papers/download_papers.py
```

- **Celebrating Three Decades of Worldwide Stock Market Manipulation** (2019) — [arXiv:1912.01708](https://arxiv.org/abs/1912.01708)
- **Strikingly Suspicious Overnight and Intraday Returns** (2020) — [arXiv:2010.01727](https://arxiv.org/abs/2010.01727)
- **They Still Haven't Told You** (2022) — [arXiv:2201.00223](https://arxiv.org/abs/2201.00223)
- **Nothing to See Here: How to Say It When You Need to** (2023) — [SSRN 4619084](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619084) *(login wall)*

## 2 · So What?

Three very different things ride on the answer: a **fraud accusation** against the
financial system (true or careless?), **your savings** (is "buy the close, sell the
open" a free lunch? — two real ETFs, NSPY/NIWM, tried it and were liquidated in
2023), and **how markets actually work** (the honest cause teaches more than any
get-rich scheme). Same data, three very different worlds.

## 3 · How We'd Know

We take the fact seriously, then audit the inference end-to-end on the desk's
6-step protocol — decompose, robust inference (HAC/Lo), critique the magnitude,
alpha vs beta, execution & capacity, verdict. The numbers come from the shared
[`../../quantlab/`](../../quantlab/) engine; the full teardown lives in two
notebooks, the *same story at two altitudes*:

| | For whom | Inside |
|---|---|---|
| **[`notebooks/01_for_the_curious.ipynb`](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes in plain language: why markets move at night, and the tricks that inflate the headline |
| **[`notebooks/02_for_the_quants.ipynb`](notebooks/02_for_the_quants.ipynb)** | quants | the full teardown: HAC/Lo inference, the clock illusion, firm-level breadth, alpha-vs-beta, decay, capacity, the steelman manipulator, a Reality Check and a Bayesian posterior |

Both render inline on GitHub (pre-executed — nothing to run).

## 4 · The Teardown

All reproduced on real data:

- **It's real.** SPY overnight mean: Newey-West (HAC) *t* ≈ 5; intraday *t* ≈ 1.
- **Firm-level breadth.** Across ~441 S&P 500 members, ~69% have a higher overnight
  than intraday Sharpe; equal-weight overnight +372% vs +233% intraday.
- **The clock illusion.** The overnight window averages ~28 calendar hours vs the
  6.5-hour day. Per *session* the night beats the day ~4×; **per calendar hour,
  ~1.3×.** Most of the gap is a unit error.
- **Mostly beta, and fading.** ~40% of the overnight return is gap-risk beta; the
  residual alpha (~1.9 bps) is below break-even cost (~3.3 bps); the 5-year
  overnight Sharpe fell from ~2 (1998) to ~0.5 (2026).
- **The steelman fails.** Granting a "manipulator" the entire overnight drift, it
  still loses ~\$0.75B/yr at \$1B and ~\$25B/yr at \$10B — self-defeating at the
  scale the accusation requires.
- **Not manipulation.** Foreign-ETF and Chinese T+1 inversions are predicted by
  microstructure; the Bayesian posterior on manipulation is ≈ 2–3%.

## 5 · The Verdict

Real signal, oversold magnitude, no tradable residual, unsupported fraud — the
three stamps in the box up top, each now earned. A point-by-point rebuttal of
Knuteson's specific claims is in **[`RESPONSE.md`](RESPONSE.md)** (every claim
mapped to a quantified counter-measure), and the working paper *"Overnight, or
Overhyped?"* compiles the argument: **[`paper/overnight_alpha.pdf`](paper/overnight_alpha.pdf)**
(LaTeX source + reproducible figure script in [`paper/`](paper/)).

A few gotchas this study is careful about — the kind of thing that quietly makes or
breaks a result:

- **Adjustment mode is a decision, not a detail** — it moves return between night
  and day (ex-dividend happens at the open). Default `split_only`; document yours.
- **Sharpe > raw return**, and **per-hour > per-session** — normalise before you
  marvel.
- **The factor 2** — you cross the spread to buy *and* sell, ~252×/year.
- **Execution ≠ academic prints** — the anomaly is measured on close/open auctions
  retail can't touch.

## 6 · Could You Trade It?

No — and that's the decisive part. Charge realistic costs against the *alpha* (not
the gross) and there's nothing left: the residual ~1.9 bps sits below the ~3.3-bps
break-even. Square-root market-impact capacity is **single-digit \$millions** before
the strategy's own trading erases the edge; by ~\$10M it's break-even and beyond
~\$100M deeply negative. The same capacity maths is what sinks the manipulation
story: a book big enough to *move world markets* would pay far more in impact than
a 3-bps-a-night edge could ever return. The "night ETFs" that tried this in size
are the real-world footnote — liquidated within ~14 months.

## 7 · Going Further

**What would change the verdict:** toward `INVESTABLE`, a residual edge that
survives costs *and* scales (smarter auction-only execution, or a sub-population
with genuinely positive per-hour, post-beta, post-decay alpha); toward
`manipulation`, participant-level open-auction order data, characteristic intraday
reversals, or a firm-specific signature. We found neither — fork it and try.

Next in the desk's queue: momentum's overnight/intraday split, the weekend effect,
post-earnings drift. New studies follow the same seven beats
([METHODOLOGY.md](../../METHODOLOGY.md)).

## References

Author–date (Chicago / *JFE*). Full list + literature map:
[`docs/references.md`](docs/references.md); BibTeX in [`references.bib`](references.bib).
The engine that produced every number lives at [`../../quantlab/`](../../quantlab/).
