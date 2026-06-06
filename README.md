# Overnight Alpha — verify the night-trade edge, then stress-test it honestly

[![tests](https://github.com/Guillain-RDCDE/Overnight-Alpha/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Overnight-Alpha/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **An independent, reproducible response to Bruce Knuteson's claim that the
> overnight stock-market return pattern is the fingerprint of large-scale market
> manipulation.**
>
> Almost all the long-run gain in the world's stock indices accrues **overnight**
> (yesterday's close → today's open); the **intraday** session (open → close) is
> flat to negative. That fact is real. Knuteson reads it as orchestrated fraud.
> This repository gives you the **code to check every figure yourself**, an
> honest **two-track write-up** (one for the curious, one for quants), and a
> measured **verdict** — separating what the data supports from what it doesn't.
>
> **Not investment advice.** Research & education only. See [LICENSE](LICENSE).

---

## 📄 Read the article first

This repo is a *response*, so start with the source. One command fetches every
openly available paper straight from its official source (we don't redistribute
the PDFs — [here's why](papers/README.md)):

```bash
python papers/download_papers.py
```

Knuteson's articles:

- **Celebrating Three Decades of Worldwide Stock Market Manipulation** (2019) — [arXiv:1912.01708](https://arxiv.org/abs/1912.01708) · the figures we reproduce
- **Strikingly Suspicious Overnight and Intraday Returns** (2020) — [arXiv:2010.01727](https://arxiv.org/abs/2010.01727) · the data paper
- **They Still Haven't Told You** (2022) — [arXiv:2201.00223](https://arxiv.org/abs/2201.00223)
- **Nothing to See Here: How to Say It When You Need to** (2023) — [SSRN 4619084](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619084) *(login wall)*

See [`papers/README.md`](papers/README.md) for the full reading list and the
supporting literature; [`docs/references.md`](docs/references.md) maps which
explanation each paper argues.

**→ Our structured reply, claim by claim, is in [`RESPONSE.md`](RESPONSE.md)** —
each of Knuteson's assertions mapped to a quantified counter-measure.

---

## TL;DR — our response in three numbers (offline, one command)

```bash
pip install -r requirements.txt
python examples/run_synthetic_demo.py
```

1. **(A) Compounding inflates the headlines.** A *completely innocent* 1–3 bps
   per-night drift, compounded over 30 years on a log axis, becomes "hundreds of
   percent" — even "billions of percent". The explosion is the **exponent**, not
   fraud.
2. **(B) Data artefacts manufacture the signal.** A handful of mis-adjusted
   closes (splits/dividends) mechanically move return *out of* the day leg and
   *into* the night leg — the engine behind the wildest emerging-market figures.
   A built-in detector flags them.
3. **(C) Costs kill the strategy.** Buying every close and selling every open is
   ~252 round-trips/year. The honest backtest shows a positive *gross* Sharpe
   going **negative net** at a realistic spread — exactly what liquidated the
   NSPY / NIWM "night effect" ETFs in 2023.

---

## 📓 Two notebooks, two audiences

The repo is built to be read two ways. Both render inline on GitHub (figures and
outputs are pre-executed — you don't have to run anything to read them):

| | For whom | What's inside |
|---|---|---|
| **[`notebooks/01_for_the_curious.ipynb`](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: the night/day pattern, then the three traps (compounding, dirty data, fees) that explain why it's subtler than it looks |
| **[`notebooks/02_for_the_quants.ipynb`](notebooks/02_for_the_quants.ipynb)** | quants / practitioners | real-data teardown: HAC/Lo significance, the **clock illusion** (per-hour normalization), foreign-ETF & China inversions, alpha-vs-beta, **post-publication alpha decay**, **market-impact capacity**, and a likelihood-ratio argument on the manipulation claim |

> Notebooks are *generated* from [`notebooks/build_notebooks.py`](notebooks/build_notebooks.py)
> then executed with `nbconvert` — so the figures you read are reproducible outputs, not screenshots.

### Findings worth flagging (all reproduced on real data)

We take the fact seriously, then dismantle the framing:

- **It's real.** SPY's overnight mean carries a Newey-West (HAC) **t ≈ 5**; the
  intraday mean is insignificant (t ≈ 1). We concede the empirical fact outright.
- **The clock illusion.** The overnight window averages **~28 calendar hours**
  (weekends/holidays) vs the 6.5-hour day. Per *session* the night beats the day
  ~4×; **per calendar hour, only ~1.3×**. Most of the famous gap is a unit error.
- **Mostly beta, and fading.** ~40% of SPY's overnight return is gap-risk beta;
  the residual alpha (~1.9 bps) is **below** the break-even cost (~3.3 bps); and
  the trailing 5-year overnight Sharpe has **decayed from ~2 (1998) to ~0.5
  (2026)** — textbook post-publication decay.
- **Unscalable.** Square-root market-impact capacity is **single-digit millions**
  of dollars — too thin for retail *and* for any firm large enough to "manipulate
  world markets".
- **The cross-section betrays the story.** UK/Germany/France/Japan ETFs are
  **inverted** (they trade in New York while their home market trades overnight
  US-time → the "overnight" window contains the home session), and China (FXI) is
  **not statistically distinguishable from zero** — both predicted by
  microstructure, awkward for a single global manipulator.

---

## For beginners — what's the idea?

Imagine a stock that, on average, drifts up a tiny bit while the market is
*closed* (overnight) and drifts flat while it's *open* (intraday). Over decades
that tiny overnight drift compounds into a huge-looking number. The strategy
"buy at the close, sell at the next open" tries to harvest it. The catch: you'd
pay the buy/sell spread **twice every day**, ~250 days a year. This repo
measures the drift exactly, and then subtracts the real costs so you can see
what's actually left (usually: not much).

## For practitioners — what's inside

| Module | Role |
|---|---|
| [`overnight/decompose.py`](overnight/decompose.py) | Exact night/day/close-close split. Identity `(1+r_on)(1+r_id)=(1+r_cc)`, `summary()` with **Sharpe**. |
| [`overnight/data.py`](overnight/data.py) | Yahoo fetch + parquet cache. Adjustment modes `split_only` / `total_return` / `raw`. |
| [`overnight/diagnostics.py`](overnight/diagnostics.py) | Critique layer (offline): compounding table, split-artefact injector + detector, weighting/selection effects, synthetic market. |
| [`overnight/backtest.py`](overnight/backtest.py) | Cost-aware backtest, break-even cost, cost sweep. |
| [`overnight/stats.py`](overnight/stats.py) | Bootstrap Sharpe confidence intervals, alpha-vs-beta (gap-risk) decomposition. |
| [`overnight/analytics.py`](overnight/analytics.py) | Research-grade tools: Newey-West (HAC) & Lo (2002) Sharpe inference, calendar-time normalization (the "clock illusion"), rolling-Sharpe alpha decay, square-root market-impact capacity. |
| [`overnight/plots.py`](overnight/plots.py) | Knuteson Figure-1(c) style (log) with magnitudes in plain text + `linear` toggle. |
| [`overnight/brokers/`](overnight/brokers/) | Swappable `BrokerBase` + MT5 template (`dry_run=True`). |

```text
overnight/        the package          examples/   runnable demos
  decompose.py      <- core              run_synthetic_demo.py   (offline, validated)
  data.py                                verify_world_indices.py (needs Internet)
  diagnostics.py
  backtest.py     tests/      pytest: decomposition identity, cost model, stats
  stats.py        notebooks/  01_for_the_curious / 02_for_the_quants (executed)
  plots.py        .github/    CI: tests + offline demo on Python 3.10–3.12
  brokers/{base,mt5_connector}.py
```

## Install & run

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1     |  *nix:  source .venv/bin/activate
pip install -r requirements.txt

python examples/run_synthetic_demo.py      # offline — no network needed
python examples/verify_world_indices.py    # real data — needs Yahoo/Internet
pytest -q                                  # run the test-suite
python -c "from overnight import decompose, data; print('ok')"   # smoke test
```

## The honest verdict (what the analysis actually supports)

Keep three levels apart — the pamphlet tends to blur them:

1. **The empirical fact is REAL** and well documented (Cooper, Cliff, and Gulen
   2008; Berkman et al. 2012; Lou, Polk, and Skouras 2019; Boyarchenko, Larsen,
   and Whelan 2023). Credit to Knuteson for publishing data and code.
2. **The magnitudes are INFLATED** by (a) 30-year compounding on a log scale,
   (b) split/dividend artefacts in free data, (c) selection/survivorship ("the 25
   most problematic"), and above all (d) the **clock illusion** — the night spans
   ~28 calendar hours vs the 6.5-hour day, so per-hour the edge is ~1.3×, not 4×.
   What survives is **mostly gap beta**, sits below trading costs, and has
   **decayed** (5-year overnight Sharpe ~2 → ~0.5).
3. **The fraud attribution is NOT proven.** Framed as a likelihood ratio, the
   headline pattern is ~equally probable under "risk premium + microstructure" as
   under manipulation, so it discriminates little; the *discriminating* evidence
   (the foreign-ETF and Chinese **T+1** inversions, Qiao and Dam 2020; and a
   single-digit-million capacity) points away from a global manipulator. The
   SEC's 2023 D.E. Shaw action concerned whistleblower-agreement language (Rule
   21F-17), **not** manipulation.

**Reality check:** the NSPY / NIWM night-effect ETFs launched June 2022 and were
liquidated August 2023 after heavy underperformance. A beautiful paper strategy
is worth no more than the paper until it pays real execution costs.

## A few quant gotchas this repo is careful about

- **Adjustment mode is a decision, not a detail** — it moves return between night
  and day (ex-dividend happens at the open). Default `split_only`; document yours.
- **Sharpe > raw return** — part of "overnight alpha" is a gap-risk premium
  (disguised beta): you carry equity risk every night.
- **The factor 2** — you cross the spread to buy *and* to sell, ~252×/year.
- **CFD/MT5 swap** — the overnight financing alone can erase the edge. The MT5
  loop refuses to trade if `swap_long` exceeds the expected edge.
- **Execution ≠ academic prints** — the anomaly is measured on close/open
  auctions retail can't touch; at T±5 min you're in continuous trading.

## Roadmap

- [x] Core decomposition + offline critique demo + tests (23 passing) + CI
- [x] Live world-index verification (the foreign-ETF inversion + China T+1 finding)
- [x] Two narrative notebooks (curious / quant), executed and reproducible
- [x] Statistics layer: bootstrap Sharpe CIs + alpha-vs-beta decomposition
- [ ] Alpaca paper-trading connector (`BrokerBase`) alongside MT5
- [ ] Capacity / slippage study by order size; market-neutral (long/short) variant
- [ ] Reproduce the India (Figure 8) artefacts on raw emerging *spot* data

## Citing this repository

A [`CITATION.cff`](CITATION.cff) is provided — use GitHub's **"Cite this
repository"** button, or:

> d'Erceville, Guillain. 2026. *Overnight Alpha: verifying and stress-testing the
> overnight return anomaly.* https://github.com/Guillain-RDCDE/Overnight-Alpha

## References

Citation style: author–date (Chicago / *JFE*). A literature map (which
explanation each paper argues) and machine-readable entries are in
[`docs/references.md`](docs/references.md) and [`references.bib`](references.bib).
Get the PDFs with `python papers/download_papers.py` (see [`papers/`](papers/README.md)).

Berkman, Henk, Paul D. Koch, Laura Tuttle, and Ying Jenny Zhang. 2012. "Paying
Attention: Overnight Returns and the Hidden Cost of Buying at the Open." *Journal
of Financial and Quantitative Analysis* 47 (4): 715–741.

Boyarchenko, Nina, Lars C. Larsen, and Paul Whelan. 2023. "The Overnight Drift."
*The Review of Financial Studies* 36 (9): 3502–3547.
[doi:10.1093/rfs/hhad020](https://doi.org/10.1093/rfs/hhad020). Working paper:
Federal Reserve Bank of New York Staff Reports, no. 917 (2020).

Cooper, Michael J., Michael T. Cliff, and Huseyin Gulen. 2008. "Return
Differences between Trading and Non-Trading Hours: Like Night and Day." Working
paper. [SSRN 1004081](https://ssrn.com/abstract=1004081).

Haghani, Victor, Vladimir Ragulin, and Richard Dewey. 2024. "Night Moves: Is the
Overnight Drift the Grandmother of All Market Anomalies?" *Journal of Investment
Management* 22 (2). Working paper (2022): [SSRN 4139328](https://ssrn.com/abstract=4139328).

Knuteson, Bruce. 2019. "Celebrating Three Decades of Worldwide Stock Market
Manipulation." [arXiv:1912.01708](https://arxiv.org/abs/1912.01708).

Knuteson, Bruce. 2020. "Strikingly Suspicious Overnight and Intraday Returns."
[arXiv:2010.01727](https://arxiv.org/abs/2010.01727).

Knuteson, Bruce. 2022. "They Still Haven't Told You."
[arXiv:2201.00223](https://arxiv.org/abs/2201.00223).

Knuteson, Bruce. 2023. "Nothing to See Here: How to Say It When You Need to."
[SSRN Working Paper 4619084](https://ssrn.com/abstract=4619084).

Lou, Dong, Christopher Polk, and Spyros Skouras. 2019. "A Tug of War: Overnight
Versus Intraday Expected Returns." *Journal of Financial Economics* 134 (1):
192–213. [doi:10.1016/j.jfineco.2019.03.011](https://doi.org/10.1016/j.jfineco.2019.03.011).

Qiao, Kenan, and Lammertjan Dam. 2020. "The Overnight Return Puzzle and the
'T+1' Trading Rule in Chinese Stock Markets." *Journal of Financial Markets* 50:
100534. [doi:10.1016/j.finmar.2020.100534](https://doi.org/10.1016/j.finmar.2020.100534).
