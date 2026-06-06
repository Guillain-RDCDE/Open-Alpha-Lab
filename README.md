# Overnight Alpha — verify the night-trade edge, then stress-test it honestly

[![tests](https://github.com/Guillain-RDCDE/overnight-alpha/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/overnight-alpha/actions/workflows/tests.yml)
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

This repo is a *response*, so start with the source. One command fetches the
openly available papers straight from arXiv (we don't redistribute the PDFs —
[here's why](papers/README.md)):

```bash
python papers/download_papers.py
```

- **Nothing to See Here** (2023) — [SSRN 4619084](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619084) *(login wall)*
- **Celebrating Three Decades of Worldwide Stock Market Manipulation** (2019) — [arXiv 1912.01708](https://arxiv.org/abs/1912.01708) · the figures we reproduce
- **They Still Haven't Told You** (2022) — [arXiv 2201.00223](https://arxiv.org/abs/2201.00223)

See [`papers/README.md`](papers/README.md) for the full reading list and licences.

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
| **[`notebooks/01_pour_les_curieux.ipynb`](notebooks/01_pour_les_curieux.ipynb)** | the curious | the story in plain language: the night/day pattern, then the three traps (compounding, dirty data, fees) that explain why it's subtler than it looks |
| **[`notebooks/02_pour_les_quants.ipynb`](notebooks/02_pour_les_quants.ipynb)** | quants / practitioners | real Yahoo data on 10 world indices, the critique with numbers, bootstrap Sharpe CIs, alpha-vs-beta decomposition, dividend-adjustment sensitivity, and a cost-aware backtest |

> Notebooks are *generated* from [`notebooks/build_notebooks.py`](notebooks/build_notebooks.py)
> then executed with `nbconvert` — so the figures you read are reproducible outputs, not screenshots.

### One real-data finding worth flagging

Run the world-index decomposition and the pattern is **not universal**. The US
(SPY, QQQ) and Brazil show the classic huge-overnight shape (overnight Sharpe
≈ 0.7, bootstrap 95% CI excludes zero). But the UK / Germany / France / Japan
ETFs are **inverted** — because they trade in New York while their underlying
markets trade *overnight* US-time, so the "overnight" window simply *contains
the home session*. The night/day split is relative to the listing clock, not a
universal anomaly. And China (FXI) overnight Sharpe ≈ 0.26 is **not statistically
distinguishable from zero** (bootstrap P(Sharpe<0) ≈ 9%), consistent with the
T+1 microstructure explanation rather than one global manipulator.

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
| [`overnight/plots.py`](overnight/plots.py) | Knuteson Figure-1(c) style (log) with magnitudes in plain text + `linear` toggle. |
| [`overnight/brokers/`](overnight/brokers/) | Swappable `BrokerBase` + MT5 template (`dry_run=True`). |

```text
overnight/        the package          examples/   runnable demos
  decompose.py      <- core              run_synthetic_demo.py   (offline, validated)
  data.py                                verify_world_indices.py (needs Internet)
  diagnostics.py
  backtest.py     tests/      pytest: decomposition identity, cost model, stats
  stats.py        notebooks/  01_pour_les_curieux / 02_pour_les_quants (executed)
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

1. **The empirical fact is REAL** and well documented (Lou–Polk–Skouras 2019,
   Cooper–Cliff–Gulen 2008, NY Fed). Credit to Knuteson for publishing data + code.
2. **The magnitudes are INFLATED** by (a) 30-year compounding on a log scale,
   (b) split/dividend artefacts in free data, (c) selection/survivorship
   ("the 25 most problematic").
3. **The fraud attribution is NOT proven.** The China test shows an *inverted*
   pattern, cleanly explained by the **T+1** rule (Qiao & Dam 2020) — awkward for
   a single global manipulator. And the SEC's 2023 D.E. Shaw action concerned
   whistleblower-agreement language (Rule 21F-17), **not** manipulation.

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

- [x] Core decomposition + offline critique demo + tests (16 passing) + CI
- [x] Live world-index verification (the foreign-ETF inversion + China T+1 finding)
- [x] Two narrative notebooks (curious / quant), executed and reproducible
- [x] Statistics layer: bootstrap Sharpe CIs + alpha-vs-beta decomposition
- [ ] Alpaca paper-trading connector (`BrokerBase`) alongside MT5
- [ ] Capacity / slippage study by order size; market-neutral (long/short) variant
- [ ] Reproduce the India (Figure 8) artefacts on raw emerging *spot* data

## References

Knuteson, *Celebrating Three Decades of Worldwide Stock Market Manipulation*
(arXiv [1912.01708](https://arxiv.org/abs/1912.01708)); *They Still Haven't Told
You* (arXiv [2201.00223](https://arxiv.org/abs/2201.00223)); *Nothing to See
Here* (SSRN 4619084). Lou, Polk, Skouras (2019); Cooper, Cliff, Gulen (2008);
Haghani et al. / Elm Wealth, *Night Moves* (2022); Qiao & Dam (2020, T+1).
See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for the full handoff notes.
