# Open-Alpha-Lab — an open quant research desk

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **An open hedge-fund research desk.** I take trading ideas — famous anomalies,
> folk strategies, things people swear by — and put each through the *same brutal
> protocol*, then publish the verdict, **edge or mirage**. Most are mirages. The
> honest write-up of *why* is the point.
>
> Built by someone who ran the real thing (a fully systematic global-macro fund,
> sub-\$100M → \$9B+/month) — so every idea is judged on **both** questions most
> repos skip: *is the signal real?* **and** *does it survive real execution and
> scale?*
>
> **Not investment advice.** Research & education. See [LICENSE](LICENSE).

---

## How the desk works

Every study answers two questions on a fixed rubric, so results are comparable:

| Axis | Stamps |
|---|---|
| **Signal** — is the effect statistically real? | `REAL` · `WEAK` · `NONE` |
| **Tradability** — does it survive costs, capacity and scale? | `INVESTABLE` · `FRAGILE` · `MIRAGE` |

The protocol each idea goes through:

1. **Decompose / measure** the raw effect (exact identities, no fitting).
2. **Robust inference** — Newey-West (HAC) and Lo (2002) SEs, bootstrap CIs,
   White (2000) Reality Check for data-snooping. *Is it real?*
3. **Critique the magnitude** — compounding, log-scale, data artefacts, selection,
   calendar-time confounds.
4. **Alpha vs beta** — how much is just risk premium you were always paid for?
5. **Execution & capacity** — cost sweeps, square-root market impact, the scale at
   which the edge dies. *Is it investable?*
6. **Verdict** — the two stamps above, with the numbers behind them.

Every study's front page also follows the **same seven narrative beats** (claim →
stakes → how we'd know → teardown → verdict → could you trade it? → going further),
so a reader always lands in the same place. The full house style — the beats, the
two-readers-one-page convention, the rubric — is written up in
**[METHODOLOGY.md](METHODOLOGY.md)**, and the drop-in scaffold lives at
[`studies/_TEMPLATE/`](studies/_TEMPLATE/).

---

## Studies

| # | Study | Question | Signal | Tradability |
|---|---|---|---|---|
| **[01](studies/01-overnight-anomaly/)** | **Overnight Anomaly** | Do stocks really make their money overnight — and is it manipulation? | `REAL` (HAC *t*≈5) | `MIRAGE` (mostly beta, dies after costs, capacity ~\$10M, decaying) |
| **[02](studies/02-falling-knife/)** | **Falling-Knife** | Does buying the Nasdaq-100 / S&P 500 after a −3% drop beat buying a random day? | `NONE` at −3% · `WEAK` in deep panic | `MIRAGE` (tiny crash-clustered capacity, fails out-of-sample) |
| **[03](studies/03-fear-gauge/)** | **Fear-Gauge** | Does buying the VIX spike / "VIX ≥ 30, double down at 50" beat buying a random day? | `REAL` for the level (VIX≥30) · `NONE` for the spike | `MIRAGE` (barely beats a −3% day, underperforms buy-and-hold, martingale draws down −33%) |

> **Study 01 in one line:** the overnight effect is *real and broad* (confirmed
> across ~441 S&P 500 stocks), but its magnitude is inflated by a calendar-time
> illusion, what remains is mostly gap-risk beta below trading costs, it doesn't
> scale, and the "market-manipulation" reading is not supported (Bayesian
> posterior ≈ 2–3%). Full audit, two narrative notebooks, a point-by-point
> rebuttal, and a working paper: **[studies/01-overnight-anomaly/](studies/01-overnight-anomaly/)**.

> **Study 02 in one line:** the famous "−3% dip" is folklore — buying it is
> statistically indistinguishable from buying a random day, on both the Nasdaq-100
> and the S&P 500, and the prettiest backtests collapse out-of-sample (data-mining).
> Genuine *panic* (−5% to −7%) does leave a real fingerprint of a bounce, but it
> fails the tests that matter for trading it: the clustering-aware bootstrap
> straddles zero, capacity is ~3 events a decade dominated by 2000/2008/2020, and a
> fixed deep-dip rule flips from positive to negative Sharpe out-of-sample. Method,
> figures and reproducible code: **[studies/02-falling-knife/](studies/02-falling-knife/)**.

> **Study 03 in one line:** the twin of 02 in **volatility space**. A high VIX
> *is* followed by a real S&P rebound — VIX≥30 beats a random day by ~1% a week and
> ~1.3% a month (p≈0.00 / 0.01) — but that's the trap: it's the **variance risk
> premium**, it doesn't significantly beat just buying a −3% day, and the famous
> *+30% spike* has no monthly edge at all once you leave its 2016–2026 window
> (excess +0.5% in-window → −0.02% full sample). Traded, it underperforms
> buy-and-hold (in cash ~88% of the time), and "double down at 50" is a martingale
> whose −33% worst drawdown the no-2008 window hides. Method, two notebooks and
> reproducible code: **[studies/03-fear-gauge/](studies/03-fear-gauge/)**.

Ideas in the queue: momentum vs the overnight/intraday split, the weekend effect,
post-earnings drift, pairs/cointegration decay. Suggestions welcome via issues.

---

## The engine — `quantlab/`

A small, tested, reusable toolkit that powers every study:

| Module | Role |
|---|---|
| [`quantlab/decompose.py`](quantlab/decompose.py) | Exact overnight/intraday/close-close return decomposition + summary (Sharpe). |
| [`quantlab/data.py`](quantlab/data.py) | Yahoo fetch + parquet cache; split/total-return/raw adjustment modes. |
| [`quantlab/diagnostics.py`](quantlab/diagnostics.py) | Critique layer (offline): compounding, split-artefact injector/detector, synthetic markets. |
| [`quantlab/backtest.py`](quantlab/backtest.py) | Cost-aware backtest, break-even cost, cost sweep. |
| [`quantlab/stats.py`](quantlab/stats.py) | Bootstrap Sharpe CIs, alpha-vs-beta (gap-risk) decomposition. |
| [`quantlab/analytics.py`](quantlab/analytics.py) | HAC & Lo (2002) inference, calendar-time normalization, rolling-Sharpe decay, market-impact capacity. |
| [`quantlab/universe.py`](quantlab/universe.py) | Firm-level cross-section across an index (S&P 500 breadth). |
| [`quantlab/simulate.py`](quantlab/simulate.py) | Adversarial steelman of a strategy/manipulator P&L vs capital. |
| [`quantlab/bayes.py`](quantlab/bayes.py) | Bayesian hypothesis posteriors + White (2000) Reality Check. |
| [`quantlab/plots.py`](quantlab/plots.py) | Decomposition / grid plots. |
| [`quantlab/brokers/`](quantlab/brokers/) | Swappable `BrokerBase` + MT5 template (`dry_run=True`). |

```text
Open-Alpha-Lab/
├── quantlab/                     # the reusable research engine
├── tests/                        # deterministic test-suite (CI on 3.10–3.12)
├── studies/
│   ├── 01-overnight-anomaly/     # study #1: notebooks, paper, RESPONSE, docs, data
│   ├── 02-falling-knife/         # study #2: buy-the-dip — package + notebooks + examples + tests
│   └── 03-fear-gauge/            # study #3: buy-the-VIX-spike — twin of #2 in vol space
└── pyproject.toml · CITATION.cff · LICENSE
```

## Quickstart

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1   |   *nix:  source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                                   # the engine's test-suite
python studies/01-overnight-anomaly/examples/run_synthetic_demo.py   # offline, no network
python -c "import quantlab; print('ok')"                    # smoke test
```

Then open **[studies/01-overnight-anomaly/](studies/01-overnight-anomaly/)** —
start with the notebook *for the curious*, or read the working paper.

## Citing

A [`CITATION.cff`](CITATION.cff) is provided — use GitHub's **"Cite this
repository"** button.

---

*Built by [Guillain d'Erceville](https://github.com/Guillain-RDCDE) — production
systems, trading & market-data plumbing, and a habit of publishing the
dead-ends, not just the wins.*
