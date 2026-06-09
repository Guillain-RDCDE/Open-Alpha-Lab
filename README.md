<div align="center">

# Open-Alpha-Lab

### An open quant research desk.

I take famous trading ideas — anomalies, folk strategies, things people swear by —
put each through the **same brutal protocol**, and publish the verdict: **edge or mirage.**

***Most are mirages. The honest write-up of why is the point.***

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

> Built by someone who ran the real thing — a fully systematic global-macro book scaled
> from sub-\$100M to **\$9B+ in monthly traded notional** — so every idea is judged on the
> two questions most repos skip: **is the signal real?** *and* **does it survive real
> execution and scale?**

## How it works

Every idea goes through the *same* protocol and earns **two stamps**, so results are comparable:

| | |
|---|---|
| **Signal** — is the effect statistically real? | 🟢 `REAL`&nbsp;·&nbsp; 🟡 `WEAK`&nbsp;·&nbsp; 🔴 `NONE` |
| **Tradability** — does it survive costs, capacity & scale? | 🟢 `INVESTABLE`&nbsp;·&nbsp; 🟡 `FRAGILE`&nbsp;·&nbsp; 🔴 `MIRAGE` |

Robust inference (Newey-West / Lo SEs, bootstrap CIs, White Reality Check for data-snooping),
an honest alpha-vs-beta split, and a square-root market-impact capacity test — the full house
style is written up in **[METHODOLOGY.md](METHODOLOGY.md)**.

---

## The studies

| # | Study | The claim — tested to destruction | Real? | Tradable? |
|:--:|---|---|:--:|:--:|
| **[01](studies/01-overnight-anomaly/)** | **Overnight Anomaly** | Do stocks really make all their money overnight? | 🟢 Real | 🔴 Mirage |
| **[02](studies/02-falling-knife/)** | **Falling-Knife** | Does buying the dip (−3%) beat a random day? | 🔴 None | 🔴 Mirage |
| **[03](studies/03-fear-gauge/)** | **Fear-Gauge** | Does buying the VIX spike pay? | 🟡 Mixed | 🔴 Mirage |
| **[04](studies/04-social-oracle/)** | **Social-Oracle** | Does following a viral crowd's stock picks pay? | 🔴 None | 🔴 Mirage |
| **[05](studies/05-twin-spread/)** | **Twin-Spread** | Does textbook pairs trading still pay after everyone copied it? | 🔴 None | 🔴 Mirage |
| **[06](studies/06-clockwork-vol/)** | **Clockwork-Vol** | Does the VIX run on a fixed-period cycle you can time? | 🔴 None | 🔴 Mirage |
| **[07](studies/07-coiled-spring/)** | **Coiled-Spring** | Does the "20-EMA breakout" deliver explosive +30% pops? | 🟡 Weak | 🟡 Fragile |
| **[08](studies/08-true-strength/)** | **True-Strength** | Is the "True" Strength Index truer than MACD/RSI? | 🔴 None | 🔴 Mirage |
| **[09](studies/09-phantom-kernel/)** | **Phantom-Kernel** | Is Avellaneda-Stoikov's "optimal spread" built on a real arrival law? | 🔴 None | 🟡 Fragile |
| **[10](studies/10-markov-mint/)** | **Markov-Mint** | Can a Markov-chain pipeline "win every trade" on Polymarket? | 🔴 None | 🔴 Mirage |
| **[11](studies/11-vanishing-penny/)** | **Vanishing-Penny** | How fast does a guaranteed \$40M Polymarket arbitrage close? | 🟢 Real | 🔴 Mirage |
| **[12](studies/12-paper-prophet/)** | **Paper-Prophet** | Does an ARIMA+GARCH stack forecast the SPY, or is it vol-targeting in a trenchcoat? | 🔴 None | 🔴 Mirage |
| **[13](studies/13-crimson-hour/)** | **Crimson-Hour** | Does a red opening hour + IB-rejection really call the close at 88%? | 🟡 Weak | 🔴 Mirage |
| **[14](studies/14-gamma-gospel/)** | **Gamma-Gospel** | Does dealer gamma (GEX) call the day's character, or is it the VIX in a trenchcoat? | ⏳ Pre-reg | ⏳ Pre-reg |
| **[15](studies/15-sigma-sleight/)** | **Sigma-Sleight** | Does length-aware "AdaptiveRSI" beat fixed 70/30, or is the σ-transform a monotone relabel? | 🟡 Weak | 🔴 Mirage |

> **Click any study** for the full teardown — two narrative notebooks (one for the curious,
> one for the quant), reproducible code, and every number behind the two stamps.

*In the queue:* the weekend effect, post-earnings drift, a cointegration-gated pairs variant,
a VIX term-structure follow-up. Suggestions welcome via issues.

---

## Run it

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1   |   *nix:  source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                                            # the engine's test-suite
python studies/01-overnight-anomaly/examples/run_synthetic_demo.py   # offline, no network
```

Then open **[studies/01-overnight-anomaly/](studies/01-overnight-anomaly/)** — start with the
notebook *for the curious*, or read the working paper.

<details>
<summary><b>The engine — <code>quantlab/</code></b> (a small, tested, reusable toolkit that powers every study)</summary>

<br>

| Module | Role |
|---|---|
| [`decompose.py`](quantlab/decompose.py) | Exact overnight/intraday/close-close return decomposition + Sharpe summary. |
| [`data.py`](quantlab/data.py) | Yahoo fetch + parquet cache; split/total-return/raw adjustment modes. |
| [`diagnostics.py`](quantlab/diagnostics.py) | Critique layer (offline): compounding, split-artefact injector/detector, synthetic markets. |
| [`backtest.py`](quantlab/backtest.py) | Cost-aware backtest, break-even cost, cost sweep. |
| [`stats.py`](quantlab/stats.py) | Bootstrap Sharpe CIs, alpha-vs-beta (gap-risk) decomposition. |
| [`analytics.py`](quantlab/analytics.py) | HAC & Lo (2002) inference, calendar-time normalization, rolling-Sharpe decay, market-impact capacity. |
| [`universe.py`](quantlab/universe.py) | Firm-level cross-section across an index (S&P 500 breadth). |
| [`simulate.py`](quantlab/simulate.py) | Adversarial steelman of a strategy/manipulator P&L vs capital. |
| [`bayes.py`](quantlab/bayes.py) | Bayesian hypothesis posteriors + White (2000) Reality Check. |
| [`plots.py`](quantlab/plots.py) | Decomposition / grid plots. |
| [`repro.py`](quantlab/repro.py) | Reproducibility stamp: pin an as-of date + content fingerprint so headline numbers reproduce. |
| [`brokers/`](quantlab/brokers/) | Swappable `BrokerBase` + MT5 template (`dry_run=True`). |

```text
Open-Alpha-Lab/
├── quantlab/        # the reusable research engine
├── tests/           # deterministic test-suite (CI on 3.10–3.12)
├── studies/         # one folder per study: notebooks, code, data, docs
└── pyproject.toml · CITATION.cff · LICENSE
```

</details>

A [`CITATION.cff`](CITATION.cff) is provided — use GitHub's **"Cite this repository"** button.

---

<div align="center">

*Built by [**Guillain d'Erceville**](https://github.com/Guillain-RDCDE) — production systems,
trading & market-data plumbing, and a habit of publishing the dead-ends, not just the wins.*

**Not investment advice.** Research & education only. See [LICENSE](LICENSE).

</div>
