<div align="center">

<img src="docs/social-preview.png" width="100%" alt="Open-Alpha-Lab — famous trading edges, taken apart one protocol at a time">

# Open-Alpha-Lab

### Almost every famous trading edge is a mirage. Here is the graveyard, and the handful that survived.

I put every market anomaly, folk strategy and named factor
people swear by through the **same brutal protocol**, and publish the verdict: **edge or mirage.**

***Most are mirages. The honest write-up of why is the point. The survivors don't forecast anything — they manage risk, harvest a premium or a mechanical identity, stop a cost you were paying, or impose a discipline that pays for itself.***

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

> Built by someone who ran the real thing — a fully systematic global-macro book scaled
> from sub-\$100M to **\$9B+ in monthly traded notional** — so every idea is judged on the
> two questions most repos skip: **is the signal real?** *and* **does it survive real
> execution and scale?**

## The protocol

Every idea goes through the *same* protocol and earns **two stamps**, so results are comparable:

| | |
|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) |
| **Tradability** — does it survive costs, capacity & scale? | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) |

Robust inference (Newey-West / Lo SEs, bootstrap CIs, White Reality Check for data-snooping),
an honest alpha-vs-beta split, and a square-root market-impact capacity test — the full house
style is written up in **[METHODOLOGY.md](METHODOLOGY.md)**.

---

## The graveyard

The bench on one grid — each study a numbered chip, sorted by its two stamps.
Almost everything ends up bottom-right; the green corner is nearly empty — and not one chip in it
is a forecaster. A **Mixed** signal — the verdict splits by regime or leg — counts with Weak, in
the same amber bucket. The map is regenerated from the [ledger](docs/REFERENCE.md), which lists
every study and is the only place worth counting.

[![The bench map — every study placed on a Signal × Tradability grid](docs/bench_map.png)](https://guillain-rdcde.github.io/Open-Alpha-Lab/)

> **▶ [Explore the live map](https://guillain-rdcde.github.io/Open-Alpha-Lab/)** — the same grid, but zoomable:
> **click any chip to open its study**, search by name or claim, and filter the whole bench by verdict.
> (The static image above never gets less readable; the interactive page is where it scales.)

---

## Where to go next

| | |
|---|---|
| **[The full ledger →](docs/REFERENCE.md)** | Every study, two stamps each, with the greens called out first. This is the single source of truth the map and the live page are built from. |
| **[What the teardowns taught us →](docs/bench.md)** | The view from above: mortality by family of idea, and the lessons the bench keeps repeating. |
| **[The method →](METHODOLOGY.md)** | How a claim earns its two stamps — inference, the alpha-vs-beta split, the capacity test. |
| **[Reproduce the numbers →](docs/reproducibility.md)** | Data caches, fingerprints and the release bundle, to verify the published figures byte-for-byte. |

**New here?** Open **[study 01, *for the curious*](studies/01-overnight-anomaly/notebooks/01_for_the_curious.ipynb)** — one famous idea, taken apart in plain language, no finance background needed.

**Here for the method?** Open **[study 01, *for the quants*](studies/01-overnight-anomaly/notebooks/02_for_the_quants.ipynb)** and the **[working paper](studies/01-overnight-anomaly/paper/overnight_alpha.pdf)** — the same result with the inference, the robustness checks and the capacity work shown. Every study carries the same pair of notebooks.

---

## Run it

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1   |   *nix:  source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                                            # the engine's test-suite
python studies/01-overnight-anomaly/examples/run_synthetic_demo.py   # offline, no network
```

Then open **[studies/01-overnight-anomaly/](studies/01-overnight-anomaly/)**, or any of the
others — every study folder has the same shape.

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
