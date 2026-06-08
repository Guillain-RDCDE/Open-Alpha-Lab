# Open-Alpha-Lab — an open quant research desk

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **An open hedge-fund research desk.** I take trading ideas — famous anomalies,
> folk strategies, things people swear by — and put each through the *same brutal
> protocol*, then publish the verdict, **edge or mirage**. Most are mirages. The
> honest write-up of *why* is the point.
>
> Built by someone who ran the real thing — a fully systematic global-macro book
> scaled from sub-\$100M to **\$9B+ in monthly traded notional** — so every idea is
> judged on **both** questions most repos skip: *is the signal real?* **and** *does
> it survive real execution and scale?*
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
| **[04](studies/04-social-oracle/)** | **Social-Oracle** | Does following a viral social-media crowd's cashtag surges actually pay? | `NONE` (no abnormal edge; −0.66% vs random by 1mo) | `MIRAGE` (gross is pure beta, median trade −1.3%, sleeve −44%) |
| **[05](studies/05-twin-spread/)** | **Twin-Spread** | Does textbook pairs trading (GGR 1999) still pay after the world copied it? | `NONE` (no convergence edge; −0.48%/mo gross in the modern era, Sharpe CI [−0.84, −0.03]) | `MIRAGE` (negative before costs, −85% drawdown, β≈0; **Decay** `CONFIRMED`, and the obvious fixes don't rescue it) |
| **[06](studies/06-clockwork-vol/)** | **Clockwork-Vol** | Does the VIX run on a fixed-period (40-/80-day) cycle you can time, or are its "cycles" shapes in red noise? | `NONE` (claimed periods sit *inside* the AR(1) red-noise envelope: p ≈ 0.998/0.9995/0.99; **Fixed clock** `NOT SUPPORTED`, period wanders 83–333 sessions) | `MIRAGE` (walk-forward forecast is a coin flip; cycle trade Sharpe 0.33 < buy-and-hold 0.56 and < its random-phase null, p≈0.74 — diluted beta) |

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

> **Study 04 in one line:** the desk's first study whose trigger lives in the
> **information flow**, not the price tape — measured on **1,468 real
> r/WallStreetBets viral surges** (2021–2025, CC-BY data, abnormal vs SPY). Buying
> what the crowd screams carries **no abnormal edge**: a tiny insignificant one-day
> flicker (+0.08%, p≈0.23) that *fades to negative* by a month (−0.66% vs a random
> day, the crowd's pick less likely to be up than a coin flip), worse than the
> momentum the name already had, and robust to a name jackknife. The backtest's gross
> +0.72%/trade is **pure market beta** (abnormal +5 bps); the **median trade is
> −1.3%**, costs erase the rest, and the equal-weight follower's sleeve compounds to
> **−44% with an −84% drawdown** — while **42 of the most-viral names literally
> delisted** (a survivorship bias the result survives anyway). A single-guru feed
> (Serenity) is the motivating anecdote; the reproducible crowd is the measurable
> instance. Method, real run and code: **[studies/04-social-oracle/](studies/04-social-oracle/)**.

> **Study 05 in one line:** the desk's first **cross-sectional** study — the bet isn't on
> one name moving but on two moving back *together*. We run the textbook **GGR (1999)**
> minimum-distance rule (the one a [viral thread](https://x.com/MatiasScalbi/status/2063042609816252666)
> resurfaced: ~1.4%/mo, Sharpe ~0.6, near-zero beta, "still paying after publication") over
> a cached liquid 174-name universe, 1962–2026. The machinery is sound — on a synthetic
> universe with real cointegrated twins it recovers ~85–100% of them and harvests a
> +0.95%/mo. On **real** pairs it doesn't pay: the modern era (2005–2026, the only stretch
> with enough names for tight pairs) earns **−0.48%/mo gross** (Sharpe −0.44, bootstrap CI
> [−0.84, −0.03], **negative even at a zero spread** — so it isn't a cost artefact),
> **−0.54%/mo net** with a **−85% drawdown**, cleanly market-neutral (β≈0) so there's no
> beta to bank. Win rate 56.2% — *more winners than losers* — and still a loss: the
> short-gamma tail of pairs that break and never reconverge. The good years cluster in
> 1983–2004; the modern era is mostly red, green only in dislocations (2008 +0.9%/mo). And
> the obvious modern fixes don't rescue it — a stop-loss tames the −85% drawdown to −24% but
> leaves it ~flat-negative, a cointegration gate doesn't help. A textbook the market
> arbitraged past. Method, reproducible run and two notebooks:
> **[studies/05-twin-spread/](studies/05-twin-spread/)**.

> **Study 06 in one line:** the desk's first study of **periodicity** — does the VIX run on a
> fixed-period clock (a viral [cycles thread](https://x.com/Namzes_G) dates an "80-day cycle low
> to May 29", a 40-day cresting late July, synced to a stock 20-week low)? Tested against an
> **AR(1) red-noise null** (the thing that fakes "cycles" to the eye), every period the thread
> names — VIX **40d / 80d**, stocks' **100d / 250d / 1000d** — sits *inside* the noise envelope
> (p ≈ 0.998 / 0.9995 / 0.99 / 0.9995 / 0.76): noise routinely fakes peaks taller than the VIX's.
> The "dominant period" wanders **83→333 sessions** (it must be re-drawn — a curve-fit, not a
> clock), the walk-forward forecast is a **coin flip** (49–51%, p ≈ 0.74), and the tradeable
> expression earns **Sharpe 0.33 — below buy-and-hold's 0.56 and below its own random-phase
> null** (diluted beta from 59% exposure, not timing). The machinery is validated on a synthetic
> series with a *real* cycle baked in (it lights up 53–70× over the envelope, forecasts at 89%) —
> so the VIX's silence is a fact about the market. The one flicker — hard-wiring exactly 80 days
> gives 53% at p≈0.04 — is a single uncorrected period-search the spectral test contradicts.
> **Signal `NONE` · Tradability `MIRAGE` · Fixed clock `NOT SUPPORTED`.** Method, reproducible
> run and two notebooks: **[studies/06-clockwork-vol/](studies/06-clockwork-vol/)**.

Ideas in the queue: momentum vs the overnight/intraday split, the weekend effect,
post-earnings drift, a cointegration-gated / stop-loss pairs variant (the beat-7 forks of
Study 05), and a wavelet / VIX-futures-term-structure follow-up to Study 06's cycle null.
Suggestions welcome via issues.

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
| [`quantlab/repro.py`](quantlab/repro.py) | Reproducibility stamp: pin an as-of date + content fingerprint so headline numbers reproduce (or flag drift). |
| [`quantlab/brokers/`](quantlab/brokers/) | Swappable `BrokerBase` + MT5 template (`dry_run=True`). |

```text
Open-Alpha-Lab/
├── quantlab/                     # the reusable research engine
├── tests/                        # deterministic test-suite (CI on 3.10–3.12)
├── studies/
│   ├── 01-overnight-anomaly/     # study #1: notebooks, paper, RESPONSE, docs, data
│   ├── 02-falling-knife/         # study #2: buy-the-dip — package + notebooks + examples + tests
│   ├── 03-fear-gauge/            # study #3: buy-the-VIX-spike — twin of #2 in vol space
│   ├── 04-social-oracle/         # study #4: follow-the-guru — event study in attention space
│   ├── 05-twin-spread/           # study #5: pairs-trading decay — relative-value in the cross-section
│   └── 06-clockwork-vol/         # study #6: VIX fixed-period cycles vs an AR(1) red-noise null
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
