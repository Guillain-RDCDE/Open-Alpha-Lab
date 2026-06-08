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
| **[07](studies/07-coiled-spring/)** | **Coiled-Spring** | Does a stock resting on its rising 20-EMA spring into the trading-book's "explosive" +30-50% breakout? | `WEAK` (breakout beats a random same-stock entry by only +1.2%/10d, HAC *t*≈2.0; **Explosive as advertised?** `BUSTED` — only 1.7% of trades do +30%) | `FRAGILE` (median trade −0.25%, win 41%, per-trade Sharpe 0.05; bull-regime beta; break-even ≈75 bps, which the small-caps it targets blow through) |
| **[08](studies/08-true-strength/)** | **True-Strength** | Is the "True" Strength Index a *truer* momentum read than the MACD/RSI, or the same trade repainted? | `NONE` (TSI 84% spanned by MACD+RSI, pooled R²=0.84; same position as MACD 99.4% of days; equity-curve ρ=0.994; **"Truer"?** `BUSTED`) | `MIRAGE` (the 0.61 crossover Sharpe is long-equity beta — long/short timing Sharpe collapses to 0.05; nothing the cheaper MACD doesn't give, decays 0.77→0.15 over 0–40 bps) |
| **[09](studies/09-phantom-kernel/)** | **Phantom-Kernel** | Does market-making's "optimal spread" (Avellaneda-Stoikov) rest on an order-arrival law real markets obey? | `NONE` (under heavy-tailed reach the kernel is a **power law**, not the assumed exponential: R²=0.9996 vs 0.68, AIC +1.26M; a static *k* misprices the spread ±163%; **"Optimal spread" the source of edge?** `MISATTRIBUTED` — the value is in the *k*-free inventory skew) | `FRAGILE` (a brainless inventory clamp beats full AS when inventory is cheap, Sharpe 3.27 vs 1.59; AS wins only in jumpy/informed markets, 2.12 — and the touted rolling-vol "fix" collapses to 0.17 under jumps) |
| **[10](studies/10-markov-mint/)** | **Markov-Mint** | Can a Markov-chain → Monte-Carlo → calibration → Kelly pipeline (a viral Polymarket thread) "win every single trade"? | `NONE` (on a *provably-fair* martingale market the directional edge is −0.68 pp, HAC *t* = −0.77; oracle edge exactly 0; the raw MC "edge" is noise whose std collapses 19.8→2.1 pp as history grows; **"Win every trade"?** `BUSTED` — win rate 51.6%) | `MIRAGE` (Kelly-sized vs truth the bankroll → 0.0003× @2¢, 0.0017× @0 — it loses *before* costs because the calibration table's 0.958 ceiling forces a BUY NO on every favorite; the one real effect, the longshot bias, nets −13.6%/trade even for an oracle) |

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

> **Study 07 in one line:** the desk's first **chart-pattern** study — a retail trading book
> (*Trade the 20 EMA*) promises "explosive" +30-50% pops in days from a tidy three-step setup:
> a stock breaks above its rising 20-EMA, pulls back *without losing the EMA*, then breaks its
> pivot high on **2× volume** — buy it. Mechanised verbatim (no fitted parameter) over the
> cached liquid 174-name universe, 1962–2026, and tested **exit-agnostically** against the only
> fair benchmark — a *random entry in the same stock* over the same horizon. There's a faint
> real pulse: the breakout beats that baseline by **+1.2% over 10 days** (HAC *t*≈2.0, and *0*
> at 5 days) — a *whisper* of short-term momentum, not fireworks, and not corrected for the
> universe of TA rules one could've searched. The book's promise is the survivor's tail: only
> **1.7%** of 1,674 breakouts do +30%, the **median trade loses (−0.25%)**, win rate is **41%**,
> and the positive mean is a thin right tail (per-trade Sharpe **0.05**). The "edge" clusters in
> momentum blow-off years (2000 +4.3%, 2020 +4.5%, 2024 +2.6%; 2008 −3.6%, 2013 −3.2%) — it's
> bull-regime beta — and break-even is **≈75 bps round-trip**, right where the small-caps the
> book actually trades live. Validated on a synthetic universe with planted springboards (the
> detector fires and the backtest pays), so the real verdict is the market talking.
> **Signal `WEAK` · Tradability `FRAGILE` · Explosive as advertised? `BUSTED`.** Method,
> reproducible run and two notebooks: **[studies/07-coiled-spring/](studies/07-coiled-spring/)**.

> **Study 08 in one line:** an **indicator-redundancy** study — the "**True** Strength Index"
> claims, by its very name, to be a cleaner, *truer* read on momentum than the MACD or the RSI
> (QuantifiedStrategies sells a 1.7-profit-factor gold backtest on it, rules paywalled). We
> can't test their hidden rule, so we test the claim the name makes: is the TSI a *distinct*
> signal? Computed on textbook settings over the cached 174-name universe, 1962–2026, with each
> oscillator reduced to a zero-centred, z-scored momentum level for a like-with-like compare.
> It isn't distinct: the TSI is **84% spanned** by the MACD line and RSI (pooled R²=0.835), takes
> the **same long/flat position as the MACD 99.4%** of days, and its long/short **equity curve
> correlates 0.994** with it — three indicators, one trade. Its standalone 0.61 crossover Sharpe
> is **long-equity beta**: symmetrise the position to cancel the unconditional drift and the
> oscillator's own *timing* is Sharpe **0.05** (MACD 0.05, RSI −0.29), and the thin remainder
> decays **0.77→0.15** across a 0–40 bps cost sweep. A grid Reality Check (White 2000) finds the
> best-tuned TSI is a *real but generic* momentum signal (p≈0) — which is the point: it's
> redundant, not fake. **Signal `NONE` · Tradability `MIRAGE` · "Truer" than MACD/RSI? `BUSTED`.**
> Method, reproducible run and two notebooks: **[studies/08-true-strength/](studies/08-true-strength/)**.

> **Study 09 in one line:** a **model-teardown** study — the Avellaneda-Stoikov market-making
> model (the two equations a generation of crypto/HFT bots quote from) is sold as the optimal
> spread you're "leaving money on the table" without. Its whole closed form rests on one
> assumption: order-arrival intensity fades *exponentially* with quote distance,
> `λ(δ)=A·e^(−kδ)`, with a stable `k`. Because we can't market-make on a CFD broker and AS is a
> *theorem about a model world*, the reproducible core is a **seed-fixed order-flow simulator**:
> a world A that obeys the assumptions (where the estimator must — and does — recover `k`, R²
> 0.99998) and a world B with the documented frictions (heavy-tailed order reach, jumps,
> informed flow). On realistic flow the kernel is a **power law, not an exponential** (R²=0.9996
> vs 0.68, AIC prefers it by +1.26M), and a `k` that drifts 4× intraday misprices the "optimal"
> spread by up to **±163%** — the celebrated equation is calibrated to a phantom. Yet AS still
> *works* in the hostile world (best-of-four Sharpe 2.12) — because the part that does the work,
> the inventory **skew**, contains **no `k`**; the phantom corrupts only the spread *width*.
> Meanwhile a four-line inventory clamp beats full AS whenever inventory is cheap (Sharpe 3.27
> vs 1.59), and the touted "rolling-vol" production fix collapses under jumps (0.17). **Signal
> `NONE` · Tradability `FRAGILE` · "Optimal spread" the source of edge? `MISATTRIBUTED`.**
> Method, reproducible run and two notebooks: **[studies/09-phantom-kernel/](studies/09-phantom-kernel/)**.

Ideas in the queue: momentum vs the overnight/intraday split, the weekend effect,
post-earnings drift, a cointegration-gated / stop-loss pairs variant (the beat-7 forks of
Study 05), a wavelet / VIX-futures-term-structure follow-up to Study 06's cycle null, and a
genuinely small-cap universe + White (2000) Reality Check for Study 07's breakout rule.
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
│   ├── 06-clockwork-vol/         # study #6: VIX fixed-period cycles vs an AR(1) red-noise null
│   ├── 07-coiled-spring/         # study #7: the "20 EMA pivot breakout" chart rule, tested honestly
│   ├── 08-true-strength/         # study #8: is the "True" Strength Index distinct from MACD/RSI, or a repaint?
│   ├── 09-phantom-kernel/        # study #9: does Avellaneda-Stoikov's "optimal spread" rest on a real arrival law?
│   └── 10-markov-mint/           # study #10: "Markov chain wins every trade" on Polymarket — tested on a martingale null
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
