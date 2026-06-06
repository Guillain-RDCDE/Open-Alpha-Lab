# Study 02 — Falling-Knife 🔪 — does buying the Nasdaq-100 dip actually pay?

> *"Buy when there's blood in the streets."* Catchy. But on the Nasdaq-100, does
> buying after a **−3% drop** actually beat just… buying any random day? This study
> answers that question honestly, with reproducible code, for both beginners and
> quants.

| Axis | Verdict |
|---|---|
| **Signal** — is the effect statistically real? | `NONE` at −3% · `WEAK` in deep panic (−5%/−7%) |
| **Tradability** — does it survive costs, capacity, scale? | `MIRAGE` (tiny crash-clustered capacity, fails out-of-sample) |

**Short version of the answer:** the famous **−3%** level is *not* special — at
−3% you're just harvesting the index's drift, with no edge over buying a random
day, on the Nasdaq *and* the S&P 500. Deep panic (−5% to −7%) *does* leave a
genuine statistical fingerprint of a bounce — but it still isn't tradeable: the
clustering-aware bootstrap straddles zero, the events are rare and crash-dominated,
and a fixed rule flips negative out-of-sample. Full numbers in
[Results](#results-on-the-real-nasdaq-100) below — reproducible with one command.
The whole point of this project is that *an idea is not a strategy until it
survives costs, a fair benchmark, a threshold sweep, the block bootstrap, and an
out-of-sample test.*

---

## Why this exists

"Buy the dip" is the most repeated piece of market folk-wisdom there is. It *feels*
obviously true on the Nasdaq, which has spent decades making new highs. But that
feeling hides a trap: **the index goes up on average anyway**, so almost any
"buy and hold a few days" rule looks profitable in a backtest. A green equity
curve proves nothing on its own.

So we don't test *one* rule. We test the whole **family** of "−3% NDX" rules and
we hold every one of them to a single honest standard:

> Does buying the dip beat buying a **random day**, after **realistic costs**, and
> does it still work when you split history into **different market regimes**?

This is the same philosophy as the desk's companion study
[**01 — Overnight Anomaly**](../01-overnight-anomaly/): take a popular market idea,
build the tooling to verify it, and let the numbers — not the narrative — decide.

---

## Two doors: pick your level

### 🟢 If you're new to this

The single most important picture is the **event study**: line up every day the
Nasdaq fell 3%, and look at what happened *on average* over the following days.

- If the line goes **up** after the drop → dips tend to bounce (good for the idea).
- If it goes **flat or down** → "buy the dip" is a story, not an edge.

But there's a catch even beginners must internalise: **the market drifts up
anyway.** So the real question isn't "did it go up after the dip?" — it's "did it
go up *more* than on a normal day?" That comparison (we call it *conditional vs a
random day*) is the heart of everything here.

Run this — no internet, no setup beyond `pip install`:

```bash
python examples/run_synthetic_demo.py
```

It builds two fake markets — one with **no** dip-edge and one with a **real** one —
and shows the tools correctly telling them apart. That's your proof the
measurement works before we point it at the real Nasdaq.

### 🔵 If you're a quant

You get the full apparatus, each piece swappable:

| Module | What it gives you |
|---|---|
| `triggers.py` | Four entry definitions of "the −3% knife" (see below) |
| `exits.py` | Time / take-profit / stop / first-touch hybrid exits, as a sweepable grid |
| `eventstudy.py` | CAAR-style forward paths around t=0, with dispersion and t-stats |
| `benchmark.py` | **Conditional vs unconditional** forward returns + permutation p-values |
| `backtest.py` | Event-driven, single-position PnL with a *panic-slippage* entry cost |
| `sweeps.py` | **Threshold sweep** (is −3% special?) and look-back **N sweep** |
| `robustness.py` | Regime splits, **block bootstrap** (respects clustering), **deflated Sharpe**, **IS/OOS best-cell** test |
| `plots.py` | Event-study curve, equity+drawdown, cost sweep, family + threshold heatmaps |

Headline methodological commitments:
- **Excess over a random-day null**, estimated by permutation — not absolute return.
- **Block bootstrap**, because −3% days cluster (2008, 2020, 2022): 60 raw events
  may be ~10 independent episodes.
- **Deflated Sharpe** over the whole (trigger × exit) grid, because scanning dozens
  of rules guarantees one looks great by luck.
- **Conservative fills**: when a bar touches both stop and target, the stop wins.
- **Panic slippage**: a separate, entry-only cost, because you buy into a crash
  when spreads gape — the cost most dip backtests forget.

---

## The four ways to define a "falling knife"

A single threshold ("−3%") is not a single strategy. These four can disagree wildly:

| ID | Definition | Reads as |
|---|---|---|
| **T1** | session **closes** ≤ −3% (close-to-close) | the classic "down-3%-day" |
| **T2** | price trades ≤ −3% below the open **intraday** | most reactive, same-day fill |
| **T3** | close is ≥ 3% below a **rolling high** (drawdown) | "the slide already started" |
| **T4** | **cumulative** return over N days ≤ −3% | the slow bleed |

Each feeds the *same* event study, benchmark and backtest unchanged, so the
comparison is apples-to-apples.

---

## The one trap to remember (it's the whole game)

The offline demo makes it concrete. In a synthetic market with **no dip-edge at
all** — just upward drift and clustered volatility — the family scan still coughs
up a strategy with a **Sharpe above 2** that even *survives* the deflated-Sharpe
selection test.

It's not a real edge. It's the market's drift, harvested by an asymmetric
take-profit/stop, dressed up as skill. The **only** tool that correctly flags it
as fake is the conditional-vs-random-day benchmark, which reports an excess of
~0. The same benchmark lights up (excess strongly positive, p ≈ 0) on the
synthetic market where we *did* inject a real bounce.

**Lesson: a high — even selection-robust — backtest Sharpe can be pure beta/drift.
Always benchmark against the random day.**

---

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

python examples/run_synthetic_demo.py   # offline, validates the toolchain
python examples/verify_ndx.py           # live: ^NDX (since 1985) + QQQ (since 1999)
python examples/sweep_thresholds.py     # is −3% special? threshold + N sweeps
python examples/compare_indices.py      # Nasdaq vs S&P 500 + in/out-of-sample test
```

Run with the venv interpreter directly if you didn't activate it, e.g.
`./.venv/Scripts/python.exe examples/verify_ndx.py`.

`verify_ndx.py` downloads once from Yahoo! Finance, caches to `_cache/` as
parquet, and prints an end-to-end study for **both** the spot index (deep history,
great statistics) and the tradeable ETF (real execution), closing with a
plain-language verdict.

---

## Data: why two faces of the same index

| Symbol | History | Role |
|---|---|---|
| `^NDX` | since 1985 | spot index — deep sample (1987 crash, full 2000 bust), great stats, **not tradeable** |
| `QQQ` | since 1999 | the ETF you could actually trade — real auction prints, **shorter** sample |

`^NDX` tells you whether the effect is *real and stable*; `QQQ` tells you whether
it is *capturable*. We always report both.

**Adjustment mode is a decision, not a detail.** `split_only` (default) keeps
dividend drops in the price (closest to the tape); `total_return` uses fully
adjusted closes (cleaner compounding, but silently moves return between sessions).
Document your choice in any figure you publish.

---

## What we deliberately do NOT do

- **No leverage by default.** Buying into a crash on margin is how dip-buyers blow
  up. Financing/CFD-swap cost is available as a parameter so you can see the damage.
- **No look-ahead.** Events too close to the sample edges are dropped; intrabar
  fills resolve conservatively.
- **No cherry-picking.** The best cell of the family scan is treated as a *suspect*,
  not a result, until it clears deflation, the bootstrap, and an out-of-sample split.

---

## Results on the real Nasdaq-100

Numbers below are from `python examples/verify_ndx.py` (Yahoo! Finance, `split_only`
adjustment). Reproduce them yourself; they will extend as new data arrives.

### The decisive test — does the −3% day beat a random day?

Trigger T1 (a session that *closes* ≤ −3%), forward return vs the random-day null:

| Face | Sample | Events | +5d excess | p_greater | +10d excess | p_greater |
|---|---|---|---|---|---|---|
| **^NDX** | 1985→2026 | 127 | −0.29% | 0.83 | −0.49% | 0.87 |
| **QQQ** | 1999→2026 | 89 | +0.24% | 0.25 | +1.01% | 0.07 |

Read it plainly: on 40 years of the spot index, buying the −3% close is **worse**
than buying a random day. On the tradeable ETF there's a faint positive lean by
+10 days, but at p≈0.07 it is **not statistically distinguishable** from drift,
and the block bootstrap straddles zero (95% CI on the +5d excess: [−1.6%, +3.2%]).

### The absolute-Sharpe trap, live

The family scan's best cell is a **drawdown** trigger (T3) with a Sharpe of
**1.4–1.9** that even *survives* the deflated-Sharpe selection test. Tempting — but
the regime split shows what's really going on (QQQ, T3 drawdown, hold ≤ 5d):

| Regime | Total return | Sharpe |
|---|---|---|
| Dot-com bust (2000-02) | **+92%** | 1.65 |
| QE bull (2009-20) | +47% | 1.72 |
| Post-COVID bull (2020-21) | +11% | 2.44 |
| **2022 bear** | **−15%** | **−3.31** |

That is not a stable edge — it is leveraged exposure to *whether the dip keeps
dipping*. It paid spectacularly when the Fed backstopped every selloff and bled
in the one bear market without a quick rescue.

### Is −3% special? Sweep the threshold (`examples/sweep_thresholds.py`)

A single threshold proves nothing. Sweeping the drop size shows the −3% level is a
**round-number trap** — but something real hides at deeper drops. Excess return vs
a random day, by threshold × horizon (★ = p < 0.05):

**^NDX (1985→2026)**

| Drop | n | +1d | +5d | +10d | +20d |
|---|---|---|---|---|---|
| −2% | 245 | −0.01% | −0.39% | −0.37% | −0.15% |
| −3% | 127 | −0.12% | −0.29% | −0.49% | −0.53% |
| −4% | 70 | −0.29% | +0.42% | +0.35% | +0.56% |
| −5% | 36 | +0.53%★ | +0.57% | −0.19% | +0.24% |
| **−7%** | 12 | **+3.04%★** | **+2.25%★** | +2.07% | +2.61% |

The excess rises **smoothly and monotonically** as the drop deepens — the signature
of a *real* effect rather than a data-mined fluke. At −3% there is nothing; by −7%
the one-week bounce is large and significant. The S&P 500 shows the same shape
(−5%: +0.92% at +5d, p=0.01). **Reading: the mean-reversion "buy the dip" effect is
real, but it lives in genuine panic (−5% to −7%), not in a mild −3% wobble.**

Big honest caveat: −7% days are *rare* (n=12 on 40 years) and bunched in 2008/2020,
so that bottom row leans on a handful of clustered episodes. The block bootstrap
and regime split exist precisely to keep you humble about it.

### Panic zoom — does the deep-dip edge actually survive? (`examples/panic_zoom.py`)

This is the decisive follow-up: the −5%/−7% bounce is significant by the
*permutation* test, but permutation assumes independent draws and these events
cluster hard. Run the full honest battery and the edge thins out:

- **Clustering-aware bootstrap straddles zero.** The block bootstrap 95% CI on the
  +5d excess includes 0 at every deep threshold (^NDX −7%: **[−3.7%, +7.1%]**;
  −5%: [−2.0%, +3.0%]). Permutation *overstated* the significance — exactly the
  trap the code warns about.
- **Capacity is tiny and crash-dominated.** −7% fires ~3×/decade on the Nasdaq
  (~once every 3–5 years), and **half of those events sit in 2000–2001** (the S&P's
  deep-dip sample is dominated by 1932/1935). You're not running a strategy, you're
  betting on a handful of historical crashes.
- **It fails out-of-sample.** A fixed "buy −5% close, hold 5d" rule goes from
  in-sample Sharpe **+1.30 to out-of-sample −1.35** on the Nasdaq; the regime split
  swings from +3.6 Sharpe (GFC) to −8.3 (2022).

So the deep-dip bounce is a **genuine statistical fingerprint of panic
mean-reversion, but not a robust, tradeable edge**: too rare, too clustered, not
stable out-of-sample. The overlaid event-study paths (`out_panic_*.png`) show the
shape is real; the bootstrap, capacity and OOS tests show you can't bank on it.

### Does it generalise? Nasdaq vs S&P 500 (`examples/compare_indices.py`)

If the effect were a true market mechanism it should appear on both indices — and
the family-scan winners should survive out-of-sample. They mostly don't:

| Face | −3% +5d excess | p | Best family cell | Sharpe IS → OOS |
|---|---|---|---|---|
| ^NDX | −0.29% | 0.83 | T3 drawdown | 1.98 → 1.63 (drift) |
| QQQ | +0.24% | 0.24 | T3 drawdown +tp/sl | 3.41 → **0.21** |
| ^GSPC | −0.26% | 0.89 | T2 intraday | 4.23 → 2.41 (drift) |
| SPY | +0.14% | 0.33 | T1 close-to-close | 3.45 → **−5.42** |

The −3% non-edge is identical on both indices. The flashy family-scan Sharpes that
"survive" do so only on the *spot* faces — that's the persistent market drift. On
the *tradeable* ETFs the winners **collapse out-of-sample** (QQQ 3.4→0.2, SPY
3.5→−5.4): textbook data-mining.

### Costs are not the killer here (unlike overnight)

Because you only trade a few times a year, ordinary spread barely matters. Even a
brutal 40 bps of entry **panic-slippage** only drops the QQQ T3 Sharpe from ~1.0
to ~0.72 — it never flips negative. The thing standing between you and profit
isn't transaction cost; it's that **there's little excess edge over plain beta to
begin with**.

### Verdict

> **Is "buy the −3% Nasdaq dip" an interesting strategy?**
>
> **At −3% exactly: no.** It's indistinguishable from buying any random day, on the
> Nasdaq *and* the S&P 500 — you're collecting the index's drift with extra drama
> and worse drawdowns. Any flattering Sharpe from the family scan is market beta;
> on the tradeable ETFs it collapses out-of-sample (data-mining).
>
> **At deeper drops (−5% to −7%): there's a genuine statistical fingerprint of
> panic mean-reversion — but it is not a tradeable edge.** The bounce is real in
> shape and grows smoothly with depth on both indices, yet it fails every test that
> matters for actually trading it: the clustering-aware bootstrap CI straddles zero,
> the events are rare and dominated by 2–3 historical crashes (capacity ~3/decade),
> and a fixed deep-dip rule flips from positive to negative Sharpe out-of-sample.
>
> **Bottom line:** the round number is folklore, and even the panic tail — the one
> place the effect is statistically visible — doesn't survive clustering, capacity
> and out-of-sample scrutiny. At most it's a *behavioural discipline for deploying
> long-term capital into a crash*, never a standalone alpha engine. The real
> deliverable here is the method: any backtest that skips the random-day benchmark,
> the threshold sweep, the block bootstrap and an out-of-sample split is marketing.

## Caveats (read before quoting any number)

1. **Event clustering** shrinks the effective sample far below the raw count.
2. **Regime dependence**: dip-buying loves a "Fed put" and hates a 2000-2002 grind.
   One great decade is not a law.
3. **Short-vol profile**: you add risk exactly when volatility peaks. Positive
   average return can still hide an ugly tail — watch max drawdown, not just CAGR.
4. **Execution reality**: the −3% close you see on a chart is a print you may not
   get; modelled `panic_slippage_bps` is a guess, so sweep it.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite (24 tests, offline) covers the return-decomposition identities, the
conservative intrabar fill (stop-before-target), cost monotonicity, the
single-position no-overlap guarantee, and — most importantly — that the benchmark
**detects an injected edge and stays flat when there is none**, so you can trust
the verdicts on real data.

## License & disclaimer

MIT. **This is a research and teaching tool, NOT investment advice.** Backtested
results do not guarantee future performance. Test in a paper / demo account before
risking real capital. See [LICENSE](LICENSE).
