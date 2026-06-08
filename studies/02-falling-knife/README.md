# Study 02 — Falling-Knife 🔪 — does buying the dip actually pay?

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the house style,
> see the [methodology](../../METHODOLOGY.md). This page follows the desk's
> standard seven beats. Companion study: [01 — Overnight Anomaly](../01-overnight-anomaly/).*

## Verdict — read this first

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | `NONE` at −3% · `WEAK` in deep panic (−5%/−7%) | At −3% the excess over a random day is ~0 (p≈0.8 on ^NDX, p≈0.25 on QQQ); a real, significant bounce only appears at −5%/−7%. |
| **Tradability** — does it survive costs, capacity, scale? | `MIRAGE` | The deep-dip excess straddles zero under a clustering-aware bootstrap; capacity ~3 events/decade dominated by 2–3 crashes; a fixed rule flips +1.30 → −1.35 Sharpe out-of-sample. |

> **In one sentence:** the famous −3% dip is folklore — indistinguishable from
> buying a random day — and even the genuine panic-bounce at −5%/−7% fails the
> tests that matter for trading it (clustering, capacity, out-of-sample).

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

---

## 1 · The Claim

*"Buy when there's blood in the streets."* The most repeated piece of market
folk-wisdom there is. On the Nasdaq-100 the specific version people quote is the
**−3% day**: the index closes down 3%, you buy, you wait for the bounce. Stated
testably: *buying after a −3% close earns more than buying on an ordinary day.*

But a single threshold is not a single strategy. There are four defensible ways to
define "the falling knife", and they can disagree wildly — so we test the whole
**family**, not one lucky member:

| ID | Definition | Reads as |
|---|---|---|
| **T1** | session **closes** ≤ −3% (close-to-close) | the classic down-3%-day |
| **T2** | price trades ≤ −3% below the open **intraday** | most reactive, same-day fill |
| **T3** | close is ≥ 3% below a **rolling high** (drawdown) | the slide already started |
| **T4** | **cumulative** return over N days ≤ −3% | the slow bleed |

Each feeds the *same* event study, benchmark and backtest, so the comparison is
apples-to-apples.

## 2 · So What?

If the dip pays, it's a simple edge any retail trader could run. If it *doesn't*,
millions are taking on **extra fear, extra drawdown and extra trading** to earn
exactly what they'd have made sitting still.

And there's a deeper, universal lesson — **the absolute-Sharpe trap**. In a
synthetic market with **no dip-edge at all** (just upward drift + clustered
volatility), a family scan still produces a rule with **Sharpe > 2** that even
*survives* a deflated-Sharpe selection test. It isn't skill — it's the market's
drift, harvested by an asymmetric take-profit/stop, dressed up as alpha. The
**only** tool that flags it as fake is the conditional-vs-random-day benchmark
(excess ≈ 0). *Lesson: a high, even selection-robust, backtest Sharpe can be pure
beta. Always benchmark against the random day.*

## 3 · How We'd Know

The market drifts up anyway, so a green equity curve proves nothing. The real
question isn't *"did it go up after the dip?"* but *"did it go up **more** than on
a normal day?"* That single re-framing drives five methodological commitments:

- **Excess over a random-day null**, by permutation — never absolute return.
- **Block bootstrap**, because −3% days cluster (2008, 2020, 2022): 60 raw events
  may be ~10 independent episodes.
- **Deflated Sharpe** over the whole (trigger × exit) grid — scanning dozens of
  rules guarantees a lucky winner.
- **Conservative fills** — when a bar touches both stop and target, the stop wins.
- **Panic slippage** — a separate entry-only cost, because you buy into a crash
  when spreads gape.

We run it on **two faces** of the same market, and always report both:

| Symbol | History | Role |
|---|---|---|
| `^NDX` | since 1985 | spot index — deep sample (1987, full 2000 bust), great stats, **not tradeable** |
| `QQQ` | since 1999 | the ETF you could actually trade — real prints, **shorter** sample |

The full teardown is in two notebooks — the *same story at two altitudes*:

| | For whom | Inside |
|---|---|---|
| **[`notebooks/01_for_the_curious.ipynb`](notebooks/01_for_the_curious.ipynb)** | the curious | the one trap (random-day baseline), the −3% non-result, and the deep-panic twist, in plain language |
| **[`notebooks/02_for_the_quants.ipynb`](notebooks/02_for_the_quants.ipynb)** | quants | event study, permutation benchmark, block bootstrap, threshold sweep, family scan + deflated Sharpe, regime split, capacity, and the in/out-of-sample collapse |

Prefer scripts? `python examples/verify_ndx.py` (live ^NDX + QQQ),
`sweep_thresholds.py`, `panic_zoom.py`, `compare_indices.py`. Run
`examples/run_synthetic_demo.py` first (offline) to watch the toolchain tell a real
edge from a fake one.

## 4 · The Teardown

All from real Yahoo! Finance data (`split_only`), reproducible with one command.

**The decisive test — does the −3% day beat a random day?** (T1, forward return vs
the random-day null)

| Face | Sample | Events | +5d excess | p_greater | +10d excess | p_greater |
|---|---|---|---|---|---|---|
| **^NDX** | 1985→2026 | 127 | −0.29% | 0.83 | −0.49% | 0.87 |
| **QQQ** | 1999→2026 | 89 | +0.24% | 0.25 | +1.01% | 0.07 |

On 40 years of the spot index, buying the −3% close is **worse** than a random day;
on the ETF there's a faint lean by +10d but at p≈0.07 it's not distinguishable from
drift, and the block bootstrap straddles zero. **At −3%, there is no edge.**

**Is −3% special? Sweep the threshold.** Excess vs a random day, by threshold (★ =
p<0.05), on ^NDX:

| Drop | n | +1d | +5d | +10d | +20d |
|---|---|---|---|---|---|
| −2% | 245 | −0.01% | −0.39% | −0.37% | −0.15% |
| −3% | 127 | −0.12% | −0.29% | −0.49% | −0.53% |
| −4% | 70 | −0.29% | +0.42% | +0.35% | +0.56% |
| −5% | 36 | +0.53%★ | +0.57% | −0.19% | +0.24% |
| **−7%** | 12 | **+3.04%★** | **+2.25%★** | +2.07% | +2.61% |

The excess rises **smoothly and monotonically** with depth — the signature of a
*real* effect, not a data-mined fluke. Nothing at −3%; large and significant by −7%.
The S&P 500 shows the same shape. **The mean-reversion effect is real, but it lives
in genuine panic, not a −3% wobble.**

**The absolute-Sharpe trap, live.** The family scan's best cell is a drawdown
trigger (T3) with Sharpe **1.4–1.9** that even survives deflation — but the regime
split shows what it really is (QQQ, T3, hold ≤ 5d):

| Regime | Total return | Sharpe |
|---|---|---|
| Dot-com bust (2000-02) | **+92%** | 1.65 |
| QE bull (2009-20) | +47% | 1.72 |
| Post-COVID bull (2020-21) | +11% | 2.44 |
| **2022 bear** | **−15%** | **−3.31** |

Not a stable edge — leveraged exposure to *whether the dip keeps dipping*. It paid
when the Fed backstopped every selloff and bled in the one bear market without a
quick rescue.

**Does it generalise? Nasdaq vs S&P 500, in/out-of-sample.** A real mechanism
should be significant, appear on both indices, and survive OOS. Mostly it doesn't:

| Face | −3% +5d excess | p | Best family cell | Sharpe IS → OOS |
|---|---|---|---|---|
| ^NDX | −0.29% | 0.83 | T3 drawdown | 1.98 → 1.63 (drift) |
| QQQ | +0.24% | 0.24 | T3 drawdown +tp/sl | 3.41 → **0.21** |
| ^GSPC | −0.26% | 0.89 | T2 intraday | 4.23 → 2.41 (drift) |
| SPY | +0.14% | 0.33 | T1 close-to-close | 3.45 → **−5.42** |

The −3% non-edge is identical on both indices. The flashy Sharpes survive only on
the *spot* faces (persistent drift); on the *tradeable* ETFs they **collapse
out-of-sample** — textbook data-mining.

## 5 · The Verdict

> **At −3% exactly: no.** Indistinguishable from buying any random day, on the
> Nasdaq *and* the S&P 500 — you collect the index's drift with extra drama and
> worse drawdowns. Any flattering Sharpe is beta; on the tradeable ETFs it collapses
> out-of-sample.
>
> **At deeper drops (−5% to −7%): a genuine statistical fingerprint of panic
> mean-reversion — but not a tradeable edge.** The bounce is real in shape and grows
> smoothly with depth on both indices, yet it fails every test that matters for
> trading it (below).
>
> **Bottom line:** the round number is folklore, and even the panic tail doesn't
> survive clustering, capacity and out-of-sample scrutiny. At most it's a
> *behavioural discipline for deploying long-term capital into a crash*, never a
> standalone alpha engine.

## 6 · Could You Trade It?

No — and notably, **costs aren't the reason**. Three things kill it:

- **The clustering-aware bootstrap straddles zero.** Block-bootstrap 95% CI on the
  +5d excess includes 0 at every deep threshold (^NDX −7%: **[−3.7%, +7.1%]**; −5%:
  [−2.0%, +3.0%]). Permutation *overstated* the significance.
- **Capacity is tiny and crash-dominated.** −7% fires ~3×/decade (~once every 3–5
  years), and roughly half those events sit in 2000–2001. You're betting on a
  handful of historical crashes, not running a strategy.
- **It fails out-of-sample.** A fixed "buy −5% close, hold 5d" rule goes from
  in-sample Sharpe **+1.30 to out-of-sample −1.35** on the Nasdaq; the regime split
  swings from +3.6 Sharpe (GFC) to −8.3 (2022).

And the twist vs the overnight study: because you trade only a few times a year,
ordinary spread barely matters — even a brutal 40 bps of entry panic-slippage only
nudges the QQQ T3 Sharpe from ~1.0 to ~0.72, never negative. **The thing standing
between you and profit isn't transaction cost; it's that there's little excess edge
over plain beta to begin with.**

## 7 · Going Further

> **The door this leaves ajar.** A real bounce in *real* panic isn't nothing — it's
> the **liquidity-provision premium**: in a forced-seller cascade, whoever can
> warehouse the risk gets paid for it, and that payment is genuine. It failed *here*
> because we measured it raw and alone. As a *conditional* trade — switched on by a
> volatility, breadth or credit signal, and sized for how rarely it fires — it stops
> being folklore and starts being a risk premium with money behind it. We didn't
> find the edge; we found the door to it. *(This short-horizon reversal is a paid,
> documented liquidity-provision premium — Nagel 2012, "Evaporating Liquidity",
> RFS.)*

**What would change the verdict:** toward `INVESTABLE`, a deep-dip excess whose
*block-bootstrap* CI clears zero and survives an honest out-of-sample split, with
enough events to size a real book (these don't); toward a richer signal,
conditioning on volatility regime, breadth or credit stress rather than price alone
— the panic bounce may be a *conditional* effect we've only measured marginally.

**Caveats (read before quoting any number):** event clustering shrinks the
effective sample; dip-buying is regime-dependent (loves a "Fed put", hates a
2000–2002 grind); it's a short-vol profile (you add risk when vol peaks — watch max
drawdown, not just CAGR); and the −3% close on a chart is a print you may not get.

**Tests.** `pip install -r requirements-dev.txt && pytest -q` — 24 offline tests
covering the return identities, the conservative intrabar fill (stop-before-target),
cost monotonicity, the single-position no-overlap guarantee, and — most importantly
— that the benchmark **detects an injected edge and stays flat when there is none**.

**The real deliverable is the method:** any backtest that skips the random-day
benchmark, the threshold sweep, the block bootstrap and an out-of-sample split is
marketing. New studies follow the same seven beats
([METHODOLOGY.md](../../METHODOLOGY.md)).

---

MIT. **Research & teaching tool, NOT investment advice.** Backtested results do not
guarantee future performance. Test in a paper account before risking real capital.
