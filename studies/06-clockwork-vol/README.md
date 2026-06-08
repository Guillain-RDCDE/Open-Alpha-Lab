# Study 06 — Clockwork-Vol ⏰ — does the VIX run on a fixed-period clock, or are its "cycles" shapes in red noise?

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the house style, see the
> [methodology](../../METHODOLOGY.md). This page follows the desk's standard seven beats.
> Where Study 03 read the VIX as a **trigger** ("is a high VIX followed by a bounce?"), this
> one asks a different question of the same gauge: does it move on a **clock** — a fixed-period
> cycle you could mark on a calendar weeks ahead?*

## Verdict — read this first

*Measured on a **reproducible** run over the cached **^VIX** daily close, 1990–2026 (taken
raw — adjustment is meaningless for a level; see Study 03), worked in **log-VIX** (the honest
scale for *proportional* swings). The null is **AR(1) red noise** fitted at **ρ = 0.980** with
**2,000** Monte-Carlo surrogates. As-of **2026-06-01**, VIX fingerprint `6c2029b57135`; every
number in [`docs/results.md`](docs/results.md).*

| Axis | Stamp | Why (one line) |
|---|---|---|
| **Signal** — is there a real fixed-period cycle? | `NONE` | At every period the tweet names — VIX **40d** & **80d**, stocks' 20-week (**100d**), 1-year (**250d**), 4-year (**1000d**) — the periodogram peak sits **inside** the red-noise envelope (p = **0.998 / 0.9995 / 0.994 / 0.9995 / 0.765**): red noise fakes peaks *taller* than the VIX's almost every time. The only thing clearing the 99% envelope is a ~**15.6-session** wiggle (ratio 1.10), not the claimed clocks. |
| **Tradability** — does timing it pay? | `MIRAGE` | Walk-forward, the projected cycle calls the next move at **49–51%** (a coin; p = 0.76 / 0.74 / 0.078). The tradeable expression (long the S&P when the VIX cycle is projected to fall) earns **Sharpe 0.33** — *below* buy-and-hold's **0.56** and *below* the random-phase null's mean **0.37** (p = **0.74**): it's diluted beta from 59% exposure, not timing. |
| **A fixed clock?** — does the period even hold? | `NOT SUPPORTED` | The "dominant cycle" wanders from **83 to 333 sessions** (mean 227, σ **82**) across rolling 4-year windows — it has to be re-drawn every few months. A period that won't sit still isn't a clock; it's a curve-fit. |

> **In one sentence:** the VIX's tidy 40-/80-day "cycles" don't clear what AR(1) red noise
> invents on its own, their period won't hold still, and a walk-forward forecast built on them
> is a coin flip that loses to buy-and-hold — what looks like a clock is the eye reading rhythm
> into persistent noise.

> **Not investment advice.** Research & education. See [../../LICENSE](../../LICENSE).

---

## 1 · The Claim

A [cycles-analysis thread](https://x.com/Namzes_G) reads the VIX the way an astronomer reads
an orbit — as a stack of **fixed-period cycles** you can project forward. Stated at full
strength, the way its believers do:

> *"The VIX runs on an **80-day cycle** with a nested **40-day** one. The 80-day low formed on
> May 29; the first 40-day cycle should trend higher into late June, dip into early July, the
> second crest around July 24–27, then fall into early August where the 80- and 40-day cycles
> sync up. That lines up with stocks carving a **20-week cycle low** and rallying from there."*

It is a genuinely seductive picture, and worth steelmanning, because the VIX really *is*
forecastable in a loose sense — it mean-reverts, it clusters, calm follows storm. The claim
takes that grain of truth and makes it **precise and datable**: not "vol tends to come back
down" but "the next low is due on roughly *this* day, because the clock says so." If that were
real, it would be one of the cleanest market-timing edges in existence.

> 🔬 **For the quants** — H₁: the log-VIX periodogram has a peak at P ∈ {40, 80} sessions that
> **exceeds the (1−α) AR(1) red-noise quantile**; the period is **stable** across windows; and
> a fixed-period projection has **out-of-sample direction skill > ½**, beating a random-phase
> null. H₀: the VIX is **red noise** (AR(1), persistence ρ ≈ 0.98) plus mean reversion — its
> periodogram peaks are the broad, tall, *random* peaks autocorrelation manufactures, carrying
> no forward information and no stable period.

## 2 · So What?

If volatility ran on a fixed clock, everything that hangs off vol becomes timeable: buy the
dip exactly when the cycle says a vol *peak* (and a stock *low*) is forming, lift hedges when
it says calm is due, sell premium into the predicted crest. A reliable 80-day vol clock would
be a standing, repeatable edge — and, more importantly, it would say something deep about
markets: that volatility carries a **deterministic oscillation** a chartist can read, on top
of the random walk everyone else sees.

That's exactly why it deserves a hard look. "The market moves in cycles" is one of the
oldest, most intuitive, most *over-claimed* ideas in technical analysis — and the reason it
survives is that the human eye is a relentless pattern-matcher that finds rhythm in any
wandering line. The useful question isn't whether you *can* draw cycles on a VIX chart (you
always can); it's whether those cycles are **taller and more stable than the ones pure noise
draws for free.**

> 🔬 **For the quants** — the stakes are a clean **null-vs-signal** separation. A persistent
> series (ρ ≈ 0.98) has enormous low-frequency power; its raw periodogram is dominated by
> tall, broad peaks that *look* like 60–250-session cycles but are the spectral signature of
> autocorrelation, not periodicity. Any "edge" must be measured as the **excess** over that
> red-noise background, out-of-sample, with the period chosen causally — anything else is
> fitting the noise and calling it a clock.

## 3 · How We'd Know

The trap here is unusually pure: **you can always extract a cycle.** Bandpass any series
around 80 days and you get a clean 80-day oscillation back — that's what the filter *does*,
whether or not the cycle is real. So the eye-test ("look how regular!") is worthless. We
announce three sharper tests up front, and a *mirage* is any honest "no" to all three:

- **Does the peak beat red noise?** We fit an AR(1) null to log-VIX, Monte-Carlo thousands of
  surrogate red-noise series of the same length, and read the per-frequency **envelope**. A
  real cycle pokes above the 95/99% curve; a phantom hides inside it. **Mirage signal:** the
  40- and 80-day peaks sit *inside* the envelope.
- **Does the period hold still?** A real clock has a constant period. We slide the spectral
  window across 36 years and watch the dominant in-band period. **Mirage signal:** it wanders
  over a wide range — a cycle you must re-tune is a curve-fit.
- **Does it forecast out-of-sample?** We **walk forward**: fit the period and phase on the
  past only, project the next turn, and score the projection against what the market did next
  — versus a **random-phase null** that keeps the period and amplitude but scrambles the
  timing. **Mirage signal:** the learned phase forecasts no better than an arbitrary one.

And the honesty rail this study leans on hardest: **the synthetic proves the test.** We build
a fake log-VIX with a *real* fixed cycle baked into red noise; the detector must light up on
it. It does — 53–70× over the envelope, period stable to ±3 sessions, walk-forward skill 89%.
So when the same machine stays dark on the *real* VIX, that silence is a fact about the
market, not a dull detector.

> 🔬 **For the quants** — the shared desk protocol, powered by [`quantlab/`](../../quantlab/)
> and this study's [`vix_cycles/`](vix_cycles/): (1) measure the raw effect — periodogram of
> linearly-detrended, Tukey-tapered log-VIX; (2) robust inference — the AR(1) red-noise
> envelope and a band-wise Monte-Carlo p-value (`spectral.test_period`), plus the bootstrap
> Sharpe CI on the trade; (3) critique magnitude — period stability across rolling windows,
> the single-period-search selection problem; (4) alpha vs beta — the cycle trade's Sharpe
> against buy-and-hold and exposure; (5) execution — the random-phase null and a decade-by-
> decade skill split; (6) verdict. Engine: `data`, `spectral`, `cycles`, `backtest`,
> `robustness`.

## 4 · The Teardown

> *We ran it over the cached VIX (1990–2026, log scale). Headline numbers reproduce with
> [`examples/verify_real.py`](examples/verify_real.py); full tables in
> [`docs/results.md`](docs/results.md). The detector is validated offline first with
> [`examples/run_synthetic_demo.py`](examples/run_synthetic_demo.py).*

- **The detector works — that's not the problem.** On the synthetic log-VIX (a true 80d & 40d
  cycle hidden in red noise), the periodogram clears the 99% envelope by **53–70×** at the
  injected periods, the rolling period holds to **σ = 3 sessions**, and the walk-forward
  forecast hits **89%** (null 50%). So a flat result on the real VIX is a statement about the
  *market*, not a bug in the code.
- **The claimed cycles don't clear red noise.** At the tweet's own periods, the VIX's
  periodogram peak is *weaker* than what AR(1) noise routinely produces: p = **0.998** (40d),
  **0.9995** (80d), **0.994** (100d/20-week), **0.9995** (250d), **0.765** (1000d/4-year).
  Nothing at the claimed clocks clears even the 95% envelope; the only band that pokes through
  at 99% is a **~15.6-session** wiggle (ratio 1.10) — not a cycle anyone is trading.
- **The period won't hold still.** Across rolling 4-year windows the dominant in-band period
  wanders from **83 to 333 sessions** (mean 227, σ **82**). That is the curve-fit signature in
  one number: the "cycle" you'd have drawn in 1998 is a different length from the one you'd
  draw in 2018, because it was never fixed.
- **The forecast is a coin flip.** Walking forward, the projected cycle calls the sign of the
  next move at **49.2% / 49.5% / 50.9%** (10/20/40-session horizons) — at or below 50%, with
  null-beating p-values of **0.76 / 0.74 / 0.078**. The 40-day horizon is the best of a bad
  lot and still only 50.9%, p = 0.078.
- **The trade is just diluted beta.** Long the S&P when the VIX cycle is projected to fall
  earns **Sharpe 0.33** at **59% exposure** — almost exactly buy-and-hold's **0.56** scaled
  by exposure, and **below the random-phase null's mean of 0.37** (p = **0.74**). The cycle
  adds *nothing* over being mechanically long part-time; its bootstrap Sharpe CI **[−0.00,
  0.67]** clears zero only because the market went up, not because the timing worked.
- **It never works in any decade.** Split by era, the direction skill is **0.50 / 0.48 / 0.52
  / 0.55** (1990s/2000s/2010s/2020s) — the best, 2020–2027, is 54.5% at p = 0.070, i.e. not
  significant and not stable across the other thirty years.

> 🔬 **For the quants** — periodogram on linearly-detrended, 10%-Tukey-tapered log-VIX; the
> AR(1) null vectorised across 2,000 surrogates of identical length and pre-processing
> (`spectral._surrogate_powers`), so data and null are comparable bin-for-bin; band-wise p =
> (#{surrogate in-band peak ≥ data} + 1)/(n_sim + 1). Walk-forward skill refits the dominant
> in-band period + an OLS cosine on the past only (`cycles.fit_sinusoid`), scoring sign of the
> projected `horizon`-ahead change vs a phase-rotation null (period & amplitude held). Trade
> positions are causal (refit every 5 sessions on the past), one-session execution lag, 1 bp
> cost per switch; null rotates the same fitted phase. All from
> [`docs/results.md`](docs/results.md), as-of 2026-06-01, fingerprint `6c2029b57135`.

<details>
<summary>🔬 The maths, in full</summary>

**The red-noise null.** Fit an AR(1) to the (detrended) log-VIX: ρ = lag-1 autocorrelation,
σ² = sample variance. Its theoretical power spectrum is
`P(f) = σ²(1−ρ²) / (1 − 2ρ·cos(2πf) + ρ²)` — monotone decreasing in `f`, i.e. all the power at
low frequencies (long periods). Rather than rely on the analytic form, we Monte-Carlo: draw
`x_t = ρ·x_{t−1} + ε_t`, `ε_t ~ N(0, σ²(1−ρ²))`, run each surrogate through the *identical*
detrend → taper → periodogram, and take the per-frequency 95/99% quantiles. A data peak above
the curve is significant against red noise *at that frequency*; the band-wise test takes the
max over a ±15% band around the target period (and charges for that search by using the band
max of each surrogate too).

**Why ρ ≈ 0.98 matters.** At that persistence, the red-noise spectrum is steep: enormous
low-frequency power, so 100–1000-session "peaks" are *expected* under the null. This is exactly
why the 4-year peak (power 8.6, the tallest in absolute terms) is still **insignificant**
(p = 0.76) — red noise makes peaks that big there routinely. Magnitude in the periodogram is
meaningless without the envelope.

**The walk-forward forecast.** At origin t (every 10 sessions, after ≥750 of history), fit
`x_s ≈ c₀ + A·cos(ωs) + B·sin(ωs)`, ω = 2π/P̂, P̂ the dominant in-band period on `x[:t]`. The
forecast direction is `sign(model(t+h) − model(t))`; the realized direction is
`sign(x_{t+h} − x_t)`. Skill = hit rate. The null rotates the phase, `A,B → A cos θ + B sin θ,
…`, θ ~ U(0,2π), keeping amplitude `√(A²+B²)` and period P̂ — so it tests whether the *learned
timing* beats an arbitrary one, not whether an oscillation exists.

**The trade.** Position is causal long/flat: 1 when the projected cycle slope over `h` is
negative (vol set to fall ⇒ risk-on), refit every 5 sessions, acted on the next session's
return, 1 bp per switch. Reported against buy-and-hold and the random-phase null's Sharpe
distribution. With no timing edge, the strategy degenerates to "long the market ~59% of the
time", so its Sharpe ≈ exposure × buy-hold Sharpe — which is exactly what we see.

</details>

## 5 · The Verdict

> *The stamps, now earned.*

- **Signal — `NONE`.** No claimed cycle clears the red-noise envelope (p ≥ 0.99 at 40/80/100/
  250-day, 0.76 at the 4-year), and the only band that pokes through at 99% is a ~15.6-session
  wiggle no one is trading. My going-in prior was `WEAK` — that the VIX's real mean-reversion
  would masquerade as a *weak* periodicity; instead the periodicity is simply absent once you
  hold it to the noise floor. The one flicker — hard-wiring exactly 80 days gives a 53%
  direction tick at p = 0.04 — is a single uncorrected period-search that the *spectral* test
  flatly contradicts (80d power is inside the envelope) and that dies before becoming a
  significant trade (see beat 7). Fragile to method and selection: not a clock.
- **Tradability — `MIRAGE`.** The walk-forward forecast is a coin flip, and the tradeable
  expression underperforms buy-and-hold (Sharpe 0.33 vs 0.56) *and* its own random-phase null
  (mean 0.37, p = 0.74) — it's diluted beta, not timing. There is no execution skill, venue,
  or sizing that rescues a signal that isn't there.
- **A fixed clock — `NOT SUPPORTED`.** The dominant period ranges over 83–333 sessions (σ 82)
  window to window. Whatever a cycle-drawing tool is tracking, it isn't a constant period — it
  is being continuously re-fit to the most recent wiggle, which is precisely how a method with
  no out-of-sample content stays alive in-sample.

> 🔬 **For the quants** — decisive numbers in one place: red-noise p-values {40d 0.998, 80d
> 0.9995, 100d 0.994, 250d 0.9995, 1000d 0.765}; significant-at-99% peaks = {15.6 sessions,
> ratio 1.10} only; period stability {mean 227, σ 82, range 83–333}; direction skill {10d
> 0.492 p0.76, 20d 0.495 p0.74, 40d 0.509 p0.078}; trade {Sharpe 0.33, exposure 0.59, buy-hold
> 0.56, null-mean 0.37, p 0.74, maxDD −0.34, bootstrap CI [−0.00, 0.67]}. As-of 2026-06-01,
> fingerprint `6c2029b57135`.

## 6 · Could You Trade It?

> *The honest money question — the beat that separates this desk from a chart-reading blog.*

You wouldn't, for the rare-but-clean reason that **there is nothing to execute.** You can't
even buy the thing the cycle is drawn on — spot VIX isn't investable — so the only honest
expression is timing *something else* (here, the S&P) off the projected vol cycle. And that
forecast doesn't beat a coin, or a scrambled-phase version of itself, before a single cost.
The strategy's positive Sharpe is an illusion of being part-time long a market that rose; strip
the beta (it's 59% exposure earning 0.59× the index Sharpe) and the timing contributes nothing.

The deeper reason it can't be traded is the one the whole study is about: **a forecast needs a
fixed phase, and there isn't one.** The "cycle" you'd act on is re-fit every few months to a
period that ranges from four to sixteen months — so by the time you've committed to "the low is
due in early July," the model that said so has already been replaced by one that says something
else. That isn't a timing system; it's a rolling rationalisation. The one regime where vol
genuinely *is* forecastable — the sharp mean-reversion after a spike — is Study 03's territory
(a *level* trigger, the variance risk premium), and even there it doesn't beat buy-and-hold
once you charge it honestly. A fixed *clock* adds nothing on top.

> 🔬 **For the quants** — break-even cost is moot: the gross forecast has no edge (skill ≈ 0.50,
> p ≈ 0.74 on the trade vs null). The decade split rules out a hidden regime where it pays
> (best era 2020–2027 at 54.5%, p 0.070, not significant). The capacity question never arises —
> spot VIX is non-tradeable and the index expression is pure beta. The conditions under which a
> cycle trade *would* pay are stated precisely by the synthetic control: a period stable to a
> few sessions and a spectral peak well above the red-noise floor — neither of which the VIX
> exhibits.

## 7 · Going Further

> **We didn't leave the obvious rescues as homework.** Killing the naive cycle invites the
> immediate "yes, but did you try…?" — so we ran the three standard ones on the real VIX/SPX.
> None drags it over the line. Full table in [`docs/extensions.md`](docs/extensions.md);
> reproduce with [`examples/verify_extensions.py`](examples/verify_extensions.py).

| Rescue (real VIX, 20-session horizon) | Dir. skill | skill p | Trade Sharpe | trade p | Verdict |
|---|---|---|---|---|---|
| **Baseline** (adaptive re-tune) | 0.495 | 0.74 | 0.33 | 0.74 | the result above |
| **Fixed 80d** (tweet's VIX clock) | **0.533** | **0.038** | 0.64 | 0.13 | a flicker — but fragile (see below) |
| **Fixed 40d** (tweet's VIX clock) | 0.497 | 0.52 | 0.37 | 0.49 | a coin flip |
| **Amplitude-gated** (amp ≥ 0.5σ) | 0.495 | 0.74 | — | — | never fires: no strong cycle ever present |

- **The one number that flickers — and why it's not a clock.** Hard-wiring *exactly* 80 days
  (instead of letting the search re-tune) gives a marginal **53.3% direction skill, p ≈ 0.04**
  — the only near-significant result in the study. It does not survive: it's a **single
  hand-picked period** with no correction for the several we tried; it **contradicts the
  spectral test** (80-day power is *inside* the red-noise envelope, p ≈ 0.9995 — there is no
  peak there to forecast from); and it **doesn't become a significant trade** (Sharpe 0.64 but
  p ≈ 0.13 vs null, barely over buy-and-hold). Textbook *significant-raw, fragile-to-selection*
  — a `WEAK` fragment, honestly flagged, not a rescue.
- **The amplitude gate never even fires.** Requiring a *visibly strong* in-sample cycle (amp ≥
  0.5σ of the level) zeroes the position every single day across 36 years — the cyclical
  component is always small next to the VIX's own variance. The "clear cycle" the eye sees is
  never actually there to trade.
- **Is the cycle in *stocks* instead?** The tweet ties the VIX clock to a stock 20-week and
  4-year cycle. Tested directly on the S&P, those clocks don't clear red noise either: p =
  **1.00** (100d), **1.00** (250d), **0.9995** (1000d).

What's left genuinely open — framed as worked leads, not excuses:

- **A transient (time-localised) cycle.** The global periodogram averages over 36 years; a
  *wavelet* (time-frequency) analysis could chase a cycle that is real but only for a few years
  at a time. The catch the desk would insist on: a cycle that exists only in hindsight windows
  is unforecastable by construction — the wavelet must be paired with the same walk-forward
  test, not a chart.
- **The VIX-futures term structure.** Spot VIX is non-tradeable; the curve (contango/
  backwardation) is. A genuine roll-yield seasonality there would be a *different*, testable
  claim — and the honest place to look for a tradeable vol cycle if one exists.
- **Other levels.** The same machine runs on any series: point `data.vix_series` at MOVE,
  credit spreads, or a single name's realized vol and re-run `verify_real.py`. The detector is
  validated; the question is whether *anything* in markets carries a fixed clock the VIX lacks.

The deep version — the synthetic validation, the periodogram-vs-envelope figure, the period-
stability drift, the walk-forward and the trade — is in
[`notebooks/02_for_the_quants.ipynb`](notebooks/).

---

## How this study is laid out

| Path | What's inside |
|---|---|
| [`notebooks/01_for_the_curious.ipynb`](notebooks/) | the story + the stakes, plain language |
| [`notebooks/02_for_the_quants.ipynb`](notebooks/) | the full method: periodogram, red-noise envelope, stability, walk-forward, trade |
| [`docs/results.md`](docs/results.md) | **the real run** — every headline table, fingerprinted and as-of'd |
| [`docs/extensions.md`](docs/extensions.md) | **the beat-7 rescues worked** — fixed periods, amplitude gate, the stock clocks |
| [`docs/references.md`](docs/) | sources + literature map (Hurst cycles; Mann–Lees, Torrence–Compo on red-noise significance; VIX dynamics) |
| [`vix_cycles/`](vix_cycles/) | the study package: `data` · `spectral` · `cycles` · `backtest` · `robustness` |
| [`examples/`](examples/) | [`run_synthetic_demo.py`](examples/run_synthetic_demo.py) (offline) · [`verify_real.py`](examples/verify_real.py) (the real run) · [`verify_extensions.py`](examples/verify_extensions.py) (the rescues) |

Every number is produced by [`vix_cycles/`](vix_cycles/), in the house style of the shared
[`../../quantlab/`](../../quantlab/) engine; `pytest` covers it in CI.
