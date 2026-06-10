# Results — Study 35 (Contango): commodity carry / roll yield

> ⚠️ **Real run pending a term-structure fetch.** Computing a commodity's **roll yield** needs its
> **term structure** — at least the front and first-deferred contract price, every week (the *slope* of
> the curve). The desk's cache holds only the **front-month continuous** returns
> (`_cache/commodity_futures_weekly.parquet`, 12 commodities), and no free source in this environment
> reliably serves the individual deferred contracts (yfinance returns a single continuous front-month
> series; DoltHub / OptionsDX do not carry futures curves). So this study's real tape is **not yet
> available** — exactly the position [Study 27 (Steamroller)](../../27-steamroller/) was in before its FRED
> download. When a curve source is wired in, run it and this file is overwritten with the fingerprinted
> numbers:
>
> ```
> python examples/verify.py --fetch     # reads a populated commodity_term_structure.parquet, runs, writes this file
> ```
>
> Until then, the verdict below is earned on the **synthetic control** (a commodity panel with a *known*
> roll-yield premium and a disconnected null) and the long-run academic literature; the offline core is
> fully validated and reproducible via [`examples/run_synthetic_demo.py`](../examples/run_synthetic_demo.py).
> As-of pin for the eventual real run: **2026-06-10**.

## The verdict — Signal `REAL` · Tradability `FRAGILE` · Real-tape run? `PRE-REG`

The commodity carry premium — long the most-**backwardated** futures (positive roll yield: the curve
rolls *up* toward expiry), short the most-**contangoed** (negative roll yield) — is one of the most durable
documented anomalies in commodities (Gorton–Rouwenhorst 2006; Erb–Harvey 2006; Koijen–Moskowitz–Pedersen–
Vrugt 2018). It is a slow signal, so it is comparatively **cheap to run** (low turnover); its real risk is
not transaction cost but a **crash-prone, volatile** return stream that, like FX carry (Study 27), unwinds
hard in commodity-wide risk-off episodes. The honest framing: the **signal** is real (literature + control),
**tradability** is fragile (volatile, crash-prone, capacity-limited to liquid contracts), and the
**real-tape run is pre-registered** — its apparatus, mirage line, and expected shape are all fixed here,
awaiting only the term-structure data.

## What the synthetic control proves (offline, reproducible)

On a synthetic 12-commodity weekly panel with a baked roll-yield premium (seed 35, 20 years,
`carry_strength=0.9`), fingerprint `b502aaa6304f`:

- **The premium is real and recovered:** high-minus-low roll-yield bucket spread **+27.6%/yr**
  (top **+21.4%**, bottom **−6.2%**); the dollar-neutral carry book earns a gross Sharpe **+1.86**
  (CAGR **+16.5%**), net @5 bp **+1.80** — costs barely dent it.
- **The book is cheap to run:** weekly turnover **0.19**, break-even cost **~160 bp** — costs are *not* the
  binding constraint, so the tradability question is the crash tail, not the spread (the deliberate
  contrast with [Study 33 (Slingshot)](../../33-slingshot/), where daily turnover buried the edge).
- **The null collapses:** with the roll-yield signal **disconnected** from returns (`carry_strength=0`),
  the same book earns Sharpe **−0.28** and a bucket spread of just **−4.8%/yr** — proving the apparatus
  measures the effect, not itself.
- **Carry diversifies with momentum:** a 50/50 blend with a commodity time-series-momentum sleeve lifts the
  Sharpe **above either standalone leg** (carry **1.80**, momentum **1.43** → blend **2.03**) at a low
  leg-to-leg correlation **+0.27** — see [docs/extension.md](extension.md).

## What `--fetch` will fill in (pre-registered)

The real run, once a term-structure source is wired in, will report on the actual commodity curves: the
roll-yield premium and its Newey–West *t*, the realized Sharpe and turnover, the **skew / worst week /
drawdown** of the actual carry crashes (e.g. the 2008 and 2014–15 commodity routs), the break-even cost
on liquid contracts, and the carry-vs-momentum blend — each fingerprinted and as-of pinned. The expected
shape, from the literature, is a standalone Sharpe of roughly **0.5–0.8** with deep, volatile drawdowns —
i.e. exactly the `REAL` / `FRAGILE` verdict the synthetic control already earns. **Mirage line** (fixed
now, so we can't move it later): if the real backwardated-minus-contangoed spread is statistically
indistinguishable from zero (HAC *t* < 2), or if the only contracts liquid enough to trade carry no
premium, the signal drops to `WEAK`/`NONE`.

*Sources & literature map: [docs/references.md](references.md). Engine: [`quantlab/`](../../../quantlab/).*
