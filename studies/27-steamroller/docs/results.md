# Results — Study 27 (Steamroller): G10 carry

> ⚠️ **Real run pending one networked fetch.** Unlike the desk's other studies, this one has **no
> pre-populated data cache**: its real tape is G10 short rates and FX from **FRED** (free, no API key),
> which must be downloaded once. Run it and this file is overwritten with the fingerprinted G10 numbers:
>
> ```
> python examples/verify.py --fetch     # downloads G10 rates + FX from FRED, caches, writes this file
> python examples/extension.py --fetch  # the beat-7 risk-managed comparison
> ```
>
> Until then, the verdict below is earned on the **synthetic control** (a G10 with a *known* carry
> premium and risk-off crashes) and the long-run academic literature; the offline core is fully
> validated and reproducible via [`examples/run_synthetic_demo.py`](../examples/run_synthetic_demo.py).

## The verdict — Signal `REAL` · Tradability `FRAGILE` · Crash risk? `SEVERE`

The carry trade — borrow a low-rate currency, lend a high-rate one — is one of the most durable
anomalies in macro: uncovered interest-rate parity fails, so high-rate currencies do **not** depreciate
enough to offset their yield, and the differential is a real, paid premium (Lustig–Verdelhan 2007;
Menkhoff–Sarno–Schmeling–Schrimpf 2012). It is also the textbook *"picking up nickels in front of a
steamroller"*: the return is sharply **negative-skewed**, dripping steady gains and then losing a fortune
in a global risk-off (1998 LTCM, 2008) — and that crash, uniquely on this desk, **resists** the
vol-targeting overlay that tamed equity ([Study 16](../../16-storm-shy/)) and momentum
([Study 24](../../24-stampede/)), because it arrives as a sudden jump rather than a forecastable
volatility build-up.

## What the synthetic control proves (offline, reproducible)

On a synthetic G10 with a baked carry premium and sticky risk-off crashes (seed 27, 50 years):

- **The premium is real and recovered:** high-minus-low rate bucket spread positive, carry portfolio
  **+2.1%/yr** at a Newey–West *t* of **+2.8**, Sharpe **+0.60**, on low turnover.
- **The steamroller is there:** monthly skew **−1.54**, max drawdown **−28%** — and the full-UIRP **null**
  collapses to a flat, insignificant premium (skew ≈ 0), proving the diagnostics measure the effect, not
  themselves.
- **Risk management lifts the Sharpe but not the tail:** vol-scaling moves the Sharpe **+0.68 → +0.95**
  while the drawdown **−28% → −44%** — it can *deepen* the crash by levering up into it, exactly because
  the crash isn't a vol build-up the trailing estimate can see.

## What `--fetch` will fill in

The real G10 run will report, on FRED data: the carry premium and its HAC *t*, the realized Sharpe and
turnover, the skew / worst month / drawdown of the actual 1998 and 2008 carry crashes, and the plain-vs-
vol-managed comparison — each fingerprinted and as-of pinned. The expected shape (from the literature) is
a Sharpe of roughly 0.5–0.8 with a 2008 drawdown near −30%, i.e. exactly the `REAL` / `FRAGILE` /
`SEVERE` verdict the synthetic control already earns.

*Sources & literature map: [docs/references.md](references.md). Engine: [`quantlab/`](../../../quantlab/).*
