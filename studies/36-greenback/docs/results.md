# Results — Study 36 (Greenback): dollar-carry & the carry⊕momentum combo

> ⚠️ **Real run PENDING one networked fetch (as-of 2026-06-10).** This study's carry signal needs
> **short-term interest rates from FRED** (e.g. `DGS3MO` / the G10 3-month interbank series). FX spot from
> yfinance works in this sandbox, but the **FRED rates download times out** here — and without the rates
> there is no carry signal. So, exactly like [Study 27 (Steamroller)](../../27-steamroller/), this file is
> a transparent **pending-fetch stub**: the verdict below is earned on the fully-validated **synthetic
> control** and the long-run academic literature, and the real-tape numbers await one networked FRED fetch:
>
> ```
> python examples/verify.py --fetch     # downloads short rates (FRED) + FX spot (yfinance), caches, writes this file
> ```
>
> The offline core is fully validated and reproducible via
> [`examples/run_synthetic_demo.py`](../examples/run_synthetic_demo.py).

## The verdict — Signal `REAL` · Tradability `FRAGILE` · Real-tape run? `PRE-REG`

The FX carry premium — high-short-rate currencies out-earn low-rate ones, because uncovered interest-rate
parity fails — is one of the most durable anomalies in macro (Lustig–Roussanov–Verdelhan 2011;
Menkhoff–Sarno–Schmeling–Schrimpf 2012). It is also the textbook *"picking up nickels in front of a
steamroller"*: sharply negative-skewed, dripping steady gains then losing a fortune in a global risk-off
(1998, 2008). **Study 27 (Steamroller) already measured the carry premium itself, and that vol-targeting
can't dodge its crash.** Greenback builds on it with the *next* question — the one the believers actually
trade: combine carry with its natural complement, **momentum**, into the classic **carry⊕momentum combo**
(Asness–Moskowitz–Pedersen 2013; Koijen–Moskowitz–Pedersen–Vrugt "Carry" 2018). Because carry and momentum
pay at **different times**, the combo earns a *higher* Sharpe than either standalone and — the part that
matters for the steamroller — momentum tends to ride the trend *out of* a carry crash, cushioning the
drawdown that carry alone can't escape. That is why the combo is the standard diversifier, and why the
verdict is `REAL` signal / `FRAGILE` tradability (the crash never fully goes away).

## What the synthetic control proves (offline, reproducible)

On a synthetic currency panel with a baked carry premium, sticky risk-off crashes, and an independent
trend the momentum sleeve can ride (seed 36, 9 currencies × 600 months, net of 10 bp):

| sleeve | Sharpe | ann. return | vol | skew | max-DD |
|---|---|---|---|---|---|
| carry (vol-scaled) | **+1.18** | +10.4% | 8.9% | **−1.55** | **−60%** |
| dollar-carry tilt | +0.17 | +0.9% | 5.3% | −0.13 | −44% |
| momentum | +1.53 | +11.9% | 7.8% | −0.36 | −18% |
| **carry⊕momentum combo** | **+1.69** | +11.2% | 6.6% | −1.62 | **−20%** |

- **The carry premium is real and recovered:** high-minus-low rate bucket spread **+5.3%/yr**; the
  full-UIRP **null** collapses to a flat **+0.8%/yr** (carry Sharpe ≈ 0.19) — the diagnostics measure the
  effect, not themselves.
- **The steamroller is there:** the carry book's monthly skew is **−1.55**, worst month **−13.7%**, max
  drawdown **−60%** — the negative-skew crash the literature warns of.
- **The combo beats either leg and dulls the crash:** combo Sharpe **+1.69** vs carry **+1.18** /
  momentum **+1.53**, because the two legs barely correlate (**+0.26**). And momentum cushions the
  steamroller: the combo's max drawdown is **−20%** (vs carry's **−60%**), and in carry's worst 5 months
  the combo lost only **−9.3%** vs carry's **−11.0%**.
- **Cost behaviour:** carry rebalances slowly (its break-even is effectively unbounded on the control's
  fixed rates); momentum's break-even is **≈14.6 bp**; the combo's Sharpe decays gracefully from **+1.75
  (0 bp)** to **+1.13 (100 bp)** — tradability is `FRAGILE` because of the *crash*, not the cost.
- **The dollar-carry tilt is weak on this control by construction:** the synthetic's average rate gap is
  near-constant, so the time-series dollar signal barely moves (Sharpe +0.17). On the real tape — where
  the USD rate cycles versus the basket — Lustig–Roussanov–Verdelhan's dollar factor is a genuine,
  separate premium; that is precisely what `--fetch` will measure.

## What `--fetch` will fill in

The real G10 run will report, on FRED rates + yfinance FX: the carry premium and its Newey–West *t*, the
**dollar-carry** tilt's standalone Sharpe (the LRV dollar factor), the **momentum** sleeve, and the
**carry⊕momentum combo** — its Sharpe uplift over either leg, the leg correlation across the 1998/2008
crises, and how much momentum cushioned the actual carry crashes — each fingerprinted and as-of pinned. The
expected shape (from the literature) is a combo Sharpe meaningfully above either standalone (~0.8–1.0) with
the carry crash still present but shallower, i.e. exactly the `REAL` / `FRAGILE` verdict the synthetic
control already earns. **Real-tape run? `PRE-REG`** — announced before the fetch, so the goalposts can't move.

*Sources & literature map: [docs/references.md](references.md); the carry⊕momentum writeup is in
[docs/extension.md](extension.md). Engine: [`quantlab/`](../../../quantlab/).*
