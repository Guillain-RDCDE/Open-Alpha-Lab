# Results — Study 34 (Aftershock): post-earnings-announcement drift (PEAD)

> ⚠️ **Real-tape run pre-registered and PENDING an earnings-history fetch (as-of 2026-06-10).** Unlike
> the desk's cached studies, this one has **no pre-populated data cache and no wired long-history
> source.** A credible PEAD backtest needs *years* of reported-earnings dates + surprises per name; no
> free feed supplies that in this sandbox (yfinance exposes only ~6-8 reported quarters, and there is no
> reliable long free surprise history — the same wall the desk hit for options open-interest). So the
> real measurement is **pre-registered**: the apparatus, the protocol and the mirage-line are fixed
> below, and the moment an earnings-history feed is wired the run is one command —
>
> ```
> python examples/verify.py --fetch     # pulls earnings dates + surprises + prices, caches, writes this file
> ```
>
> — and this file is overwritten with the fingerprinted, as-of'd real numbers. Until then, the verdict
> rests on the **synthetic control** (a stock panel with a *known* surprise→drift relationship) and the
> decades-deep academic literature; the offline core is fully validated and reproducible via
> [`examples/run_synthetic_demo.py`](../examples/run_synthetic_demo.py).

## The verdict — Signal `REAL` · Tradability `FRAGILE` · Real-tape run? `PRE-REG`

Post-earnings-announcement drift is one of the most **robustly documented anomalies in finance**: a
stock's price keeps drifting in the direction of its earnings *surprise* for weeks after the
announcement (Ball-Brown 1968; Bernard-Thomas 1989, 1990). The effect is real and has survived fifty
years of out-of-sample scrutiny — so **Signal `REAL`**. But its *tradability* is the catch the literature
has been just as clear about: the drift is small, and it **concentrates in the illiquid, small, high-cost
names** where it is most expensive to harvest (Chordia et al. 2009), while shrinking toward zero in the
liquid stocks you can trade at scale — and the premium has attenuated since publication as it was
arbitraged. So **Tradability `FRAGILE`**, pending the real measurement.

## What the synthetic control proves (offline, reproducible)

On a synthetic stock panel where each name reports quarterly, every surprise is a standardised z-score,
and the ~60 trading days after each event carry a small drift proportional to the surprise that decays to
zero (seed 34, 120 names × 12 years, 5,760 earnings events; the **null** draws the same surprises but
gives them no drift):

- **The drift is real and recovered.** The dollar-neutral long-positive / short-negative-surprise book
  earns a gross Sharpe of **+3.73** (CAGR **+10.2%**), while the **null collapses to +0.40** (≈ noise) —
  proving the apparatus measures the surprise→drift effect, not itself.
- **It survives costs on the control, *because* turnover is low.** Turnover is only **0.068/day** (the
  book rolls on the earnings calendar, not daily), so the **break-even cost is ~57 bp** and the net
  Sharpe at a realistic 5 bp is still **+3.41**. *(On the real tape this break-even will be far lower —
  the literature's whole point is that real PEAD is small and lives in high-cost names.)*
- **The drift-decay curve has the PEAD shape.** The mean surprise-signed cumulative abnormal return rises
  steadily over the post-event window and then flattens — **+0.0004 (day 0) → +0.0059 (day 20) → +0.0116
  (day 40) → +0.0153 (day 69)** — Bernard-Thomas's Figure 1, reproduced; the null curve is flat
  (**+0.0032** at day 69).

## Data stamp

- **Synthetic control**: 120 stocks × 3,024 trading days, 5,760 events, drift_strength 0.0005 (null = 0),
  seed 34, inputs fingerprint `3316fcb4614f`
- **Real tape**: *not yet fetched* — pre-registered, pending an earnings-history source (see above)

## What `--fetch` will measure (pre-registered)

The real run will report, on genuine reported-earnings dates + SUEs + prices: the PEAD book's gross and
net Sharpe and its HAC *t*-stat; turnover and **break-even cost** (the headline tradability number); the
**cost sweep** and **holding-period sweep** (does the drift persist long enough to pay?); and the
**drift-decay curve** on real events. The **mirage-line is pre-registered**: if the break-even cost sits
inside the realistic equity round-trip band (~2-10 bp) — which the liquidity literature predicts,
especially in the small names where PEAD is strongest — then Tradability is confirmed `FRAGILE` (or worse
for the scalable, liquid slice). The expected shape, from the literature, is a real but modest gross edge
that the costs and illiquidity of its strongest names erode to a thin residual — exactly the `REAL` /
`FRAGILE` verdict the synthetic control and the literature already earn.

*Sources & literature map: [docs/references.md](references.md). The beat-7 drift-decay writeup is in
[docs/extension.md](extension.md). Engine: [`quantlab/`](../../../quantlab/).*
