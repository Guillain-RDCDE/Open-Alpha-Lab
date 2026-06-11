# Results — Study 36 (Greenback): dollar-carry & the carry⊕momentum combo

> **Real G10 run · offline from cache · as-of 2024-01-31 · fingerprint `ef7450ae792e`.**
> OECD 3-month interbank short rates (% p.a., monthly) + yfinance FX (USD per 1 unit of nine foreign
> currencies, resampled to month-end), a **USD-funded monthly book** over **2001-08 → 2024-01** (270
> months; the strategy sleeves report 264 months after the 12-month momentum warm-up). The as-of is pinned
> to the rates' **2024-01** end — OECD's MEI series was discontinued there — so the headline numbers don't
> creep with the calendar. Reproduce: `python examples/verify.py` (no network); the offline synthetic
> machinery proof is `python examples/run_synthetic_demo.py`.

## The verdict — Signal `WEAK` · Tradability `FRAGILE` · Combo diversifies the crash? `PARTIAL`

The FX carry premium — high-short-rate currencies out-earn low-rate ones, because uncovered interest-rate
parity fails — is one of the most durable anomalies in macro (Lustig–Roussanov–Verdelhan 2011;
Menkhoff–Sarno–Schmeling–Schrimpf 2012). It is also the textbook *"picking up nickels in front of a
steamroller"*: sharply negative-skewed, dripping steady gains then losing a fortune in a global risk-off.
**Study 27 (Steamroller) already measured the carry premium itself, and that vol-targeting can't dodge its
crash.** Greenback is the *next* question — the believer's actual trade: combine carry with its supposed
complement, **momentum**, into the classic **carry⊕momentum combo** (Asness–Moskowitz–Pedersen 2013;
Koijen–Moskowitz–Pedersen–Vrugt "Carry" 2018), and add the **dollar-carry** tilt (long/short the USD basket
by the average rate gap; LRV's dollar factor).

On the real 2001–2024 G10 tape the verdict splits honestly:

- **Carry pays, but this sample alone can't certify it** — Sharpe **+0.22**, +1.8%/yr, with the textbook
  negative-skew crash (**skew −0.70**, worst months **Oct-2008 −10.6%** and **Mar-2020 −9.1%**). The bucket
  spread is **+3.0%/yr**, but the Sharpe is only **Lo *t* ≈ 1.0** and its bootstrap 95% CI is
  **[−0.17, +0.69]** (14% of resamples negative) — below the desk's *t* ≥ 2 robust-inference bar, so the
  Signal stamp is **`WEAK`**: it leans on three decades of literature for carry's existence, while this
  23-year sample alone reads weak — exactly as the literature finds carry decayed post-2000. (The bucket
  table itself is a same-month *descriptive* sort, unlagged; the tradable sleeves lag their weights one
  month.)
- **FX momentum *failed* this sample** — Sharpe **−0.14**, −1.1%/yr. Cross-sectional FX momentum, strong in
  the 1980s–90s, eroded to roughly zero-to-negative post-2008 (Menkhoff et al; the well-documented FX-mom
  decay). On this tape it lost money.
- **The combo can't beat carry — but it still diversifies the crash.** Combo Sharpe **+0.06** is *below*
  carry's +0.22, because a losing momentum leg drags the blend (`combo beats best leg: False`). Yet the
  diversification mechanics fire as advertised: the legs are **decorrelated (+0.05)**, so the combo's
  **skew improves (−0.57 vs −0.70)**, its **worst month halves (−6.4% vs −10.6%)**, and in carry's worst 5
  months the combo lost only **−3.0% vs carry's −7.6%**. The cushion is real; the Sharpe uplift is not —
  *this sample's* momentum simply had no edge to lend. Hence **`PARTIAL`**.

So: **Signal `WEAK`** (carry pays here, with its crash, but at *t* ≈ 1.0 and a CI spanning zero this
sample can't certify it — the literature carries the existence case), **Tradability `FRAGILE`** (a thin,
crash-prone, cost-sensitive premium), and the combo thesis is **`PARTIAL`** on the real tape — it cushions
the steamroller exactly as designed, but cannot lift the Sharpe while its momentum leg is itself losing.

## The real G10 books (net @10 bp, vol-scaled to 8%, USD-funded monthly, 2001–2024)

| sleeve | Sharpe | ann. return | vol | skew | max-DD |
|---|---|---|---|---|---|
| carry (cross-sectional) | **+0.22** | +1.8% | 8.3% | **−0.70** | −27% |
| dollar-carry tilt (LRV) | +0.17 | +1.5% | 8.9% | −0.44 | −45% |
| momentum (12-month trend, no skip month) | **−0.14** | −1.1% | 8.0% | +0.39 | −47% |
| **carry⊕momentum combo** | **+0.06** | +0.4% | 5.9% | −0.57 | −26% |

- **Carry premium:** high-minus-low rate bucket spread **+3.0%/yr**; carry book Sharpe **+0.22**, Lo *t* ≈
  **1.0**, bootstrap 95% CI **[−0.17, +0.69]**, 14% of resamples negative — statistically `WEAK` over this
  sample; the existence case is the literature's.
- **The steamroller is on the real tape:** carry skew **−0.70**, worst month **−10.6% (Oct 2008)**, next
  **−9.1% (Mar 2020)** then **−7.0% (Sep 2008)** — the GFC and COVID risk-offs, the crash the literature
  warns of.
- **Momentum decayed:** the 12-month-trend FX-momentum sleeve (trailing 12-month return with **no skip
  month** — a 12-0 signal, not the academic 12-1 convention) earned **−0.14** Sharpe on 2001–2024 — the
  well-documented erosion of cross-sectional FX momentum after the mid-2000s.
- **The combo diversifies but doesn't out-Sharpe:** combo **+0.06** < carry **+0.22** (`combo_beats_legs =
  False`) because momentum lost; but leg correlation is **+0.05**, the combo's worst month is **−6.4%** (vs
  carry's −10.6%) and in carry's worst 5 months it lost **−3.0%** vs carry's **−7.6%** — the crash *is*
  cushioned, the Sharpe uplift is not.
- **The dollar-carry tilt is a separate, weak stream:** Sharpe **+0.17**, −45% DD — the LRV dollar factor
  is present but, like static carry, thin and crash-exposed on this sample. (One honest accounting note:
  this sleeve's cost term divides the basket flip's turnover across the nine legs, under-charging it by
  roughly the basket width — the +0.17 is, if anything, overstated. It is not a headline number.)
- **Cost behaviour (the FRAGILE call):** carry turns over slowly (**0.55×/yr**) but its **break-even is only
  ≈13 bp** — a thin gross edge; momentum turns over **4.4×/yr** and loses gross (break-even 0). The combo's
  net Sharpe decays from **+0.17 (0 bp)** to **+0.06 (10 bp)** to **−0.10 (25 bp)** — cost bites fast.
  Tradability is `FRAGILE` on both counts: the crash *and* the thin, cost-sensitive edge.

## How this differs from Study 27 (Steamroller)

Steamroller measured the **G10 carry premium itself** and showed a **vol-targeting overlay cannot dodge the
carry crash**. Greenback takes that as given and tests the believer's *fix*: not an overlay but
**diversification** — the **carry⊕momentum combo** plus the **dollar-carry tilt**. The real-tape finding
sharpens the Steamroller story: the *mechanical* cushion is genuine (decorrelated legs ⇒ shallower crash,
−10.6% → −6.4% worst month, −7.6% → −3.0% in carry's worst 5), but on 2001–2024 **momentum had no edge to
lend**, so the combo cushions the crash *without* lifting the Sharpe. The honest fix dulls the steamroller;
it does not turn carry into a free lunch.

## What the offline synthetic control proves (machinery)

The seeded synthetic panel (9 currencies × 600 months, a baked carry premium + sticky risk-off crashes + an
*independent profitable* trend) is the controlled proof that the books recover what's there: carry Sharpe
**+1.18** on the control vs a flat **+0.8%/yr** full-UIRP null, carry skew **−1.55**, and — *with a winning
momentum leg by construction* — the combo Sharpe **+1.69** beats either leg at correlation **+0.26**. The
control demonstrates the combo *would* out-Sharpe when momentum has an edge; the real tape shows that on
2001–2024 FX momentum did not, which is precisely why the real combo verdict is `PARTIAL`, not the control's
unconditional win. Reproduce: `python examples/run_synthetic_demo.py`.

*Sources & literature map: [docs/references.md](references.md); the carry⊕momentum writeup is in
[docs/extension.md](extension.md). Engine: [`quantlab/`](../../../quantlab/).*
