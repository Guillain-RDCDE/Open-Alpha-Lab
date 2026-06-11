# Beat-7 worked complement — the carry⊕momentum diversification (Study 36)

> ✅ **Real run done — offline from cache, as-of 2024-01-31, fingerprint `ef7450ae792e`.** The synthetic
> control (below) is the controlled machinery proof; the **real G10 tape (2001–2024)** result follows it.
> The headline: on the real tape the diversification *mechanics* fire (decorrelated legs cushion the crash)
> but the Sharpe uplift does **not**, because FX momentum decayed to negative this sample — so the combo
> thesis is `PARTIAL`, not the control's unconditional win.

## The idea — two premia that pay at different times

Carry is a real, paid premium with a brutal negative-skew crash (the steamroller — [Study
27](../../27-steamroller/) showed vol-targeting can't dodge it). Momentum is a *separate* currency premium:
rank currencies by their trailing trend, long the up-trenders, short the down-trenders (Menkhoff et al.
2012b). The classic result (Asness–Moskowitz–Pedersen 2013; Koijen et al. "Carry" 2018) is that carry and
momentum are **lowly- or negatively-correlated** — so blending them into an equal-risk **carry⊕momentum
combo** raises the Sharpe above either standalone, *and* momentum tends to ride the trend out of a carry
crash, cushioning the drawdown carry alone can't escape.

## The result on the synthetic control (seed 36, 9 currencies × 600 months, net @10 bp)

| | carry (vol-scaled) | momentum | **carry⊕momentum combo** |
|---|---|---|---|
| Sharpe | +1.18 | +1.53 | **+1.69** |
| monthly skew | −1.55 | −0.36 | −1.62 |
| max drawdown | **−60%** | −18% | **−20%** |

- **The combo beats either leg:** Sharpe **+1.69** vs carry **+1.18** and momentum **+1.53** — a genuine
  uplift, not just an average, because the two legs' returns correlate only **+0.26**.
- **Momentum cushions the steamroller:** the carry book draws down **−60%**; the combo only **−20%**. In
  carry's *worst 5 months* the combo lost just **−9.3%** versus carry's **−11.0%** — momentum is leaning
  the other way exactly when carry is being run over.
- **The crash doesn't vanish.** The combo is still negatively skewed (−1.62): diversification *dulls* the
  steamroller, it doesn't remove it. That is the honest reason tradability stays `FRAGILE`, not
  `INVESTABLE` — you still need to be willing and able to hold a crash-prone book through its worst months.

## The result on the real G10 tape (OECD short rates + yfinance FX, 2001–2024, net @10 bp)

| | carry (cross-sectional) | momentum (12-1m) | **carry⊕momentum combo** |
|---|---|---|---|
| Sharpe | **+0.22** | **−0.14** | +0.06 |
| monthly skew | −0.70 | +0.39 | −0.57 |
| worst month | −10.6% (Oct-2008) | — | −6.4% |
| max drawdown | −27% | −47% | −26% |

- **The legs decorrelate exactly as the literature says** — carry↔momentum correlation **+0.05** on the
  real tape — so the diversification *mechanics* are intact: the combo's **worst month halves** (−10.6% →
  −6.4%), its **skew improves** (−0.70 → −0.57), and in carry's **worst 5 months the combo lost just −3.0%
  vs carry's −7.6%**. Momentum *is* leaning the other way in the GFC/COVID risk-offs.
- **But the Sharpe uplift fails** — combo **+0.06** is *below* carry **+0.22**, because **FX momentum lost
  money** over 2001–2024 (Sharpe **−0.14**; the well-documented post-2000s erosion of cross-sectional FX
  momentum). A losing leg can't lift a blend, however nicely it decorrelates.
- **So the combo thesis is `PARTIAL` on the real tape:** the crash cushion is real and earned, the Sharpe
  uplift is not — the diversification only pays in full when *both* legs have an edge, as the synthetic
  control (where momentum is profitable by construction) shows it can.

## Why this is the right beat-7 for Greenback (and not a repeat of Study 27)

Study 27's beat-7 asked *"can risk management (vol-targeting) dodge the carry crash?"* and answered **no**.
Greenback's beat-7 asks a different question — *"can a second, decorrelated premium (momentum) cushion
it?"* — and answers **partly yes**: the combo halves-and-then-some the drawdown and lifts the Sharpe,
without ever pretending the jump risk is gone. Diversification is the honest fix the vol-overlay couldn't
provide.

## Forks worth a PR

- **Optimal combo weight** — sweep `w_carry` (not just 50/50) and risk-parity vs the realised crisis
  correlation; does the uplift hold out-of-sample?
- **Dollar-carry as a third leg** — add the LRV dollar factor (§8.3) to the blend; on the real tape (where
  the USD rate cycles) it is a genuine separate premium the synthetic can't show.
- **Crisis-conditional tilt** — scale toward momentum when a risk-off indicator (global FX vol,
  Brunnermeier–Nagel–Pedersen) is rising, the regime where carry is most exposed.

*Engine: [`quantlab/`](../../../quantlab/). Not investment advice — research and education.*
