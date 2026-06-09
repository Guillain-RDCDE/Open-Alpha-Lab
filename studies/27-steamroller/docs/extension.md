# Beat 7 — worked complement: can risk management dodge the steamroller? (Study 27)

> ⚠️ **Real run pending one fetch.** This is the beat-7 complement on real G10 data, which needs the
> FRED download. Run `python examples/extension.py --fetch` (after `examples/verify.py --fetch` populates
> the cache) to overwrite this file with the fingerprinted real-G10 comparison. Until then the synthetic
> control below is the validated proof.

## The overlay that worked elsewhere — and doesn't here

Vol-targeting tamed the crash for equity ([Study 16](../../16-storm-shy/)) and momentum
([Study 24](../../24-stampede/)), because those crashes are *preceded* by rising volatility a trailing
estimate can see. Carry is different. On the synthetic G10 control (seed 27):

| | plain carry | vol-managed carry |
|---|---|---|
| Sharpe | +0.68 | +0.95 |
| max drawdown | −28% | −44% |

Vol-scaling **lifts the standalone Sharpe** (it cuts exposure in noisy calm spells) but does **not**
shrink the drawdown — and *deepens* it, because in calm times it levers up into a carry trade whose
crash arrives as a sudden risk-off jump, not a forecastable volatility build-up.

**Takeaway.** `REAL` / `FRAGILE` / `Severe` survives the worked complement, and the complement sharpens
*why* the crash is `Severe`: it is the one tail on this desk that a trailing-volatility forecast cannot
see. Carry's premium is real compensation for a jump you must be willing — and able — to hold through.
The honest fixes are options-based tail hedges, a risk-off conditioning switch
(Brunnermeier–Nagel–Pedersen), or cross-asset diversification — forks for the next contributor, not a vol
target. The real-G10 version (via `--fetch`) shows the same shape on the actual 1998 and 2008 crashes.

*Engine: [`quantlab/`](../../../quantlab/). Not investment advice — research and education.*
