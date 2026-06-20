# Study 307 — Coffee-Seasonality ☕

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the harvest/frost calendar real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The frost-minus-harvest spread is **+0.57%/month, t = 0.34**, with a block-bootstrap 95% CI of **[−3.47%, +4.46%]** — a coin flip. No calendar month clears even a nominal \|t\| ≥ 2 (best is November, t = 1.82, off-thesis). |
| **Tradability** — does the calendar timer add value? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The long-Jun–Aug / short-May–Sep timer earns **Sharpe 0.04** vs buy-and-hold KC=F's **0.20** (excess of T-bill, both legs); 10 bp/leg costs push it to 0.02. It gives back an already-poor commodity's whole risk-adjusted return. |
| **"Frost premium"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The supposedly bullish frost window is *negative* in 2000–2012 (−0.49% vs harvest +1.18%) and only nominally positive 2013-on (t = 1.26). A sign-flipping premium is no premium. |

> **Does coffee have a tradable harvest-season calendar?** A single Brazilian frost can double the price overnight — which is exactly why the *calendar* fails: frost is a tail event, not a date on the calendar. The seasonal means are swamped by noise and the timer wrecks you.

> **In one sentence:** the most romantic seasonal story in commodities — buy ahead of the Brazilian frost, sell into the harvest — produces a frost-minus-harvest spread of t = 0.34, no significant month, a sign-flipping sub-period premium, and a calendar timer at Sharpe 0.04 against buy-and-hold's 0.20.

## What we tested

The folklore: Arabica coffee is a Southern-Hemisphere crop and Brazil grows a third of the world's
beans, so the calendar should bite twice — the **frost-risk window** (Jun–Aug, the Brazilian winter,
when an overnight frost can wipe out a crop and spike prices) is the bullish leg, and the
**harvest-pressure window** (May/Sep, when new crop floods the market) is the bearish leg. We test it
on **every calendar month of Arabica futures (KC=F)** available (2000-02 → 2026-05, 316 months on a
verified hole-free monthly grid built from daily closes): (1) per-month one-sample t-stats, naive and
HAC; (2) a frost-vs-harvest Welch t-test; (3) a circular block-bootstrap 95% CI on the spread; (4) a
calendar timer (long frost, short harvest, T-bill otherwise) vs buy-and-hold; and (5) a 2000–2012 /
2013-on sub-period split. The offline control is a synthetic world with a tunable frost premium and a
null — it pins the machinery and can never back the Signal stamp.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a vivid, true weather story ("a frost can double the price!") makes a *terrible* calendar trade |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-month HAC t-stats, Bonferroni, frost-vs-harvest spread + bootstrap CI, timer race, sub-period split |

The fingerprinted real-data run (KC=F + ^IRX, 2000–2026, fp `89301021ccad`) is in
[docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py)
(`--fetch` to download); the offline machinery proof runs on the synthetic world in
[coffee_seasonality/data.py](coffee_seasonality/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
