# Study 710 — Olympic-Host-Effect 🏅📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the HOST country's market rally around its own Games? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Mean abnormal return (host ETF vs ^GSPC, [−6mo..+2mo]) is **+3.63%** across **n = 6** hosts, but the **median is −8.91%**: one-sample *t* = **+0.25** (df=5, *p* = 0.82), Wilcoxon *p* = 0.44, bootstrap 95% CI **[−15.1%, +34.1%]**, random-window placebo *p* = 0.60. Only **1/6** hosts actually outperformed. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No signal to trade, and even setting that aside: one non-overlapping 8-month window every ~4 years on a *different* single-country ETF each time — no capacity, no repeatability, nothing a book can be built around. |
| **Do most host countries actually outperform?** | ![BUSTED](https://img.shields.io/badge/Myth--check-BUSTED-8b949e?style=flat-square) | **5 of 6** hosts (Sydney, Beijing, London, Tokyo, Paris) underperformed the world benchmark around their own Games. The lone winner, Rio 2016 (+75.8pp), plausibly rode Brazil's 2016 commodity-bust rebound — not the Games — and single-handedly drags the mean positive. |

> **In one sentence:** across the six Summer Olympics hosts with a contemporaneous
> country ETF (2000→2024), the median host actually *underperformed* the world benchmark
> by about 9 points around its own Games, the mean is rescued only by one outlier with an
> obvious alternative explanation, and every test the desk runs at this sample size — *t*,
> Wilcoxon, bootstrap, placebo — agrees there is no detectable "host effect" to trade.

## What we tested

The claim, as host-city bid books and financial media tell it: *hosting the Summer
Olympics lifts the host's stock market — national pride, an infrastructure boom, a
tourism wave and years of positive coverage should show up as an abnormal return around
the Games.* We hardcode all seven Summer host editions 2000→2024 (IOC results archive),
map six of them to a contemporaneous single-country ETF (Athens 2004 has none — GREK
launched in 2011, seven years too late, and is excluded rather than proxied), and measure
each host ETF's total return over [−6mo..+2mo] around the Games minus the ^GSPC return
over the identical window (a named substitute for URTH/ACWI, whose inception postdates
part of the sample). Beijing 2008 sits inside the Global Financial Crisis; Rio 2016
coincides with Brazil's commodity-bust rebound — both confounders are named on the Signal
axis, not buried. With **n = 6**, the tiny sample is the headline risk, so the study runs
five independent checks (mean, median, Wilcoxon, bootstrap CI, random-window placebo) and
a power curve on a calibrated synthetic control before calling it. **Dedup:**
[234-olympic-year](../234-olympic-year/) tests the US market in *any* Olympic year (not
host-specific); [235-world-cup-effect](../235-world-cup-effect/) tests the *global* market
during World Cup windows (not host-specific); [708-eurovision-effect](../708-eurovision-effect/)
asks the same host-lift question of a much smaller event; [313-geopolitical-shock](../313-geopolitical-shock/)
shares the event-study machinery on an unrelated (negative-valence) trigger. This study
is the one that isolates the **specific host country's** market around **its own** Games.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the host calendar, the six abnormal returns bar by bar, why one big winner isn't the same as a real effect, and the honest "tiny sample" caveat |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-t/Wilcoxon/bootstrap/placebo battery, the sensitivity cuts and why they're a warning not a finding, the power curve, and a 20-seed synthetic null |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`olympic_host_effect/`](olympic_host_effect/). Host-country ETFs are six fixed,
named single instruments (no survivorship panel); the benchmark substitution (^GSPC for
URTH/ACWI) is named throughout. **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
