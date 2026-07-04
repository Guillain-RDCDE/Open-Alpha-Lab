# Study 626 — Unemployment-Trend-Timing 🚦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Only obey the 200-day SMA when unemployment is also rising — does the recession filter really halve the whipsaws?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the filter improve pure trend-following? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | On 77 years of real tape (1949–2026), GTT beat pure Faber timing by **+12.26 bps/mo** at **HAC t = +2.32** (rising to +2.49 at 25 bps costs), cutting whipsaw spells **64%** and switches **46%** while keeping the 2008 and 2000-02 saves. Robust to the filter window (6m/12m; 24m = 1.96). Named caveats: **current-vintage** unemployment (reporting lag modeled, revisions not), price-only leg t = 1.52, and most of the edge is exposure (3rd axis). |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Anyone can run it (S&P fund + T-bills, ~0.8 switches/yr, survives 25 bps) — but the edge over plain **buy & hold** is uncertified (+4.37 bps/mo, *t* = 0.61) and the shield is **regime-dependent**: blind to 1987/2020-style fast crashes (GTT max DD −29.6% is dated Nov-1987), valuable only in slow, recessionary bears. A better Faber, not a money machine. |
| **Skill — or just more time in the market?** | ![Mixed](https://img.shields.io/badge/Skill_beyond_exposure%3F-Mixed-8b949e?style=flat-square) | Of the +12.26 bps/mo, **+10.12 is mechanical exposure** (GTT is long 14.9% more months; any same-duty-cycle filter collects that); the pure-timing residual is **+2.15 bps/mo**, and the exposure-matched **rotation placebo lands at p = 0.0509** — beats ~95% of fake filters, misses the 5% line by a hair. Certified: *same or better with half the trades*. Not quite certified: *the filter times recessions*. |

> **In one sentence:** gating Faber's 200-day SMA on rising unemployment really did make the classic trend rule better — +12 bps/month at *t* = 2.3 over 77 years, with 64% fewer whipsaws and every big recession save intact — but most of that improvement is simply *staying invested more*, the pure-timing residual sits exactly at the 5% line, and the shield is blind to fast crashes like 1987 and COVID — **Real, Fragile, and honest about why**.

## What we tested

Livingston's "Growth-Trend Timing" (Philosophical Economics, 2016): trend-following sell
signals only pay inside recessions, so **veto them when the economy is fine**. We race three
rules on the S&P 500 total-return tape 1949–2026 (^GSPC + Shiller dividend yield; cash =
3-month T-bills): buy & hold, pure Faber (risk-off below the 200-day SMA — the base rule of
[110-faber](../110-faber-timing/)), and GTT (risk-off only if *also* unemployment —
BLS LNS14000000, 1-month reporting lag — is above its 12-month SMA). One execution lag,
5 bps one-way costs (0/5/10/25 sweep), excess-vs-excess Sharpe, Newey-West t on active
returns. The sharp part is the null: GTT holds equities more, so it collects the equity
premium mechanically — we decompose the edge into exposure vs timing and test the filter's
alignment against **all 904 circular rotations** of itself (same persistence and duty cycle,
alignment destroyed). Unlike [268-sahm](../268-sahm-rule/), where unemployment *alone* was a
useless sell button, here it is a **veto on a price signal** — and that construction is what
survives. A deterministic two-regime synthetic control proves the placebo refuses the
raw-t exposure bait. As-of **2026-06-30**, fingerprint `a6195d69eb89`.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why trend rules whipsaw, what the unemployment veto does in plain words, the six bears one by one, and the honest "it mostly just stays invested more" catch |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t on the three active-return pairs, cost sweep, exposure-vs-timing decomposition, the 904-rotation placebo, window/price-only/sub-period robustness, and the synthetic exposure-bait control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`unemployment_trend_timing/`](unemployment_trend_timing/). The signal is the 200-day SMA gated by unemployment vs its 12-month SMA (current-vintage BLS series — named); the myth-check is the exposure-matched rotation placebo. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
