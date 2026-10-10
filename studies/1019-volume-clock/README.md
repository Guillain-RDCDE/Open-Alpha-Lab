# Study 1019 — The Market Runs on Volume Time ⏱️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a volume clock make daily returns Gaussian? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Volume and volatility move together on the same day without any doubt (correlation of relative volume with the absolute return **+0.22 / +0.19**, Newey–West *t* **8.3** on the S&P 500 and the Nasdaq). But dividing each day's return by the square root of its detrended volume removes only **14%** (S&P, block-bootstrap CI −12% to 26%) and **21%** (Nasdaq, CI 1% to 38%) of the excess kurtosis — under the pre-registered 25% bar on both tapes — and Jarque–Bera still rejects normality. A past-only GARCH, using no same-day information at all, removes **76% / 78%**. The fat tail is mostly volatility clustering, not a volume clock. |
| **Tradability** — is the clock usable before the close it is read at? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Lagged volume added to a range-based log-HAR changes next-day QLIKE with Diebold–Mariano *t* **+0.95 / +0.12** (negative would favour volume), and the vol-targeting overlay built on it beats the plain one by **+0.001 / +0.000** Sharpe after 2 bp, bootstrap *p* 0.80 / 0.95. Sizing by *same-day* volume — impossible, since it is published after the close — would add about 0.1 Sharpe: the clock's value sits on the wrong side of the close. |

> **In one sentence:** Volume and volatility move together, but volume time is a weak clock: dividing by same-day volume removes 14%–21% of the excess kurtosis of daily index returns and leaves them far from normal, and by the time anyone can read the clock it is spent — lagged volume adds nothing significant to a range-based HAR forecast and nothing to a vol-targeting overlay.

## What we tested

Returns are Gaussian — just not in calendar time. [Clark (1973, *Econometrica*)](docs/references.md)
modelled daily prices as a normal random walk run on a random *operational* clock that ticks with
trading volume, so that fat tails are simply what you see when busy and quiet days are mixed;
[Ané & Geman (2000, *Journal of Finance*)](docs/references.md) reported near-normal returns once
time is counted in trades. Steelmanned: *divide each day's return by the square root of how much
the clock ran, and the fat tails go away.*

On the S&P 500 and Nasdaq Composite daily tapes frozen inside `arch` (1999–2018, price indices,
composite index volume), we standardise returns by Clark's volume clock (past-only detrended),
by the day's own range (an upper benchmark) and by a past-only GARCH (the usual one), with paired
block-bootstrap intervals; then take the honest catch seriously — volume is only known after the
close — and race a log-HAR with and without **lagged** volume out of sample, with HAC
Diebold–Mariano tests and a vol-targeting overlay traded one session late, after costs. A
synthetic subordinated process with the clock planted and switched off checks the machinery.
**Dedup:** distinct from **991-aggregational-gaussianity** (normality by lengthening the calendar
horizon, not by changing the clock), **965-range-vol-estimators** and **966-har-vs-garch** (which
estimator or model forecasts best — here the range-based HAR is the baseline and the question is
the marginal value of volume), and **992-vol-clustering-halflife** (the persistence that turns out
to carry most of the fat tail).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a busy day might be "more time", what volume does and does not explain, and why yesterday's volume is old news |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | kurtosis removed per clock with paired block-bootstrap CIs, exponent and detrend robustness, volume on top of GARCH, the lag profile, HAR-X vs HAR vs GARCH with Diebold–Mariano, the lagged overlay and its cost sweep, and the synthetic control |

Reproducible headline run (SHA-pinned tapes, fingerprints, every table): [docs/results.md](docs/results.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/) + [`volclock/`](volclock/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
