# References & literature map — Study 516 (Dividend-Month-Premium)

## The claim under test

- **The seminal paper.** Samuel M. Hartzmark & David H. Solomon, *The Dividend Month Premium*
  (2013, **Journal of Financial Economics** 109(3), 640–660). Their headline finding: stocks
  earn **abnormally high returns in the calendar months they are predicted to pay a dividend**.
  Because most dividends are paid on a **regular, predictable schedule** (quarterly for most US
  names), the payment month is knowable in advance — and a portfolio long stocks in their
  predicted-dividend months and short otherwise earns a significant premium (they report on the
  order of **~0.4–0.5% per month** in the in-month leg, robust to standard risk controls).
- **The mechanism.** Hartzmark-Solomon attribute it to **price pressure from yield-seeking
  demand**: a clientele of investors (retirees, income funds, "dividends-as-income" households)
  buys ahead of the expected payment to capture the dividend, pushing prices up in the predicted
  month and pulling them back afterward. It is a *demand-driven*, behavioural/clientele effect,
  not a compensation for risk — closely related to the "free dividends fallacy" they document
  elsewhere (Hartzmark & Solomon, *The Dividend Disconnect*, JF 2019).

## What we measure, and why "predicted" not "actual"

- **Predicted, past-only flag.** For each name and month we ask whether its *history* (dividends
  paid **strictly before** that month) has it paying in that calendar month at least twice. This
  is the operational version of "predicted to pay" — knowable in advance from the cadence, no
  contemporaneous information. It is the honest analogue of Hartzmark-Solomon's predicted-payment
  indicator.
- **Total-return vs price-only.** We use yfinance **auto-adjusted (total-return)** closes, so the
  premium is **not** the mechanical drop-and-pay of the cash dividend — the adjustment removes
  the ex-day price drop. The in-month excess return is a genuine *price-pressure* effect, exactly
  the quantity Hartzmark-Solomon isolate (their result holds on returns net of the dividend).
- **Timing / no look-ahead.** The predicted flag is known a month ahead; we enter at the start of
  the predicted month and hold it (a clean one-month execution lag). The standard event/calendar
  convention; no same-bar fills.

## Why a high *t* still needs a placebo + a within-firm unit

- **Per-firm one-sample t** (not per-month). Earnings/dividends cluster by name and by season, so
  pooling stock-months over-counts; we use **one premium per firm** (in-month minus the firm's
  own non-dividend months) as the conservative within-firm unit, then a one-sample *t* against
  zero. Welch (1947, *The generalization of "Student's" problem*) for the pooled comparison.
- **Random-calendar placebo.** We keep each name's *number* of predicted months fixed but
  randomly relocate which months carry the tag (Fisher's randomization logic; Efron & Tibshirani,
  *An Introduction to the Bootstrap*, 1993). The honest answer to "could a random set of calendar
  months of the same density have looked this special?" — here **p ≈ 0.001**.
- **Post-publication decay.** McLean & Pontiff (2016, *Does academic research destroy stock return
  predictability?*, JF) and Harvey, Liu & Zhu (2016, *…and the Cross-Section of Expected Returns*,
  RFS) warn documented anomalies decay after publication; the dividend-month premium has been
  studied since 2013 — our 2000–2026 large-cap replication still clears the bar, which is itself
  notable.

## Why the tradable surplus is thin here

- **Limits to arbitrage / clientele.** The premium is a demand-pressure effect concentrated where
  yield-seeking flows are strongest; on the **most liquid large-caps** (our basket by
  construction) the in-month premium is real but the *tradable* surplus over simply staying
  invested is modest, and costs erode it as the one-way fee rises.
- **Costs and turnover.** A predicted-month overlay round-trips every predicted month; we charge
  one-way costs × the round trip (2/5/10 bps). Frazzini, Israel & Moskowitz (2018, *Trading
  costs*) on the paper-vs-net gap motivates the gross-vs-net discipline.

## Method lineage (the desk's shared engine)

- **Predicted-month flags (past-only).** [`data.build_predicted_flags`](../dividend_month_premium/data.py)
  — learns each name's payment-month cadence from history and tags future months with no
  look-ahead.
- **Per-name premium + one-sample t.** [`strategy.per_name_premium`](../dividend_month_premium/strategy.py)
  and [`strategy.ttest_vs_zero`](../dividend_month_premium/strategy.py).
- **Random-calendar placebo.** [`strategy.placebo_pvalue`](../dividend_month_premium/strategy.py)
  — 20,000 same-density random calendars; the honest small-effect null.
- **Predicted-vs-actual third axis.** [`strategy.build_actual_flags`](../dividend_month_premium/strategy.py)
  — the contemporaneous ex-div flag, to prove the premium is *predictable in advance*.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../dividend_month_premium/data.py)
  plants a known in-month premium; with the edge set to zero the inference must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for a fixed 30-name large-cap survivor basket
  + per-name `Ticker.dividends`, 2000-01-03 → 2026-05-31, cached under `_cache/dmp_prices.csv`
  and `_cache/dmp_divs.csv`. All headline numbers are pinned in [`docs/results.md`](results.md)
  and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [Dividend-capture](../143-dividend-capture) — the *around-the-ex-day* trade (a different,
  shorter-horizon cousin); [dividend-aristocrats](../206-dividend-aristocrats) and
  [dividend-growth](../201-dividend-growth) — payer-quality sorts. This study is distinct: a
  **calendar/month-of-payment** premium, not a payer-quality or ex-day timing effect.
- The **earnings-announcement premium** ([515-earnings-announcement-premium](../515-earnings-announcement-premium))
  is the structural sibling — a predictable calendar window that carries an outsized share of
  return; both are price-pressure-around-a-scheduled-event stories.
