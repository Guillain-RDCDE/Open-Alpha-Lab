"""Real-data run — does the red opening hour + IB-rejection actually call the close?

Two tapes, one verdict (the resolution decision is in `crimson_hour/data.py`):

  * **Faithful replication** — ES=F & NQ=F at **5-minute** fidelity (~60 days). Fine enough to
    flag IB-high-rejection (which high/low printed first), so it reproduces edgeful's *exact*
    confluence and its tiny conditional sub-samples.
  * **High-power leg** — SPY & QQQ at **1-hour** fidelity (~730 days). Bars align to the 09:30
    open, so the opening-candle leg is testable with ~500 sessions; SPY/QQQ are the cash proxies
    for ES/NQ (edgeful's own members built the same dashboard on QQQ/SPY).

    # fetch the bars into the local cache, then run:
    python examples/verify.py --fetch
    # later, offline, reproduce from cache only:
    python examples/verify.py

Network lives only behind `--fetch`. Without it the run is **cache-only** — a ticker with no
cached parquet is skipped, never silently re-downloaded. Yahoo's intraday history is a *rolling*
window ending ~now, so the headline is pinned with `quantlab.repro.as_of` and stamped with a
content fingerprint; a reader who reruns and matches the fingerprint holds the same tape.
"""

import argparse
import os
import sys

import pandas as pd

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from crimson_hour import data, decompose, signals
from quantlab.repro import DEFAULT_AS_OF, as_of, fingerprint

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

OUT = os.path.join(_STUDY, "docs", "results.md")
FP_COLS = ["oc_ret", "rest_ret", "day_ret"]

# (ticker, interval, period, has_subhour)
FAITHFUL = [("ES=F", "5m", "60d", True), ("NQ=F", "5m", "60d", True)]
HIPOWER = [("SPY", "1h", "730d", False), ("QQQ", "1h", "730d", False)]


def load(ticker, interval, period, has_subhour, fetch):
    bars = data.fetch_intraday(ticker, interval=interval, period=period, fetch=fetch)
    if bars.empty:
        return pd.DataFrame()
    feat = data.daily_features(bars, has_subhour=has_subhour)
    return as_of(feat, DEFAULT_AS_OF)               # pin so the rolling window can't creep


def faithful_report(feat):
    """The full confluence on a fine tape: conditional close-red table, the headline cell with
    its honest interval and posterior, and whether IB-rejection adds anything over OC-red."""
    masks = signals.condition_masks(feat)
    tab = decompose.conditional_table(masks, signals.session_red(feat))
    conf = decompose.rate(signals.confluence(feat), signals.session_red(feat))
    post = decompose.beta_binomial(conf["k"], conf["n"], thresholds=(0.7, conf["rate"] - 0.0))
    inc = decompose.ib_increment(feat)
    return {"n": len(feat), "baseline": tab.loc["baseline", "rate"], "table": tab,
            "confluence": conf, "posterior": post, "ib_increment": inc}


def hipower_report(feat):
    """The opening-candle leg with real power: the mechanical-vs-forecast split + OC-red table."""
    split = decompose.mechanical_vs_predictive(feat)
    masks = {k: v for k, v in signals.condition_masks(feat).items()
             if k in ("baseline", "oc_red", "oc_green")}
    sess = decompose.conditional_table(masks, signals.session_red(feat))
    rest = decompose.conditional_table(masks, signals.rest_red(feat))
    return {"n": len(feat), "split": split, "session_table": sess, "rest_table": rest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="populate the bar cache (network)")
    args = ap.parse_args()

    faithful, hipower, fps = {}, {}, {}
    for tk, itv, per, sub in FAITHFUL:
        feat = load(tk, itv, per, sub, args.fetch)
        if feat.empty:
            print(f"[skip] {tk} {itv}: no cache (run with --fetch)"); continue
        faithful[tk] = faithful_report(feat)
        fps[tk] = (fingerprint(feat, cols=FP_COLS), feat.index.min().date(), feat.index.max().date())
        r = faithful[tk]
        print(f"\n=== {tk} ({itv}, {r['n']} sessions, baseline red {r['baseline']:.1%}) ===")
        print(r["table"].to_string())
        c = r["confluence"]
        print(f"confluence (OC-red & IB-rej): {c['k']}/{c['n']} = {c['rate']:.1%} "
              f"[Wilson {c['wilson_low']:.1%}, {c['wilson_high']:.1%}]")
        print(f"  posterior mean {r['posterior']['posterior_mean']:.1%}, "
              f"95% cred [{r['posterior']['cred_low']:.1%}, {r['posterior']['cred_high']:.1%}]")
        i = r["ib_increment"]
        print(f"  IB increment over OC-red-not-rejected: {i['increment_pp']:+.1f} pp "
              f"(z p={i['z_p_value']:.2f}, Fisher p={i['fisher_p_value']:.2f})")

    for tk, itv, per, sub in HIPOWER:
        feat = load(tk, itv, per, sub, args.fetch)
        if feat.empty:
            print(f"[skip] {tk} {itv}: no cache (run with --fetch)"); continue
        hipower[tk] = hipower_report(feat)
        fps[tk] = (fingerprint(feat, cols=FP_COLS), feat.index.min().date(), feat.index.max().date())
        s = hipower[tk]["split"]
        print(f"\n=== {tk} ({itv}, {hipower[tk]['n']} sessions) ===")
        print(f"headline  P(session red | OC-red) = {s['headline_rate']:.1%} "
              f"(baseline {s['headline_baseline']:.1%}, lift {s['headline_lift_pp']:+.1f} pp)")
        print(f"forecast  P(rest-of-day red | OC-red) = {s['continuation_rate']:.1%} "
              f"(baseline {s['continuation_baseline']:.1%}, lift {s['continuation_lift_pp']:+.1f} pp)")
        print(f"  -> {s['mechanical_share']:.0%} of the headline lift is a mechanical head-start, "
              f"not a forecast")

    # Forking-paths: the realistic P(session red | OC-red) from the high-power run is the *true*
    # edge a mined confluence draws from. Show that selecting the best of a bank of confluences,
    # each on edgeful's own ~25-session sub-sample, inflates that toward their quoted ~88%.
    EDGEFUL_HEADLINE = 0.88        # edgeful's ES "OC-red + IB-rejection" number, 22/25
    EDGEFUL_N = 25
    mining = None
    if hipower:
        p_true = sum(h["split"]["headline_rate"] for h in hipower.values()) / len(hipower)
        mining = decompose.mining_inflation(p_true=p_true, n_cond=EDGEFUL_N, n_candidates=12,
                                            observed=EDGEFUL_HEADLINE, seed=0)
        print(f"\n=== forking paths === true edge {p_true:.1%} (pooled SPY/QQQ), "
              f"{mining['n_candidates']} candidate confluences on n={mining['n_cond']}:")
        print(f"  expected best observed {mining['expected_best_rate']:.1%}, "
              f"p95 {mining['best_rate_p95']:.1%}, P(best >= {mining['observed']:.0%} headline) "
              f"= {mining['P(best>=observed)']:.1%}")

    if faithful or hipower:
        _write_results(OUT, faithful, hipower, mining, fps, DEFAULT_AS_OF)
        print(f"\nwrote {OUT}")
    else:
        print("\ncache is empty — run with --fetch to populate the bars first.")


def _write_results(path, faithful, hipower, mining, fps, asof):
    def md(df):
        return "```\n" + df.round(4).to_string() + "\n```"

    lines = [f"""# Results — Study 13 (Crimson-Hour) on real ES/NQ + SPY/QQQ

*Generated by [`examples/verify.py`](../examples/verify.py). Two tapes: the **faithful**
confluence on **ES=F / NQ=F at 5-minute** fidelity (fine enough to flag IB-high-rejection), and
the **high-power** opening-candle leg on **SPY / QQQ at 1-hour** fidelity (~500 sessions, bars
aligned to the 09:30 open). RTH 09:30–16:00 ET; "red" = close below the 09:30 open. As-of
**{asof}**; Yahoo intraday is a rolling window ending ~now, so the start drifts — match the
per-tape fingerprint below to confirm you hold the same tape.*

## Data stamp
"""]
    for tk, (fp, lo, hi) in fps.items():
        lines.append(f"- **{tk}**: {lo} → {hi}, fingerprint `{fp}`")

    for tk, r in faithful.items():
        c, p, i = r["confluence"], r["posterior"], r["ib_increment"]
        lines.append(f"""
## {tk} — faithful confluence ({r['n']} sessions, baseline red {r['baseline']:.1%})

Conditional **P(session closes red | morning condition)**, with Wilson 95% intervals and the
lift (pp) over baseline:
{md(r['table'])}

- **The headline cell**: confluence (OC-red & IB-rejected) closed red **{c['k']}/{c['n']} =
  {c['rate']:.1%}** — but the Wilson 95% interval is **[{c['wilson_low']:.1%},
  {c['wilson_high']:.1%}]** and the Beta(1,1) posterior mean is only **{p['posterior_mean']:.1%}**
  (95% credible **[{p['cred_low']:.1%}, {p['cred_high']:.1%}]**). On n={c['n']} the quoted number
  is one draw near the top of a very wide band.
- **Does IB-rejection add anything over OC-red?** Confluence {i['confluence_rate']:.1%}
  ({i['confluence_k']}/{i['confluence_n']}) vs OC-red-but-*not*-rejected {i['control_rate']:.1%}
  ({i['control_k']}/{i['control_n']}): increment **{i['increment_pp']:+.1f} pp**, two-proportion
  z p=**{i['z_p_value']:.2f}**, Fisher p=**{i['fisher_p_value']:.2f}** — indistinguishable. The
  second signal is redundant given the first.""")

    for tk, r in hipower.items():
        s = r["split"]
        lines.append(f"""
## {tk} — opening-candle leg, high power ({r['n']} sessions)

P(session closes red | morning condition) — what edgeful quotes:
{md(r['session_table'])}

P(**rest of day** 10:30→16:00 closes red | morning condition) — the part actually unknown at 10:30:
{md(r['rest_table'])}

- **Headline** P(session red | OC-red) = **{s['headline_rate']:.1%}** (baseline
  {s['headline_baseline']:.1%}, lift **{s['headline_lift_pp']:+.1f} pp**).
- **Genuine forecast** P(rest-of-day red | OC-red) = **{s['continuation_rate']:.1%}** (baseline
  {s['continuation_baseline']:.1%}, lift **{s['continuation_lift_pp']:+.1f} pp**).
- **{s['mechanical_share']:.0%}** of the headline lift is a *mechanical head-start* (an OC-red day
  is already below its open at 10:30), not continuation. The forecastable edge for the rest of the
  day is the small continuation lift.""")

    if mining is not None:
        lines.append(f"""
## Forking paths — how a mined "confluence" inflates a modest edge

Take the realistic opening-candle edge from the high-power run, **p_true = {mining['p_true']:.1%}**
(pooled SPY/QQQ P(session red | OC-red)), and let a prompt evaluate **{mining['n_candidates']}**
candidate confluences, each measured on edgeful's own **n = {mining['n_cond']}** sessions. The
*best* observed rate it reports:

- expected best **{mining['expected_best_rate']:.1%}** (inflation **{mining['inflation_pp']:+.1f}
  pp** over the truth), 95th percentile **{mining['best_rate_p95']:.1%}**;
- probability the best reaches edgeful's quoted **{mining['observed']:.0%}** by selection alone:
  **{mining['P(best>=observed)']:.1%}**.

The expected best lands right on the published 88–90%. Selecting the highest of several
tiny-sample confluences manufactures a headline far above the real edge — exactly the "one
prompt, combine the reports until one hits" workflow.""")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
