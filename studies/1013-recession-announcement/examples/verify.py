"""Real-tape verification — Study 1013 (It's Official). Regenerates docs/results.md.

Measures how far the stock market's own turning points lead the NBER's official peak and
trough dates over every US cycle since 1926, what was already done by the day each of the
twelve official announcements since 1980 went out (the drawdown already realised, the rebound
already missed), whether the announcement dates are followed by better returns than every
circular shift of the same dates, how much power twelve events could ever have, and what the
believer's trading rules earn net of costs with one execution lag.

    python studies/1013-recession-announcement/examples/verify.py

Offline: both tapes come from ``quantlab.bundled`` (frozen, SHA-256 pinned).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from nberclock import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

COST_BPS = 10.0
HORIZON_M = 12           # pre-registered primary horizon, months
RULE_START = "1980-01-01"
POWER_SIMS = 200


def _rot(level, anns, kinds, h):
    r = st.rotation_test(level, st.event_positions(level.index, anns, kinds), h)
    return {k: v for k, v in r.items() if k != "perm"}


def report() -> dict:
    t0 = time.time()
    m = data.load_monthly()
    d = data.load_daily()
    tr = data.total_return_index(m)
    xs = st.excess_index(m)
    cyc = data.cycles()
    anns = data.announcements()
    prov = data.provenance()
    h: dict = {"as_of": data.AS_OF, "as_of_monthly": data.AS_OF_MONTHLY,
               "as_of_daily": data.AS_OF_DAILY,
               "fp_monthly": data.fingerprint(m), "fp_daily": data.fingerprint(d),
               "monthly_span": f"{m.index[0].date()} → {m.index[-1].date()}",
               "daily_span": f"{d.index[0].date()} → {d.index[-1].date()}",
               "provenance": prov, "cost_bps": COST_BPS}
    print(f"as-of {data.AS_OF_MONTHLY} (monthly TR) / {data.AS_OF_DAILY} (daily price)")
    print(f"  fingerprints monthly {h['fp_monthly']}  daily {h['fp_daily']}")

    # ---------------------------------------------------------------- 1. calendar
    print("\n=== 1. the official calendar ===")
    print(anns.to_string(index=False))
    h["anns"] = anns.assign(reference=anns["reference"].dt.strftime("%Y-%m"),
                            date=anns["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
    h["lag_peak_median"] = float(anns.loc[anns.kind == "peak", "lag_months"].median())
    h["lag_trough_median"] = float(anns.loc[anns.kind == "trough", "lag_months"].median())
    print(f"  median lag: peak {h['lag_peak_median']:.1f} months, trough "
          f"{h['lag_trough_median']:.1f} months")

    # ---------------------------------------------------------------- 2. lead / lag
    print("\n=== 2. how far the market leads the committee ===")
    ll = st.lead_lag_all(tr, d, cyc)
    show = ll.copy()
    for c in ("peak", "trough", "mkt_peak", "mkt_trough"):
        show[c] = show[c].dt.strftime("%Y-%m")
    print(show[["peak", "trough", "mkt_peak", "mkt_trough", "lead_peak", "lead_trough",
                "dd_at_nber_peak", "max_dd", "censored", "tape"]].round(3).to_string())
    h["lead_table"] = show.to_dict("records")
    s_all = st.lead_summary(ll)
    s_bear = st.lead_summary(ll, min_fall=0.10)
    h["lead_all"], h["lead_bear"] = s_all, s_bear
    h["n_cycles_lead"] = s_all["n"]
    h["lead_trough_pos"] = s_all["trough_pos"]
    h["lead_trough_sign_p"] = s_all["trough_p"]
    h["lead_trough_median"] = s_all["trough_median"]
    h["lead_peak_pos"] = s_all["peak_pos"]
    h["lead_peak_median"] = s_all["peak_median"]
    h["lead_peak_sign_p"] = s_all["peak_p"]
    print(f"  uncensored cycles: {s_all['n']}. Market trough first in {s_all['trough_pos']}, "
          f"after in {s_all['trough_neg']}, same month {s_all['trough_zero']} "
          f"(sign p = {s_all['trough_p']:.4f}); median lead {s_all['trough_median']:.0f} m")
    print(f"  market peak first in {s_all['peak_pos']}, after in {s_all['peak_neg']} "
          f"(sign p = {s_all['peak_p']:.4f}); median lead {s_all['peak_median']:.0f} m")
    print(f"  only real bears (fall >= 10%): {s_bear['n']} cycles, trough first "
          f"{s_bear['trough_pos']}/{s_bear['n']} p = {s_bear['trough_p']:.4f}")
    lr = st.lead_rotation_test(tr, cyc)
    h["lead_rot"] = {k: v for k, v in lr.items() if not k.startswith("perm")}
    h["lead_rot_p_share"] = lr["p_share"]
    h["lead_rot_null_share"] = lr["null_share_mean"]
    h["lead_rot_obs_share"] = lr["observed_share"]
    h["lead_rot_p_mean"] = lr["p_mean_lead"]
    h["lead_rot_obs_mean"] = lr["observed_mean_lead"]
    h["lead_rot_null_mean"] = lr["null_mean_lead"]
    grid = []
    for slack in (0, 3, 6):
        for post in (6, 12, 18):
            g = st.lead_rotation_test(tr, cyc, peak_slack=slack, post_months=post)
            grid.append({"peak_slack": slack, "post_months": post,
                         "share": g["observed_share"], "null_share": g["null_share_mean"],
                         "p_share": g["p_share"], "p_mean": g["p_mean_lead"]})
    h["lead_grid"] = grid
    h["lead_grid_n_sig"] = int(sum(g["p_share"] < 0.05 for g in grid))
    print(pd.DataFrame(grid).round(3).to_string(index=False))
    print(f"  calendar rotation ({lr['n_cycles']} TR cycles, {lr['n_shifts']} shifts): market "
          f"first {lr['observed_share']:.0%} vs {lr['null_share_mean']:.0%} on shifted dates, "
          f"p = {lr['p_share']:.3f}; mean lead {lr['observed_mean_lead']:.1f} vs "
          f"{lr['null_mean_lead']:.1f} m, p = {lr['p_mean_lead']:.2f}")

    # ---------------------------------------------------------------- 3. at the announcement
    print("\n=== 3. what was already done on announcement day ===")
    at_m = st.announcement_table(tr, anns)
    at_d = st.announcement_table(d, anns, bars_per_month=21)
    best = st.best_available(at_m, at_d)
    for name, t in (("monthly TR", at_m), ("daily price", at_d), ("best available", best)):
        print(f"  -- {name}")
        tt = t[["kind", "date", "lag_months", "already", "still_to_come", "share_done",
                "months_since_extreme", "fwd_12m"]].copy()
        tt["date"] = tt["date"].dt.strftime("%Y-%m-%d")
        print(tt.round(3).to_string(index=False))

    def recs(t):
        t = t.copy()
        for c in ("reference", "date", "bar"):
            t[c] = pd.to_datetime(t[c]).dt.strftime("%Y-%m-%d")
        return t.to_dict("records")

    h["at_monthly"], h["at_daily"], h["at_best"] = recs(at_m), recs(at_d), recs(best)
    pk = best[best.kind == "peak"]
    tg = best[best.kind == "trough"]
    h["dd_already_median"] = float(pk["already"].median())
    h["dd_already_n_nofall"] = int((pk["already"] > -0.02).sum())
    h["share_done_median"] = float(pk["share_done"].median())
    h["still_to_come_median"] = float(pk["still_to_come"].median())
    h["missed_median"] = float(tg["already"].median())
    h["missed_min"] = float(tg["already"].min())
    h["months_since_low_median"] = float(tg["months_since_extreme"].median())
    print(f"  peaks: median drawdown already realised {h['dd_already_median']:.1%}, "
          f"{h['dd_already_n_nofall']} of {len(pk)} with no fall at all; median share of the "
          f"cycle's fall done {h['share_done_median']:.0%}; median still to come "
          f"{h['still_to_come_median']:.1%}")
    print(f"  troughs: median rebound already missed {h['missed_median']:+.0%} "
          f"(min {h['missed_min']:+.0%}), low a median "
          f"{h['months_since_low_median']:.0f} months earlier")

    # ---------------------------------------------------------------- 4. the buy signal
    print("\n=== 4. is the announcement a buy signal? (rotation test) ===")
    rot = {}
    for kinds, lbl in ((("peak", "trough"), "pooled"), (("peak",), "peak"),
                       (("trough",), "trough")):
        for hm in (3, 6, 12, 24):
            rot[(lbl, "monthly", hm)] = _rot(xs, anns, kinds, hm)
            rot[(lbl, "daily", hm)] = _rot(d, anns, kinds, hm * 21)
    xs_pw = xs[xs.index >= "1947-01-01"]
    for kinds, lbl in ((("peak", "trough"), "pooled"), (("peak",), "peak"),
                       (("trough",), "trough")):
        rot[(lbl, "postwar", 12)] = _rot(xs_pw, anns, kinds, 12)
    rows = []
    for (lbl, tape, hm), r in rot.items():
        rows.append({"events": lbl, "tape": tape, "h_months": hm, "n": r["n_events"],
                     "observed": r["observed"], "random": r["null_mean"],
                     "excess": r["excess_vs_random"], "z": r["z"],
                     "p_upper": r["p_one_sided"], "p_lower": r["p_lower"],
                     "shifts": r["n_shifts"]})
    rt = pd.DataFrame(rows)
    print(rt.round(3).to_string(index=False))
    h["rotation"] = rt.to_dict("records")
    prim = rot[("pooled", "monthly", HORIZON_M)]
    h["n_events_monthly"] = prim["n_events"]
    h["buy_obs_pooled"] = prim["observed"]
    h["buy_null_pooled"] = prim["null_mean"]
    h["buy_p_pooled"] = prim["p_one_sided"]
    h["buy_p_peak"] = rot[("peak", "monthly", HORIZON_M)]["p_one_sided"]
    h["buy_p_trough"] = rot[("trough", "monthly", HORIZON_M)]["p_one_sided"]
    h["buy_p_trough_lower"] = rot[("trough", "monthly", HORIZON_M)]["p_lower"]
    h["buy_obs_peak"] = rot[("peak", "monthly", HORIZON_M)]["observed"]
    h["buy_obs_trough"] = rot[("trough", "monthly", HORIZON_M)]["observed"]
    h["buy_p_pooled_daily"] = rot[("pooled", "daily", HORIZON_M)]["p_one_sided"]
    h["n_events_daily"] = rot[("pooled", "daily", HORIZON_M)]["n_events"]
    h["buy_p_pooled_postwar"] = rot[("pooled", "postwar", 12)]["p_one_sided"]
    h["event_returns_monthly"] = prim["event_returns"]
    print(f"  primary (pre-registered): pooled, monthly TR excess, 12 m: observed "
          f"{h['buy_obs_pooled']:+.1%} vs random {h['buy_null_pooled']:+.1%}, "
          f"p = {h['buy_p_pooled']:.3f}")

    # ---------------------------------------------------------------- 5. power
    print("\n=== 5. power: how big an effect could this many events see? ===")
    vol = float(m["mkt"].std() * np.sqrt(12))
    n_years = int(round(len(m) / 12))
    effects = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    pc10 = st.power_curve(effects=effects, n_sims=POWER_SIMS, n_cycles=5, n_years=n_years,
                          vol=vol)
    pc6 = st.power_curve(effects=effects, n_sims=POWER_SIMS, n_cycles=3, n_years=n_years,
                         vol=vol, seed=2013)
    print(pc10.round(3).to_string())
    h["power10"] = pc10.reset_index().to_dict("records")
    h["power6"] = pc6.reset_index().to_dict("records")
    h["mde80"] = st.minimum_detectable_effect(pc10, 0.8)
    h["mde80_6"] = st.minimum_detectable_effect(pc6, 0.8)
    h["size_at_null"] = float(pc10.loc[0.0, "reject_rate"])
    h["power_vol"] = vol
    print(f"  {pc10['n_events'].iloc[0]} events, vol {vol:.1%}: 80% power needs a "
          f"{h['mde80']:.1%} abnormal 12-month return; 6 events (one kind) need "
          f"{h['mde80_6']:.1%}. Size at zero effect: {h['size_at_null']:.1%}")

    # ---------------------------------------------------------------- 6. rules
    print("\n=== 6. could you trade it? ===")
    mm = m[m.index >= RULE_START]
    sig_m = {"bills while officially in recession": st.signal_avoid(mm.index, anns),
             "equity 12 m after each announcement": st.signal_buy_after(mm.index, anns, 12),
             "equity 24 m after each peak announcement":
                 st.signal_buy_after(mm.index, anns, 24, ("peak",))}
    cm = st.compare_rules(mm["mkt"], mm["rf"], sig_m, COST_BPS, 12, hac_lags=12)
    dr = d.pct_change().dropna()
    rf0 = pd.Series(0.0, index=dr.index)
    sig_d = {"bills while officially in recession": st.signal_avoid(dr.index, anns),
             "equity 12 m after each announcement": st.signal_buy_after(dr.index, anns, 252),
             "equity 24 m after each peak announcement":
                 st.signal_buy_after(dr.index, anns, 504, ("peak",))}
    cd = st.compare_rules(dr, rf0, sig_d, COST_BPS, 252, hac_lags=21)
    print("  -- monthly, FF total return vs T-bills, 1980-01 → 2018-11")
    print(cm.round(3).to_string())
    print("  -- daily, S&P 500 PRICE vs 0% cash, 1990-01 → 2022-11")
    print(cd.round(3).to_string())
    h["rules_monthly"] = cm.reset_index().to_dict("records")
    h["rules_daily"] = cd.reset_index().to_dict("records")
    key = "equity 12 m after each announcement"
    avoid = "bills while officially in recession"
    h["rule_gain_monthly"] = float(cm.loc[key, "sharpe_gain_net"])
    h["rule_gain_daily"] = float(cd.loc[key, "sharpe_gain_net"])
    h["rule_hac_t_monthly"] = float(cm.loc[key, "active_hac_t"])
    h["rule_sharpe_monthly"] = float(cm.loc[key, "sharpe_net"])
    h["bh_sharpe_monthly"] = float(cm.loc["buy-and-hold", "sharpe_net"])
    h["rule_exposure"] = float(cm.loc[key, "exposure"])
    h["avoid_gain_monthly"] = float(cm.loc[avoid, "sharpe_gain_net"])
    h["avoid_gain_daily"] = float(cd.loc[avoid, "sharpe_gain_net"])
    h["avoid_hac_t_monthly"] = float(cm.loc[avoid, "active_hac_t"])
    perm_rows = []
    for name, sig in sig_m.items():
        r = st.rule_rotation_test(mm["mkt"], mm["rf"], sig, COST_BPS, 12)
        rd = st.rule_rotation_test(dr, rf0, sig_d[name], COST_BPS, 252, step=5)
        perm_rows.append({"rule": name, "sharpe_monthly": r["observed_sharpe"],
                          "shifted_mean_monthly": r["null_mean"],
                          "p_monthly": r["p_one_sided"],
                          "sharpe_daily": rd["observed_sharpe"],
                          "shifted_mean_daily": rd["null_mean"], "p_daily": rd["p_one_sided"]})
    pr = pd.DataFrame(perm_rows).set_index("rule")
    print(pr.round(3).to_string())
    h["rule_perm"] = pr.reset_index().to_dict("records")
    h["rule_perm_p_monthly"] = float(pr.loc[key, "p_monthly"])
    sw = st.cost_sweep(mm["mkt"], mm["rf"], sig_m[key], costs=(0, 5, 10, 25, 50))
    sw_a = st.cost_sweep(mm["mkt"], mm["rf"], sig_m[avoid], costs=(0, 5, 10, 25, 50))
    h["cost_sweep"] = sw.reset_index().to_dict("records")
    h["cost_sweep_avoid"] = sw_a.reset_index().to_dict("records")
    print(sw.round(3).to_string())

    # ---------------------------------------------------------------- 7. synthetic control
    print("\n=== 7. synthetic control (machinery proof, NOT market evidence) ===")
    ctl = []
    for s in (1.0, 0.0):
        w = data.synthetic_world(signal_strength=s, seed=1013)
        lvl = st.excess_index(w["returns"])
        tri = (1.0 + w["returns"]["mkt"]).cumprod()
        lls = st.lead_summary(st.lead_lag(tri, w["cycles"]))
        lrs = st.lead_rotation_test(tri, w["cycles"])
        r = st.rotation_test(lvl, st.event_positions(lvl.index, w["announcements"]), 12)
        ctl.append({"signal_strength": s, "planted_lead": w["truth"]["lead_months"],
                    "measured_trough_lead": lls["trough_median"],
                    "trough_sign_p": lls["trough_p"],
                    "lead_rot_p": lrs["p_share"], "lead_rot_null_share": lrs["null_share_mean"],
                    "planted_premium": w["truth"]["announce_premium"],
                    "measured_excess": r["excess_vs_random"], "rotation_p": r["p_one_sided"],
                    "n_events": r["n_events"]})
    ct = pd.DataFrame(ctl)
    print(ct.round(3).to_string(index=False))
    h["control"] = ct.to_dict("records")

    h["_verdict"] = st.verdict(h)
    h["runtime_s"] = round(time.time() - t0, 1)
    print("\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    print(f"  {h['_verdict']['one_sentence']}")
    print(f"  runtime {h['runtime_s']} s")
    return h


def _pct(x, plus=False, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}%}" if plus else f"{x:.{nd}%}"


def _int(x):
    return "—" if x is None or not np.isfinite(x) else str(int(x))


def _t(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.2f}"


def results_md(h: dict) -> str:
    v = h["_verdict"]
    p = h["provenance"]
    cal = "\n".join(
        f"| {r['kind']} | {r['reference']} | {r['date']} | {r['lag_months']} |"
        for r in h["anns"])
    lead = "\n".join(
        f"| {r['peak']} → {r['trough']} | {r['mkt_peak']} | {r['mkt_trough']} | "
        f"{r['lead_peak']:+d} | **{r['lead_trough']:+d}** | {_pct(r['dd_at_nber_peak'])} | "
        f"{_pct(r['max_dd'])} | {r['tape']}{' (censored)' if r['censored'] else ''} |"
        for r in h["lead_table"])
    la, lb = h["lead_all"], h["lead_bear"]
    ann = "\n".join(
        f"| {r['kind']} | {r['date']} | {r['lag_months']} | {_pct(r['already'], True)} | "
        f"{_pct(r['still_to_come'], True)} | "
        f"{_pct(r['share_done'], nd=0)} | {_int(r['months_since_extreme'])} | "
        f"{_pct(r['fwd_12m'], True)} | {r['tape']} |"
        for r in h["at_best"])
    rot = "\n".join(
        f"| {r['events']} | {r['tape']} | {r['h_months']} | {r['n']} | "
        f"{_pct(r['observed'], True)} | {_pct(r['random'], True)} | "
        f"{_pct(r['excess'], True)} | {r['p_upper']:.3f} | {r['p_lower']:.3f} |"
        for r in h["rotation"])
    pw = "\n".join(
        f"| {r['effect']:.0%} | {r['reject_rate']:.0%} | "
        f"{next(q['reject_rate'] for q in h['power6'] if q['effect'] == r['effect']):.0%} |"
        for r in h["power10"])

    def rules(rs):
        return "\n".join(
            f"| {r['rule']} | {r['exposure']:.0%} | {r['switches']} | "
            f"{_pct(r['ann_excess_gross'], True)} | {_pct(r['ann_excess_net'], True)} | "
            f"{r['sharpe_gross']:.2f} | **{r['sharpe_net']:.2f}** | "
            f"{r['sharpe_gain_net']:+.2f} | "
            f"{_t(r['active_hac_t'])} | "
            f"{_pct(r['max_dd_net'])} |" for r in rs)

    perm = "\n".join(
        f"| {r['rule']} | {r['sharpe_monthly']:.2f} | {r['shifted_mean_monthly']:.2f} | "
        f"{r['p_monthly']:.2f} | {r['sharpe_daily']:.2f} | {r['shifted_mean_daily']:.2f} | "
        f"{r['p_daily']:.2f} |" for r in h["rule_perm"])
    sw = "\n".join(
        f"| {int(r['cost_bps'])} | {r['sharpe_net']:.3f} | {r['bh_sharpe']:.3f} | "
        f"{r['gain']:+.3f} | "
        f"{next(q['gain'] for q in h['cost_sweep_avoid'] if q['cost_bps'] == r['cost_bps']):+.3f} |"
        for r in h["cost_sweep"])
    grid = "\n".join(
        f"| {g['peak_slack']} | {g['post_months']} | {g['share']:.0%} | {g['null_share']:.0%} | "
        f"{g['p_share']:.3f} | {g['p_mean']:.2f} |" for g in h["lead_grid"])
    ctl = "\n".join(
        f"| {r['signal_strength']:.1f} | {r['planted_lead']} | {r['measured_trough_lead']:+.0f} | "
        f"{r['trough_sign_p']:.4f} | {r['lead_rot_p']:.3f} | {_pct(r['planted_premium'], True)} | "
        f"{_pct(r['measured_excess'], True)} | {r['rotation_p']:.3f} |" for r in h["control"])
    return f"""# Results — Study 1013 (It's Official) on the real tapes

*Generated by [`examples/verify.py`](../examples/verify.py) in {h['runtime_s']} s. As-of
**{h['as_of_monthly']}** for the monthly total-return tape and **{h['as_of_daily']}** for the
daily price tape (the last complete month of each; December 2022 is partial and dropped).*

## 0. Data and provenance

| Tape | Package · file | SHA-256 pin | Span used | Fingerprint | What it is |
|---|---|---|---|---|---|
| Monthly | {p['monthly']['package']} · `{p['monthly']['file']}` | `{p['monthly']['sha256'][:16]}…` | {h['monthly_span']} | `{h['fp_monthly']}` | Fama-French market **total return** (Mkt-RF + RF) and one-month T-bill |
| Daily | {p['daily']['package']} · `{p['daily']['file']}` | `{p['daily']['sha256'][:16]}…` | {h['daily_span']} | `{h['fp_daily']}` | S&P 500 **price index** — no dividends |

NBER reference dates and announcement dates are hard-coded from the NBER Business Cycle Dating
Committee (nber.org/research/business-cycle-dating). Every number below says which tape it
came from; the 2020 cycle and its two announcements exist only on the price tape.

## 1. The official calendar

| Kind | Reference month | Announced | Lag (months) |
|---|---|---|--:|
{cal}

The committee announced peaks a median **{h['lag_peak_median']:.1f} months** after they
happened and troughs a median **{h['lag_trough_median']:.1f} months** after.

## 2. How far the market leads the committee (16 cycles, 1926 → 2020)

Market peak = highest month-end in [NBER peak − 24 m, NBER peak + 3 m]; market trough = lowest
month-end in [market peak, NBER trough + 12 m]. Lead > 0 means the market turned first.

| NBER peak → trough | Market peak | Market trough | Peak lead (m) | Trough lead (m) | Drawdown at NBER peak | Full fall | Tape |
|---|---|---|--:|--:|--:|--:|---|
{lead}

| | Cycles | Market trough first | After | Same month | Exact sign-test p | Median lead |
|---|--:|--:|--:|--:|--:|--:|
| All uncensored | {la['n']} | **{la['trough_pos']}** | {la['trough_neg']} | {la['trough_zero']} | **{la['trough_p']:.4f}** | **{la['trough_median']:.0f} m** |
| Real bears only (fall ≥ 10%) | {lb['n']} | {lb['trough_pos']} | {lb['trough_neg']} | {lb['trough_zero']} | {lb['trough_p']:.4f} | {lb['trough_median']:.0f} m |

| | Cycles | Market peak first | After | Same month | Exact sign-test p | Median lead |
|---|--:|--:|--:|--:|--:|--:|
| All uncensored | {la['n']} | {la['peak_pos']} | {la['peak_neg']} | {la['peak_zero']} | {la['peak_p']:.4f} | {la['peak_median']:.0f} m |

**Is that more than the ruler's own lean?** The turning-point rule is not neutral: the market
trough is searched for after a market high, and on a rising market the lowest point after a
high tends to come early. Running the identical rule on the same tape with the whole NBER
calendar shifted by every possible number of months ({h['lead_rot']['n_shifts']} shifts,
{h['lead_rot']['n_cycles']} total-return cycles) gives the honest null:

| Statistic | Real calendar | Shifted calendars (mean) | One-sided p |
|---|--:|--:|--:|
| Share of cycles with the market trough first | **{h['lead_rot_obs_share']:.0%}** | {h['lead_rot_null_share']:.0%} | **{h['lead_rot_p_share']:.3f}** |
| Mean trough lead (months) | {h['lead_rot_obs_mean']:.1f} | {h['lead_rot_null_mean']:.1f} | {h['lead_rot_p_mean']:.2f} |

How much does that p depend on the window? The same test across nine window choices (allowed
market lag after the NBER peak × search window after the NBER trough; the pre-registered pair is
3 / 12):

| Peak slack (m) | Post-trough window (m) | Market first | Shifted calendars | p (share) | p (mean lead) |
|--:|--:|--:|--:|--:|--:|
{grid}

{h['lead_grid_n_sig']} of 9 window choices clear p < 0.05 on the share.

The *direction* survives — the market bottoms first more reliably than the rule produces on
arbitrary dates — but the *size* of the lead does not stand out: shifted calendars produce
leads at least as long. A sign test against 50% (p = {la['trough_p']:.4f}) overstates the
evidence; the calendar-rotation p is the number to quote.

The market's trough beat the official trough in {la['trough_pos']} of {la['n']} uncensored
cycles. The exception is 2001, when the market kept falling for ten months after the NBER
trough (the September 2002 low). The 1926 cycle is censored because the tape starts in July
1926, and the trough leads for the shallow cycles (1945, 1953, 1960) mark small dips rather
than bear markets, which is why the real-bear row is shown as well.

## 3. What was already done by announcement day

Peak rows: `already` = drawdown from the high of the previous 24 months, `still` = further fall
over the next 18 months, `share` = fraction of the cycle's total fall already realised.
Trough rows: `already` = rebound from the low of the window [reference − 12 m, announcement],
`since` = months since that low. Each row comes from the monthly total-return tape when that
tape covers it, otherwise from the daily price tape (2020/21).

| Kind | Announced | Lag (m) | Already | Still to come | Share done | Since extreme (m) | Next 12 m | Tape |
|---|---|--:|--:|--:|--:|--:|--:|---|
{ann}

- **Peak announcements:** median drawdown already realised **{_pct(h['dd_already_median'])}**;
  in **{h['dd_already_n_nofall']} of 6** there had been no meaningful fall at all. Median share of
  the cycle's fall already done: {_pct(h['share_done_median'], nd=0)}; median further fall
  still to come: {_pct(h['still_to_come_median'], True)}.
- **Trough announcements:** the market was already a median **{_pct(h['missed_median'], True, 0)}**
  above its low (smallest {_pct(h['missed_min'], True, 0)}), reached a median
  {h['months_since_low_median']:.0f} months earlier.

## 4. Is the announcement a buy signal? — rotation test

Statistic: mean forward return from the announcement bar's close (monthly: excess of T-bills,
total return; daily: price-only). Null: the same statistic after **every** circular shift of
the whole set of announcement dates over the tape, which keeps their spacing, their overlap and
the market's serial dependence. `p ↑` = share of shifts at least as high (the buy-signal test),
`p ↓` = share at least as low. Primary, pre-registered: pooled, monthly, 12 months.

| Events | Tape | h (m) | n | After announcements | Random dates | Excess | p ↑ | p ↓ |
|---|---|--:|--:|--:|--:|--:|--:|--:|
{rot}

**Primary result:** {_pct(h['buy_obs_pooled'], True)} after the {h['n_events_monthly']}
announcements against {_pct(h['buy_null_pooled'], True)} after random dates,
**p = {h['buy_p_pooled']:.2f}**. Peaks alone: {_pct(h['buy_obs_peak'], True)} (p =
{h['buy_p_peak']:.2f}); troughs alone: {_pct(h['buy_obs_trough'], True)} (p =
{h['buy_p_trough']:.2f}, lower tail p = {h['buy_p_trough_lower']:.2f}). Daily price tape
({h['n_events_daily']} events): p = {h['buy_p_pooled_daily']:.2f}. Postwar-only null
(1947 →): p = {h['buy_p_pooled_postwar']:.2f}.

One pattern is worth naming without leaning on it: after **trough** announcements the next
12 months were *below* random dates (lower-tail p = {h['buy_p_trough_lower']:.2f} on the full
null, {next(r['p_lower'] for r in h['rotation'] if r['events'] == 'trough' and r['tape'] == 'postwar'):.3f}
on the postwar null). That is consistent with "the recovery is long gone" — the easy rebound was
over — but it is one tail of one subgroup out of the 27 rows above, five events, and it does not
replicate on the daily tape. It is a hypothesis for the next study, not a finding.

## 5. Power — could twelve events have seen it?

Synthetic worlds matched to the tape (i.i.d. monthly returns at the tape's
{h['power_vol']:.1%} volatility, the same length), with an abnormal log return of the stated
size planted over the 12 months after every announcement; {POWER_SIMS} worlds per row; the same
rotation test at one-sided 5%. Synthetic — this is a property of the test, not of the market.

| Planted 12-month abnormal return | Rejection rate, 10 events | Rejection rate, 6 events |
|--:|--:|--:|
{pw}

With 10 events the test reaches 80% power only at a planted **{_pct(h['mde80'], nd=0)}**
abnormal 12-month return; with 6 events (one announcement kind) it needs
{_pct(h['mde80_6'], nd=0)}. Size at zero effect: {_pct(h['size_at_null'], nd=0)}. Anything
subtler than a double-digit annual edge is invisible to this sample, whatever the truth.

## 6. Could you trade it?

One execution lag (an announcement during bar *t* is acted on at *t*'s close and earns *t+1*
onward); every switch trades 100% of NAV, charged {h['cost_bps']:.0f} bp one-way. Sharpe on
returns **in excess of bills** for every leg. Active-return HAC t: Newey-West, 12 lags monthly,
21 daily.

**Monthly, Fama-French total return vs T-bills, 1980-01 → 2018-11**

| Rule | In equity | Switches | Excess ret gross | Excess ret net | Sharpe gross | Sharpe net | Δ Sharpe vs B&H (net) | Active HAC t | Max DD (net) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{rules(h['rules_monthly'])}

**Daily, S&P 500 price index vs 0% cash, 1990-01 → 2022-11** (price-only on both sides)

| Rule | In equity | Switches | Excess ret gross | Excess ret net | Sharpe gross | Sharpe net | Δ Sharpe vs B&H (net) | Active HAC t | Max DD (net) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{rules(h['rules_daily'])}

**Official dates vs the same rule on shifted dates** (identical exposure, switches and costs;
only the timing moves; one-sided p = share of shifts with a Sharpe at least as high):

| Rule | Sharpe (monthly) | Shifted mean | p | Sharpe (daily) | Shifted mean | p |
|---|--:|--:|--:|--:|--:|--:|
{perm}

**Cost sweep, monthly** (net excess Sharpe; Δ vs buy-and-hold):

| One-way cost (bp) | "12 m after each" Sharpe | B&H Sharpe | Δ, buy-after rule | Δ, recession-avoider rule |
|--:|--:|--:|--:|--:|
{sw}

The rules lose to buy-and-hold before costs, so no cost level rescues them and a break-even
cost does not exist. Capacity is not the binding constraint either: these are index switches a
few times a decade. The problem is timing.

## 7. Synthetic control — a machinery proof, not market evidence

The same pipeline on the synthetic world with the effects planted (`signal_strength = 1`) and
switched off (`0`), 16 cycles, 32 announcements:

| signal_strength | Planted lead (m) | Measured median trough lead | Sign-test p | Calendar-rotation p (market first) | Planted premium | Measured excess | Rotation p |
|--:|--:|--:|--:|--:|--:|--:|--:|
{ctl}

The lead estimator recovers the planted lead and the rotation test fires on the planted premium.
With both switched off, the raw median "lead" still reads a few months — that is the
estimator's own lean, the reason section 2 judges the lead against shifted calendars — while
the calendar-rotation test and the announcement test are both quiet.

## Caveats

- **n is tiny and that is the main finding about inference.** Ten announcements on the
  total-return tape, eight on the daily tape, twelve in all; six of each kind. Section 5 shows
  the test can only see a double-digit edge. "Not significant" here means "not large", not
  "zero".
- **Turning points on a total-return index.** The Fama-French series includes dividends, which
  were 4–7% a year before the 1960s. Total-return peaks come a little later and troughs a
  little earlier than price-index peaks and troughs; the leads are measured on what an investor
  actually earned.
- **The 2020 cycle and its announcements are price-only** (skfolio S&P 500). They are never
  pooled with the total-return tape inside one test; the daily rows are a separate check.
- **Announcement timing within the day.** The committee has released at different times of day.
  Acting at the announcement day's close (daily) or month-end (monthly) is the documented one
  lag; the monthly tape gives up the rest of the announcement month, which is conservative
  for the 1 December 2008 release (the S&P 500 fell about 9% that day).
- **Window choices are fixed, not tuned:** 24 months of look-back for the market peak, 3 months
  of allowed lag after the NBER peak, 12 months after the NBER trough for the market trough;
  12 months is the pre-registered forward horizon and 3, 6 and 24 are shown beside it.
- **The rotation null wraps around the end of the tape** (a circular shift), so a handful of
  shifted events sit in the early sample. The postwar-only null is shown as a check.
- **NBER dates are final, not real-time.** The committee does not revise its dates often, but a
  real-time investor in 1980 did not know the 2008 announcement lag would be twelve months.

## Verdict

Produced by `strategy.verdict`, with thresholds fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1013-recession-announcement](../README.md). Not investment advice.*
"""


def main() -> None:
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    slim = {k: v for k, v in h.items()
            if not isinstance(v, (list, dict)) or k == "_verdict"}
    print("##HEADLINE## " + json.dumps(slim, default=str))


if __name__ == "__main__":
    main()
