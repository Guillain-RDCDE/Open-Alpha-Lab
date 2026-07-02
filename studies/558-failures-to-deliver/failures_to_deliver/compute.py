"""Compute the headline numbers for Study 558 (offline, deterministic).

Run: ``python -m failures_to_deliver.compute``

There is no free FTD tape (see data.py), so the reproducible *headline* is the deterministic
seed-558 **null** synthetic world (folklore off): with FTD spikes carrying no forward information,
does the honest event-study engine correctly report *nothing*? The placebo, the robustness sweep
and the seed-robust positive control complete the picture.
"""
from __future__ import annotations

import json

import numpy as np

from . import data, strategy as st


def main():
    out = {}

    # --- Headline: the NULL synthetic world (folklore off), seed 558 ---
    null_panel, _ = data.synthetic_panel(squeeze_alpha=0.0, seed=558)
    es0 = st.event_study(null_panel, horizon=5, spike_z=2.5)
    p0 = st.placebo_pvalue(null_panel, horizon=5, spike_z=2.5, n_perm=1000, seed=558)
    out["fp_null"] = data.fingerprint(null_panel)
    out["null_n_events"] = es0["n_events"]
    out["null_mean_fwd_pct"] = round(es0["mean_fwd"] * 100, 3)
    out["null_baseline_pct"] = round(es0["baseline_mean"] * 100, 3)
    out["null_abn_pct"] = round(es0["excess"] * 100, 3)
    out["null_t"] = round(es0["t"], 3)
    out["null_placebo_p"] = round(p0, 3)
    out["null_net_pct"] = round(st.net_event_return(es0["excess"], 5.0, 300.0, 5) * 100, 3)

    # --- Robustness sweep on the null world (should be sign-unstable / centred on 0) ---
    sweep = st.robustness_sweep(null_panel)
    out["sweep_null"] = [
        {"h": int(r.horizon), "z": float(r.spike_z), "n": int(r.n_events),
         "abn_pct": round(float(r.excess_pct), 3), "t": round(float(r.t), 3)}
        for r in sweep.itertuples()
    ]
    out["sweep_null_t_range"] = [round(float(sweep["t"].min()), 3), round(float(sweep["t"].max()), 3)]

    # --- Seed-robust synthetic positive control (25 seeds) ---
    ctrl = []
    for a in [0.0, 0.002, 0.004, 0.006, 0.010]:
        mt = st.synthetic_mean_t(a, n_seeds=25)
        ctrl.append({"alpha": a, "mean_t_25seeds": round(mt, 3)})
    out["control"] = ctrl

    # --- Null across 30 seeds: the folklore looks significant on some cuts by luck ---
    null_ts = []
    for s in range(558, 588):
        pn, _ = data.synthetic_panel(squeeze_alpha=0.0, seed=s)
        null_ts.append(st.event_study(pn)["t"])
    null_ts = np.array(null_ts)
    out["null_30seed_mean_t"] = round(float(null_ts.mean()), 3)
    out["null_30seed_sd_t"] = round(float(null_ts.std(ddof=1)), 3)
    out["null_30seed_frac_gt2"] = round(float(np.mean(np.abs(null_ts) > 2)), 3)

    # --- A demonstrative planted world (folklore ON) for the notebook narrative ---
    on_panel, _ = data.synthetic_panel(squeeze_alpha=0.006, seed=558)
    es_on = st.event_study(on_panel, horizon=5, spike_z=2.5)
    p_on = st.placebo_pvalue(on_panel, horizon=5, spike_z=2.5, n_perm=1000, seed=558)
    out["on_alpha"] = 0.006
    out["on_n_events"] = es_on["n_events"]
    out["on_abn_pct"] = round(es_on["excess"] * 100, 3)
    out["on_t"] = round(es_on["t"], 3)
    out["on_placebo_p"] = round(p_on, 3)

    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
