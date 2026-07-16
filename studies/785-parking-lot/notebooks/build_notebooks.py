"""Generate the two narrative notebooks for Study 785 (Parking-Lot).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached WMT/SPY tapes under
../_cache/ (fetching once on a cache miss) and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The parking signal is a LABELLED PROXY (see data.py), used
ordinally. The synthetic positive control runs anywhere with no network.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers -- mirror of docs/results.md (WMT + SPY, yfinance,
# 2008-01-02 -> 2026-06-30; 64 of 64 parking quarters resolved). Fingerprint eb607d15a2de.
R = dict(
    n_events=64, n_included=64, n_busy=41, n_slow=23, fp="eb607d15a2de", rows=4652,
    busy_s_mean=+0.080, busy_s_t=+0.137, busy_s_hit=21,
    busy_l_mean=-0.481, busy_l_t=-0.564, busy_l_hit=18,
    slow_s_mean=-0.612, slow_s_t=-0.643, slow_l_mean=+0.023, slow_l_t=+0.014,
    ls_s_mean=+0.271, ls_s_t=+0.539, ls_l_mean=-0.316, ls_l_t=-0.391,
    welch_s=+0.692, welch_s_t=+0.620, welch_l=-0.504, welch_l_t=-0.266,
    spearman_s=-0.024, spearman_l=-0.041,
    pl_s_p=0.295, pl_s_mean=-0.005, pl_s_sd=0.499,
    pl_l_p=0.651, pl_l_mean=+0.002, pl_l_sd=0.796,
    jk_lo=-0.695, jk_hi=+0.204,
    ls_s_net5=+0.243, ls_s_t5=+0.48, ls_l_net5=-0.344, ls_l_t5=-0.43,
    null_mean_t=+0.23, null_sd_t=1.17, null_hits=1,
    planted2_mean=+2.634, planted2_t=+3.43, planted4_mean=+4.539, planted4_t=+5.92,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from parking_lot import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching WMT + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
bz, sl = st.busy(ev), st.slow(ev)
print(f"panel loaded; {len(inc)} of {len(ev)} parking quarters resolved ({len(bz)} busy, {len(sl)} slow)")
"""


def build_curious():
    cells = [
        md("# Study 785 — Parking-Lot 🛰️🅿️\n\n"
           "*For the curious.* The alt-data pitch that launched a hundred hedge-fund decks: "
           "**count the cars in Walmart's parking lots from space and you can beat the earnings "
           "print** — go long the busy quarters, short the empty ones. We put 64 WMT quarters "
           "(2010→2025) on the stand and asked the tape one plain question: **does a busier lot "
           "mean the stock drifts up after the print?** The answer is a clean *no*."),
        md("> ⚠️ **The parking signal here is a LABELLED PROXY, not real satellite data.** Genuine "
           "orbital car-count panels are paywalled and non-redistributable, so we use a "
           "hardcoded, deterministic, *stylised* quarterly foot-traffic index — used **ordinally "
           "only** (busier vs emptier lots year-over-year). This tests the *method and the "
           "proxy*, not a live feed. See `data.py`."),
        md("## The claim, and why it's a clean test\n\n"
           "The satellite 'verdict' for a quarter is known *before* the print, and WMT's "
           "reporting cadence (mid/late Feb, mid-May, mid-Aug, mid-Nov) is known years ahead — so "
           "*buy busy / sell slow at the print, hold K sessions* is calendar-known and "
           "zero-look-ahead. We measure WMT's **abnormal** forward return (WMT − SPY, "
           "total-return) so we're not just measuring beta."),
        code(PRELUDE),
        md("## The LABELLED-PROXY parking table we test (stylised, ordinal use only)"),
        code("dt.parking_events().head(12)"),
        md("## The picture: mean cumulative abnormal return after the print\n\n"
           "Offset 0 is the print anchor. If the folklore held, the **busy** line would climb and "
           "the **slow** line would sink. They don't."),
        code(
            "car_b = st.car_path(ev, prices, subset='busy')\n"
            "car_s = st.car_path(ev, prices, subset='slow')\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car_b.index, car_b.values * 100, color='#2ea44f', lw=2, label='busy quarters')\n"
            "ax.plot(car_s.index, car_s.values * 100, color='#c0392b', lw=2, label='slow quarters')\n"
            "ax.set_xlabel('trading days after the print (0 = anchor)')\n"
            "ax.set_ylabel('mean cumulative AR, WMT − SPY (%)')\n"
            "ax.set_title('Busy vs slow parking quarters drift the same: nowhere')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| subset | window | mean AR | *t* |\n|---|---|--:|--:|\n"
           f"| busy | 1-week | **{R['busy_s_mean']:+.2f}%** | {R['busy_s_t']:+.2f} |\n"
           f"| busy | 1-month | {R['busy_l_mean']:+.2f}% | {R['busy_l_t']:+.2f} |\n"
           f"| slow | 1-week | {R['slow_s_mean']:+.2f}% | {R['slow_s_t']:+.2f} |\n"
           f"| slow | 1-month | {R['slow_l_mean']:+.2f}% | {R['slow_l_t']:+.2f} |\n\n"
           f"Every cell is a coin-flip (|*t*| < 0.7). The busy-minus-slow long/short spread is "
           f"{R['ls_s_mean']:+.2f}% (1-week, *t* {R['ls_s_t']:+.2f}) and {R['ls_l_mean']:+.2f}% "
           f"(1-month, *t* {R['ls_l_t']:+.2f}) — and it **flips sign** between the two horizons. "
           f"The rank correlation between the parking YoY and the forward return is ≈ 0 "
           f"({R['spearman_l']:+.2f})."),
        code(
            "for label, col in [('busy 1wk','fwd_s'),('busy 1mo','fwd_l')]:\n"
            "    s = st.one_sample_t(bz[col].values); print(f'{label:<9s} mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}')\n"
            "for label, col in [('slow 1wk','fwd_s'),('slow 1mo','fwd_l')]:\n"
            "    s = st.one_sample_t(sl[col].values); print(f'{label:<9s} mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}')\n"
            "for label, col in [('L/S 1wk','fwd_s'),('L/S 1mo','fwd_l')]:\n"
            "    ls = st.longshort_returns(ev, col); s = st.one_sample_t(ls)\n"
            "    print(f'{label:<9s} mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## So what?\n\n"
           "The famous *count-the-cars* trade — at least via a stylised, public, ordinal proxy "
           "entered at the print — does **not** beat the earnings print. Verdict: **None signal, "
           "Mirage tradability.** That squares with the academic finding (Katona et al.) that any "
           "*real* satellite edge accrues **early, pre-announcement, to the paying subscribers** "
           "and decays as the data diffuses — there's no post-print drift left for a latecomer "
           "with a public proxy. The quants' notebook has the placebo, the jackknife, the costed "
           "leg and the synthetic control that proves the detector isn't just broken."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 785 — Parking-Lot — for the quants 🔬\n\n"
           "The full battery: one-sample *t* of the long/short timing P&L, a Welch busy-vs-slow "
           "spread, a Spearman rank check, a sign-shuffle placebo, a leave-one-out jackknife, the "
           "costed net leg, and a seeded synthetic positive control. Everything offline once "
           f"cached; fingerprint `{R['fp']}`. Signal = **LABELLED PROXY**, ordinal use only."),
        code(PRELUDE),
        md("## 1. One-sample *t* of the long/short timing P&L\n\n"
           "Each parking quarter is one independent event: the long/short return is `+fwd` on "
           "busy quarters, `−fwd` on slow ones. Its mean is the busy-minus-slow timing P&L — the "
           "unit is a one-sample *t*, **not** a daily panel."),
        code(
            "for label, col in [('L/S 1wk','fwd_s'),('L/S 1mo','fwd_l')]:\n"
            "    ls = st.longshort_returns(ev, col); s = st.one_sample_t(ls)\n"
            "    r2 = st.two_sample_t(bz[col].values, sl[col].values)\n"
            "    rho = st.spearman(inc['yoy'].values, inc[col].values)\n"
            "    print(f'{label:<8s} n={s[\"n\"]}  L/S mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  '\n"
            "          f'Welch busy-slow={r2[\"diff\"]*100:+.3f}% (t={r2[\"t\"]:+.3f})  spearman={rho:+.3f}')"
        ),
        md("## 2. Sign-shuffle placebo — is the spread inside the luck cloud?\n\n"
           "Hold the forward returns fixed and randomly relabel which quarters were 'busy' "
           "(independent ±1 flips); 40 seeds × 250 draws. If the observed busy-minus-slow mean "
           "sits in the tail of that null, the parking labels carry information beyond luck."),
        code(
            "pl = st.placebo_pvalue(ev, 'fwd_s', tail='right')\n"
            "n = int((inc['direction']!='flat').sum())\n"
            "fwd = inc[inc['direction']!='flat']['fwd_s'].values.astype(float)\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(6000):\n"
            "    w = rng.choice((-1.0,1.0), size=n); draws.append(float(np.mean(w*fwd)))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#2ea44f', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 1-week long/short return of random label-shuffles (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"busy-minus-slow vs luck cloud (right-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is any single quarter holding it up?"),
        code(
            "x = st.longshort_returns(ev, 'fwd_l')\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — net of costs\n\n"
           "Calendar-known entry at the print, so the signal window and the tradable window are "
           "the same. There is no gross edge to erode, so costs are almost irrelevant."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0)\n"
            "for base, label in [('fwd_s','1wk L/S'),('fwd_l','1mo L/S')]:\n"
            "    g = st.one_sample_t(st.longshort_returns(ev, base))\n"
            "    n5 = st.one_sample_t(st.longshort_returns(ev, base+'_net'))\n"
            "    n10 = st.one_sample_t(st.longshort_returns(ev10, base+'_net'))\n"
            "    print(f'{label:<8s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works, the real tape is just empty\n\n"
           "The long/short detector must stay quiet on a planted-null world and recover a planted "
           "two-sided 'busy→forward-up / slow→forward-down' link. That it fires cleanly on the "
           "synthetic world but not on WMT is what makes the real-tape null *trustworthy*, not a "
           "broken pipe."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=810+s, k=21)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.02, 0.04):\n"
            "    r = st.synthetic_detect(bump=b, seed=810, k=21)\n"
            "    print(f'planted +{b*100:.0f}%: mean L/S {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.05, 11)\n"
            "ts = [st.synthetic_detect(bump=b, seed=810, k=21)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted busy/slow link (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted parking link is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: None.** Every cut is a coin-flip: busy/slow forward AR |*t*| < 0.7, "
           "long/short spread |*t*| < 0.55 that **flips sign** across horizons, Spearman(yoy, fwd) "
           "≈ 0, sign-shuffle placebo p = 0.29 / 0.65, jackknife *t* straddling zero. The "
           "synthetic control fires (*t* = +3.4 at a +2% planted link), so this is a true null, "
           "not a broken test. **Tradability: Mirage.** No gross edge to erode, sign-unstable, and "
           "the signal is a LABELLED PROXY — not the real, paywalled satellite panel — so even a "
           "hypothetical edge here would not be a live, investable feed."),
    ]
    return new_notebook(cells=cells)


def main():
    for name, nb in (("01_for_the_curious", build_curious()),
                     ("02_for_the_quants", build_quants())):
        path = os.path.join(HERE, f"{name}.ipynb")
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                     "language": "python"}
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print("wrote", path)


if __name__ == "__main__":
    main()
