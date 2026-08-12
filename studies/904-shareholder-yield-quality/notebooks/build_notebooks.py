"""Generate the two narrative notebooks for Study 904 (Shareholder-Yield + Quality).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast
synthetic positive control, so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return,
# QSY=PKW+QUAL vs RAW=PKW vs SPY, excess of BIL; common window 2013-08 -> 2026-06, 155 mo).
R = dict(
    start="2013-08", end="2026-06", n_months=155, fingerprint="6ab3e873ef9c",
    qsy_cagr=12.96, qsy_vol=15.4, qsy_sharpe=0.764, qsy_maxdd=-25.1, qsy_wealth=4.83,
    raw_cagr=12.02, raw_vol=17.0, raw_sharpe=0.657, raw_maxdd=-29.3, raw_wealth=4.33,
    spy_cagr=14.14, spy_vol=14.5, spy_sharpe=0.874, spy_maxdd=-23.9, spy_wealth=5.52,
    # Race 2 — QSY vs RAW (does the quality overlay add value over raw buybacks?)
    qr_gap=0.106, qr_diff_bps=4.94, qr_diff_ann=0.59, qr_t1s=0.55, qr_tnw=0.57,
    qr_ci_lo=-0.013, qr_ci_hi=0.241, qr_pneg=0.04,
    # Race 1 — QSY vs SPY (does quality-screened shareholder yield beat the market?)
    qs_gap=-0.111, qs_diff_ann=-0.91, qs_diff_bps=-7.62, qs_t1s=-0.79, qs_tnw=-0.90,
    qs_ci_lo=-0.243, qs_ci_hi=0.013, qs_pneg=0.96,
    # RAW vs SPY
    rs_gap=-0.217, rs_diff_ann=-1.51, rs_tnw=-0.79,
    # Era cut — QSY vs RAW
    qr_e_gap=0.163, qr_e_diff=0.92, qr_e_t=1.12, qr_e_n=77,
    qr_l_gap=0.073, qr_l_diff=0.27, qr_l_t=0.14, qr_l_n=78,
    # Era cut — QSY vs SPY
    qs_e_gap=-0.112, qs_e_diff=-0.45, qs_e_t=-0.55, qs_e_n=77,
    qs_l_gap=-0.117, qs_l_diff=-1.37, qs_l_t=-0.72, qs_l_n=78,
    # Costs
    qsy_turn=0.76, qsy_drag=0.3, qsy_net=0.763,
    raw_turn=0.32, raw_drag=0.1, raw_net=0.657,
    # SPYD context (raw dividend yield), from 2015-11
    spyd_n=128, spyd_qsy_sh=0.716, spyd_sh=0.476, spyd_gap=0.240, spyd_t=0.86,
    # Synthetic control
    null_fire=1, null_seed_t=-0.85, planted_gap=20.46, planted_t=168.4,
    # calendar-year total returns (%), QSY / RAW / SPY
    cal_years=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    cal_qsy=[-8.1, 34.0, 12.8, 29.8, -15.4, 24.1, 20.0, 15.3],
    cal_raw=[-10.5, 34.1, 8.4, 32.6, -10.2, 17.2, 17.3, 17.9],
    cal_spy=[-4.6, 31.2, 18.3, 28.7, -18.2, 26.2, 24.9, 17.7],
)


HEADER = f"""# Study 904 — Shareholder-Yield + Quality 💰

**Do buybacks pay only when they're *real* — and does a quality screen keep the real ones?**

A raw buyback / shareholder-yield screen (PKW: any firm that cut net shares ≥5% last
year) scoops up every serial repurchaser — including the ones running **dilution
theatre** (buybacks that only mop up option grants, net share count flat) or buying
back stock at rich prices. The pitch: overlay a **quality** screen (QUAL: high ROE,
stable earnings, low leverage) and you keep the *funded, value-accretive* buyers and drop
the theatre — so a **quality-screened shareholder-yield** sleeve should beat both raw
buybacks and the plain market. We race **QSY = PKW + QUAL** against **RAW = PKW** and
against **SPY**, all on monthly total returns, all **excess of cash** (minus BIL), over
the common window {R['start']} → {R['end']} ({R['n_months']} months, QUAL-bound).

*Numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fingerprint']}`); the live cells run the fast synthetic control. Short-history /
survivor caveat: QUAL lists 2013-07, so one mostly-bull regime with two drawdowns.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one line\n\n"
           "A company can 'return cash to shareholders' by genuinely shrinking its share "
           "count with real free cash flow — or it can announce a buyback that barely "
           "offsets the stock it prints for executives. A raw buyback screen can't tell "
           "the difference; a **quality** overlay (durable ROE, low accruals) is supposed "
           "to. If the 'real-buybacks-not-theatre' story is true, the quality-screened "
           "sleeve should ride *smoother* than raw buybacks — especially when the market "
           "cracks and the theatre gets exposed."),
        code(
            "R = %r\n"
            "print('QSY (PKW+QUAL): CAGR %%5.2f%%%%  exSharpe %%.3f  maxDD %%.1f%%%%'\n"
            "      %% (R['qsy_cagr'], R['qsy_sharpe'], R['qsy_maxdd']))\n"
            "print('RAW (PKW)     : CAGR %%5.2f%%%%  exSharpe %%.3f  maxDD %%.1f%%%%'\n"
            "      %% (R['raw_cagr'], R['raw_sharpe'], R['raw_maxdd']))\n"
            "print('SPY           : CAGR %%5.2f%%%%  exSharpe %%.3f  maxDD %%.1f%%%%'\n"
            "      %% (R['spy_cagr'], R['spy_sharpe'], R['spy_maxdd']))" % (R,)
        ),
        md("## 2. Where the quality overlay actually helped — the crash tape\n\n"
           "The overlay's whole promise is a smoother ride when the market breaks. The "
           "cleanest tell is **2020 (COVID)**: raw buyback names carried more junk and "
           "fell harder; the quality screen cushioned it. Note the honest twist in "
           "**2022** — the rate shock rewarded raw buyback's cheaper value tilt, so the "
           "overlay is *not* a free lunch every year:"),
        code(
            "for yr, q, r, s in zip(R['cal_years'], R['cal_qsy'], R['cal_raw'], R['cal_spy']):\n"
            "    tag = ''\n"
            "    if q > r + 3: tag = '  <-- quality overlay cushioned'\n"
            "    elif r > q + 3: tag = '  <-- raw buyback (value) won'\n"
            "    print(f'{yr}:  QSY {q:+6.1f}%   RAW {r:+6.1f}%   SPY {s:+6.1f}%{tag}')"
        ),
        md("In **2020** QSY returned **+12.8%** vs raw buyback's **+8.4%**, and over the "
           "COVID crash the quality overlay held its worst drawdown to **−25.1%** vs raw "
           "buyback's **−29.3%** — a real ~4-point cushion. *But* in the 2022 rate shock "
           "raw buyback's value tilt (**−10.2%**) actually beat quality (**−15.4%**). The "
           "overlay cleans up the crash, not every down year."),
        md("## 3. Is it just luck? A live synthetic control\n\n"
           "We plant a real quality-over-raw edge in a seeded toy world (`edge>0`) and "
           "check the detector recovers it — and stays *silent* on the null (`edge=0`, no "
           "advantage). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from sy_quality import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_world(n_months=150, edge=0.0, seed=904))\n"
            "planted = st.synthetic_detect(data.synthetic_world(n_months=150, edge=3.0, seed=904))\n"
            "print('null world   : gap NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: gap NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md(f"## 4. The honest verdict\n\n"
           f"On the real tape the quality overlay genuinely **improves raw buybacks** — a "
           f"**{R['qsy_maxdd']:.1f}%** vs **{R['raw_maxdd']:.1f}%** shallower crash and a "
           f"Sharpe gap of **+{R['qr_gap']:.3f}** (positive in *both* eras, bootstrap "
           f"96% positive) — so the 'real buybacks, not dilution theatre' story has real "
           f"signal in the *right* direction. **But** the edge never clears the bar (HAC "
           f"*t* = **+{R['qr_tnw']:.2f}**, no single era significant), and — the punchline "
           f"— **neither buyback sleeve beat plain SPY**: QSY trailed the market by "
           f"**{R['qs_diff_ann']:.2f} pp/yr** (*t* = {R['qs_tnw']:+.2f}), raw buyback by "
           f"**{R['rs_diff_ann']:.2f} pp/yr**. **Signal: Weak** (a real but uncertified "
           f"quality-cleanup of raw buybacks, and no market-beat), **Tradability: "
           f"Fragile** (cheaply buyable, but a risk-profile tweak, not a bankable premium)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 904 — Shareholder-Yield + Quality — the teardown\n\n"
           "The excess-of-cash Sharpe race, the QSY-minus-RAW and QSY-minus-SPY HAC *t*, "
           "the paired block-bootstrap Sharpe-gap CIs, the era cut, the costed sleeves, "
           "and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The race — common window, excess of cash (minus BIL)"),
        code(
            "for name in ('qsy','raw','spy'):\n"
            "    lbl = {'qsy':'QSY (PKW+QUAL)','raw':'RAW (PKW)     ','spy':'SPY           '}[name]\n"
            "    print(f\"{lbl}: CAGR {R[name+'_cagr']:5.2f}%  vol {R[name+'_vol']:4.1f}%  \"\n"
            "          f\"exSharpe {R[name+'_sharpe']:.3f}  maxDD {R[name+'_maxdd']:6.1f}%  \"\n"
            "          f\"$1->{R[name+'_wealth']:.2f}\")"
        ),
        md("## Race 2 — does the quality overlay add value over raw buybacks? (QSY − RAW)\n\n"
           "The QSY−RAW monthly spread is cash-independent (cash cancels), so its mean / "
           "HAC *t* is the clean 'does the overlay out-earn raw buybacks?' statistic."),
        code(
            "print(f\"excess Sharpe : QSY {R['qsy_sharpe']:.3f}  vs RAW {R['raw_sharpe']:.3f}  \"\n"
            "      f\"-> GAP {R['qr_gap']:+.3f}\")\n"
            "print(f\"QSY-minus-RAW : {R['qr_diff_bps']:+.2f} bps/mo ({R['qr_diff_ann']:+.2f}%/yr)  \"\n"
            "      f\"one-sample t {R['qr_t1s']:+.2f}  NW t {R['qr_tnw']:+.2f}\")\n"
            "print(f\"bootstrap gap : {R['qr_gap']:+.3f}  95% CI [{R['qr_ci_lo']:+.3f}, {R['qr_ci_hi']:+.3f}]  \"\n"
            "      f\"P(gap<0) = {R['qr_pneg']:.2f}\")"
        ),
        md("The overlay's advantage is **positive and the bootstrap is 96% positive**, but "
           "the HAC *t* is only **+0.57** — a real direction, not a certified premium."),
        md("## Race 1 — does quality-screened shareholder yield beat the market? (QSY − SPY)"),
        code(
            "print(f\"QSY vs SPY: gap {R['qs_gap']:+.3f}  diff {R['qs_diff_ann']:+.2f}%/yr  \"\n"
            "      f\"NW t {R['qs_tnw']:+.2f}  95% CI [{R['qs_ci_lo']:+.3f}, {R['qs_ci_hi']:+.3f}]  P(gap<0)={R['qs_pneg']:.2f}\")\n"
            "print(f\"RAW vs SPY: gap {R['rs_gap']:+.3f}  diff {R['rs_diff_ann']:+.2f}%/yr  NW t {R['rs_tnw']:+.2f}\")"
        ),
        md("**Both buyback sleeves trailed SPY** — the market-beat claim is the wrong sign "
           "(insignificantly). Owning plain SPY beat owning either buyback wrapper."),
        md("## Era cut (split 2020-01-01) — QSY vs RAW is positive in both halves, never significant"),
        code(
            "print('QSY - RAW:')\n"
            "print(f\"  2013-08..2019-12 (n={R['qr_e_n']}): gap {R['qr_e_gap']:+.3f}  diff {R['qr_e_diff']:+.2f}%/yr  NW t {R['qr_e_t']:+.2f}\")\n"
            "print(f\"  2020-01..2026-06 (n={R['qr_l_n']}): gap {R['qr_l_gap']:+.3f}  diff {R['qr_l_diff']:+.2f}%/yr  NW t {R['qr_l_t']:+.2f}\")\n"
            "print('QSY - SPY:')\n"
            "print(f\"  2013-08..2019-12 (n={R['qs_e_n']}): gap {R['qs_e_gap']:+.3f}  diff {R['qs_e_diff']:+.2f}%/yr  NW t {R['qs_e_t']:+.2f}\")\n"
            "print(f\"  2020-01..2026-06 (n={R['qs_l_n']}): gap {R['qs_l_gap']:+.3f}  diff {R['qs_l_diff']:+.2f}%/yr  NW t {R['qs_l_t']:+.2f}\")"
        ),
        md("## Costed — monthly rebalance turnover x one-way spread (long-only, no borrow)\n\n"
           "Turnover is only the drift of PKW/QUAL back to 50/50 (~0.8%/mo); raw buyback "
           "is a single ETF (no rebalance). Costs are a rounding error — not the story."),
        code(
            "print(f\"QSY: turnover {R['qsy_turn']:.2f}%/mo  drag {R['qsy_drag']:.1f} bps/yr  \"\n"
            "      f\"gross exSharpe {R['qsy_sharpe']:.3f} -> net {R['qsy_net']:.3f}\")\n"
            "print(f\"RAW: turnover {R['raw_turn']:.2f}%/mo  drag {R['raw_drag']:.1f} bps/yr  \"\n"
            "      f\"gross exSharpe {R['raw_sharpe']:.3f} -> net {R['raw_net']:.3f}\")"
        ),
        md("## Context — raw dividend yield (SPYD) and the too-young BUYB"),
        code(
            "print(f\"QSY vs SPYD (raw dividend yield, from 2015-11, n={R['spyd_n']}): \"\n"
            "      f\"exSharpe {R['spyd_qsy_sh']:.3f} vs {R['spyd_sh']:.3f}  gap {R['spyd_gap']:+.3f}  NW t {R['spyd_t']:+.2f}\")\n"
            "print('BUYB (standalone buyback ETF) lists 2026-05 -> 66 days: too young to race; named only')"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the gap detector must NOT fire on the null and must recover a planted "
           "quality-over-raw premium."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from sy_quality import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_world(n_months=150, edge=0.0, seed=904+s))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_world(n_months=150, edge=3.0, seed=904))\n"
            "print(f\"planted (edge=+3%/yr): gap {planted['sharpe_gap']:+.3f}  NW t {planted['t_nw']:+.2f}  diff {planted['diff_ann_pct']:+.2f}%/yr\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The quality overlay genuinely improves raw buybacks: a "
           f"**{R['qsy_maxdd']:.1f}%** vs **{R['raw_maxdd']:.1f}%** shallower crash and a "
           f"Sharpe gap of **+{R['qr_gap']:.3f}**, positive in *both* eras (gaps "
           f"+{R['qr_e_gap']:.3f} / +{R['qr_l_gap']:.3f}) with the bootstrap 96% positive "
           f"(CI **[{R['qr_ci_lo']:+.3f}, {R['qr_ci_hi']:+.3f}]**, P(gap<0)={R['qr_pneg']:.2f}). "
           f"But it never clears the HAC bar (NW *t* = **+{R['qr_tnw']:.2f}**, no era "
           f"significant), and — decisively — **neither buyback sleeve beats plain SPY** "
           f"(QSY {R['qs_diff_ann']:.2f} pp/yr, *t* = {R['qs_tnw']:+.2f}; RAW "
           f"{R['rs_diff_ann']:.2f} pp/yr). The synthetic control recovers a *planted* edge "
           f"cleanly (*t* = {R['planted_t']:.0f}, fires on {R['null_fire']}/20 nulls), so a "
           f"real premium *would* have shown — the market-beat one didn't. Short "
           f"single-regime tape.\n"
           f"- **Tradability — Fragile.** The overlay is trivially buyable (cheap, liquid, "
           f"long-only, rebalance drag {R['qsy_drag']:.1f} bps/yr — nothing erases it, not a "
           f"Mirage) — but there is no significant premium to bank: you buy a "
           f"shallower-drawdown *cleanup* of raw buybacks, not a certified edge, and the "
           f"whole complex trails the market. Fragile is the honest stamp."),
    ]
    nb["cells"] = cells
    return nb


def main():
    for name, nb in [("01_for_the_curious", build_curious()),
                     ("02_for_the_quants", build_quants())]:
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)


if __name__ == "__main__":
    main()
