"""Generate the two narrative notebooks for Study 842 (Implementation Shortfall).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. This is a synthetic-only method demo (a real
tape can't certify a clean planted edge), so the study is capped at NONE. The frozen headline
numbers live in the ``R`` dict below (mirroring docs/results.md); the heavy cells quote ``R``,
and the live cells run only the fast synthetic control (a few seconds). Path bootstrap in the
live cells: sys.path.insert for '..' and '../../..'.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; planted panel
# fp 79082a088002; 30 names x 2,520 days; edge=0.0005, phi=0.96, frac=0.2, seed=842).
R = dict(
    fp="79082a088002", null_fp="87551053f9a3", as_of="2026-06-30",
    n_names=30, n_days=2519, edge=0.0005, phi=0.96, frac=0.2, seed=842,
    turnover=0.348,
    gross_bps=12.25, gross_t=7.65, gross_sharpe=2.27, gross_ann=30.9,
    # cost ladder: (label, one-way, impact, gross_sharpe, net_sharpe, net_bps, cost_day, net_t, ann)
    ladder=[
        ("paper (0 cost)", 0, 0, 2.27, 2.27, 12.25, 0.00, 7.65, 30.9),
        ("optimistic", 5, 20, 2.27, 1.39, 7.51, 4.74, 4.69, 18.9),
        ("realistic", 10, 50, 2.27, 0.24, 1.27, 10.98, 0.79, 3.2),
        ("stressed", 20, 100, 2.27, -1.78, -9.71, 21.96, -5.90, -24.5),
    ],
    breakeven_linear=35.24,
    # turnover curve at realistic cost (10bp + impact 50): (phi, turnover, gross_sh, net_sh, be)
    curve=[
        (0.995, 0.152, 2.06, 1.40, 72.0),
        (0.98, 0.254, 2.15, 0.86, 45.9),
        (0.96, 0.348, 2.27, 0.24, 35.2),
        (0.90, 0.533, 2.17, -1.73, 22.4),
        (0.70, 0.893, 2.21, -6.77, 13.8),
        (0.30, 1.343, 2.46, -15.54, 10.0),
    ],
    null_mean_t=0.01, null_sd_t=0.92, null_fire=1,
    plant_mean_t=6.07, plant_sd_t=0.92, plant_fire=20,
)


BOOTSTRAP = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
)


HEADER = f"""# Study 842 — Implementation Shortfall 🧾

**The paper-vs-live cost gap: the same strategy at 0 / realistic / stressed cost.**

André Perold (1988): the frictionless *paper portfolio* and the real portfolio differ by the
cost of trading into the positions — and that gap scales with **turnover**. We take a
moderate-turnover cross-sectional long-short with a **planted, genuine gross edge** (so its
0-cost Sharpe honestly dazzles: **{R['gross_sharpe']}**, NW *t* = **+{R['gross_t']}**) and
evaluate the *identical* book across a cost ladder with a turnover-scaled market-impact term.

*Synthetic-only by design (a real tape can't certify a clean planted edge), so this is a
method demo capped at `NONE`. Numbers below are the frozen headline (`docs/results.md`,
fp `{R['fp']}`, as-of {R['as_of']}); the live cells run the fast synthetic control.*
"""


def build_curious():
    nb = new_notebook()
    ladder_lines = "\n".join(
        f"print(f'{lab:<16}: gross Sharpe {gs:.2f}  ->  net Sharpe {ns:6.2f}  "
        f"(net t = {nt:+.2f}, {ann:+.1f}%/yr)')"
        for (lab, cb, ic, gs, ns, nb_, cd, nt, ann) in R["ladder"]
    )
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "A backtest buys and sells at the *decision* price and pays nothing to trade — the "
           "**paper portfolio**. The real book pays the spread, a commission, and, worst of all, "
           "**market impact**: the more of the book you rotate, the more your own trading moves "
           "the price against you. The paper return minus the real return is the *implementation "
           "shortfall*, and for a strategy that trades a lot it can be the entire edge."),
        code(
            "R = dict(gross_sharpe=%r, gross_t=%r, turnover=%r, breakeven=%r)\n"
            "print('PAPER portfolio (0 cost): gross Sharpe %%.2f  (NW t = %%+.2f)'\n"
            "      %% (R['gross_sharpe'], R['gross_t']))\n"
            "print('  it rotates %%.0f%%%% of the book every day' %% (R['turnover']*100))\n"
            "print('  looks like a real, tradable edge... on paper')"
            % (R["gross_sharpe"], R["gross_t"], R["turnover"], R["breakeven_linear"])
        ),
        md("## 2. Now charge the cost of trading — the SAME strategy\n\n"
           "Zero cost is a fantasy. Add an *optimistic* cost, then a *realistic* one, then a "
           "*stressed* one (bigger book, thinner names). The gross Sharpe never changes — it is "
           "blind to trading — but the **net** Sharpe tells the truth."),
        code(ladder_lines),
        md(f"## 3. Why it collapses — turnover\n\n"
           f"The paper alpha is real, but you have to *trade* to capture it, and this book turns "
           f"over ~{R['turnover']*100:.0f}% of NAV a day. At a realistic cost the friction "
           f"(~11 bps/day) is as big as the gross edge (~{R['gross_bps']} bps/day), so the net "
           f"edge is gone. Trade *more* and it gets worse — the same paper alpha, evaluated at "
           f"higher turnover, becomes a disaster:"),
        code(
            "curve = %r\n"
            "print('phi   turnover/day  gross Sharpe   net Sharpe (realistic cost)')\n"
            "for phi, tu, gs, ns, be in curve:\n"
            "    flag = '  <- tradable' if ns > 0.5 else ('  <- DEAD' if ns < 0 else '')\n"
            "    print(f'{phi:<5} {tu:>8.3f}      {gs:>6.2f}       {ns:>7.2f}{flag}')"
            % (R["curve"],)
        ),
        md("## 4. Is the paper edge even real? A live synthetic control\n\n"
           "We *planted* the gross edge, so it had better show up — and it must vanish on a null "
           "world where the signal predicts nothing. Live, offline, a few seconds."),
        code(
            BOOTSTRAP +
            "from cost_gap import data, strategy as st\n"
            "null = st.synthetic_detect(data, edge=0.0, n_days=1500)\n"
            "plant = st.synthetic_detect(data, edge=0.0005, n_days=1500)\n"
            "print('null world   : gross book NW t = %+.2f  (should be ~0)' % null['gross_t'])\n"
            "print('planted world: gross book NW t = %+.2f  (should light up)' % plant['gross_t'])"
        ),
        md(f"## 5. The honest verdict\n\n"
           f"The gross edge is genuinely there — but you cannot afford to trade it. "
           f"**Signal: None** (a synthetic method demo — a real tape can't certify a planted "
           f"edge). **Tradability: Mirage** (net Sharpe {R['gross_sharpe']} → "
           f"{R['ladder'][2][4]} at a realistic cost, {R['ladder'][3][4]} when stressed). "
           f"**Does ignoring costs manufacture the edge? Confirmed** — the whole {R['gross_ann']}%/yr "
           f"paper triumph is the cost of trading that the backtest simply forgot to charge. "
           f"A backtest without a cost model is meaningless."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 842 — Implementation Shortfall — the teardown\n\n"
           "The dollar-neutral cross-sectional book, the linear + super-linear (participation) "
           "impact cost model, the cost ladder, the deceptive linear break-even, the turnover "
           "curve, HAC inference, and the 20-seed synthetic control. Synthetic-only, capped at "
           "`NONE`."),
        code("R = %r" % (R,)),
        md("## The paper portfolio — the gross (0-cost) edge\n\n"
           f"Dollar-neutral long-top-{int(R['frac']*100)}% / short-bottom-{int(R['frac']*100)}% "
           f"on the signal known at close `t-1`; {R['n_names']} names, {R['n_days']} days, "
           f"moderate turnover ({R['turnover']}/day)."),
        code(
            "print(f\"gross spread : {R['gross_bps']:+.2f} bps/day  NW(10) t = {R['gross_t']:+.2f}\")\n"
            "print(f\"gross Sharpe : {R['gross_sharpe']:.2f}  (~{R['gross_ann']:+.1f}%/yr)\")\n"
            "print(f\"turnover     : {R['turnover']:.3f}/day (one-way)\")"
        ),
        md("## The cost ladder — same book, four cost worlds\n\n"
           "`net = gross - turnover·cost_bps - impact_coef_bps·turnover²` — a linear one-way cost "
           "plus a super-linear market-impact term (~ participation)."),
        code(
            "print(f\"{'scenario':<16}{'one-way/impact':>16}{'gross Sh':>10}{'net Sh':>9}"
            "{'net bps':>9}{'cost/day':>10}{'net t':>8}\")\n"
            "for lab, cb, ic, gs, ns, nb_, cd, nt, ann in R['ladder']:\n"
            "    print(f\"{lab:<16}{f'{cb:g}bp / {ic:g}':>16}{gs:>10.2f}{ns:>9.2f}"
            "{nb_:>+9.2f}{cd:>10.2f}{nt:>+8.2f}\")"
        ),
        md("## The break-even cost — the deceptive 'headroom'\n\n"
           "The break-even *linear* one-way cost (net alpha → 0) looks comfortable — but it "
           "**ignores market impact**. Quoting a break-even without an impact model is another "
           "way of ignoring costs."),
        code(
            "print(f\"break-even LINEAR one-way cost : {R['breakeven_linear']:.2f} bps\")\n"
            "print(f\"realistic all-in cost charged  : ~{R['ladder'][2][6]:.1f} bps/day\")\n"
            "print(\"the 35 bp 'headroom' is a fantasy: with impact, the realistic 11 bps/day \"\n"
            "      \"cost already matches the 12.25 bps/day gross edge -> net ~0\")"
        ),
        md("## The turnover curve — alpha dies as a FUNCTION of turnover\n\n"
           "Hold the gross edge fixed (same `edge`), turn only the persistence knob φ; charge the "
           "**realistic** cost at every point. Gross Sharpe is ~flat; net Sharpe falls off a cliff."),
        code(
            "print(f\"{'phi':>6}{'turnover':>10}{'gross Sh':>10}{'net Sh':>9}{'break-even':>12}\")\n"
            "for phi, tu, gs, ns, be in R['curve']:\n"
            "    print(f\"{phi:>6}{tu:>10.3f}{gs:>10.2f}{ns:>9.2f}{be:>10.1f}bp\")\n"
            "print('gross Sharpe range:', round(max(c[2] for c in R['curve'])-min(c[2] for c in R['curve']),2),\n"
            "      '| net Sharpe range:', round(max(c[3] for c in R['curve'])-min(c[3] for c in R['curve']),2))"
        ),
        md("## Synthetic control — the machinery is unbiased (20 seeds)\n\n"
           "Live: the gross book must recover the planted edge and stay silent on the null. A "
           "faithful-engine check only — never cited in support of a stamp."),
        code(
            BOOTSTRAP +
            "import numpy as np\n"
            "from cost_gap import data, strategy as st\n"
            "null = st.seed_robust_control(data, edge=0.0, n_seeds=8, n_days=1200)\n"
            "plant = st.seed_robust_control(data, edge=0.0005, n_seeds=8, n_days=1200)\n"
            "print(f\"null   (edge=0)     : gross t mean {null['mean_t']:+.2f} (sd {null['sd_t']:.2f}), \"\n"
            "      f\"|t|>=2 in {null['fire_count']}/{null['n_seeds']}\")\n"
            "print(f\"planted (edge=5e-4) : gross t mean {plant['mean_t']:+.2f} (sd {plant['sd_t']:.2f}), \"\n"
            "      f\"|t|>=2 in {plant['fire_count']}/{plant['n_seeds']}\")\n"
            "print('(frozen 20-seed run: null 1/20 at t~0.0; planted 20/20 at t~6.1)')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — NONE.** A synthetic-only method demo: the gross edge is *planted*, not "
           f"found on a real tape, so it can never earn `REAL` (which needs a robust *t* ≥ 2 on real "
           f"data). The 0-cost gross Sharpe is a genuine **{R['gross_sharpe']}** (NW *t* = "
           f"**+{R['gross_t']}**) — but that is the paper number.\n"
           f"- **Tradability — MIRAGE.** Net Sharpe falls **{R['gross_sharpe']} → "
           f"{R['ladder'][1][4]} → {R['ladder'][2][4]}** (net *t* = {R['ladder'][2][7]}, dead) → "
           f"**{R['ladder'][3][4]}** down the cost ladder, and reaches **{R['curve'][-1][3]}** at "
           f"high turnover. Nothing survives the trading.\n"
           f"- **Does ignoring costs manufacture the edge? — CONFIRMED.** The entire "
           f"{R['gross_ann']}%/yr paper triumph is the cost of trading the backtest forgot to "
           f"charge; it scales with turnover and the linear break-even hides it. The 20-seed control "
           f"confirms the gross edge is genuinely there to *be* eaten — report gross *and* net with a "
           f"turnover-aware cost model, or the backtest is meaningless."),
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
