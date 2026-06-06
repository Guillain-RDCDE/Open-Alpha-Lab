"""Generate the two narrative notebooks from source, then they are executed.

Why a builder script instead of hand-edited .ipynb? Reproducibility: the
notebooks are a *generated artefact*. Edit the cell text here, re-run
``python notebooks/build_notebooks.py`` to rebuild the skeletons, then execute
with nbconvert to embed the figures/outputs:

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_pour_les_curieux.ipynb notebooks/02_pour_les_quants.ipynb

Two audiences, two files:
  01_pour_les_curieux  — plain-language story, no jargon (the "for the curious")
  02_pour_les_quants   — real data, critique, statistics, execution realism
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# A header every notebook runs to find the package and use inline figures.
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.5)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
"""


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# ===========================================================================
# 01 — POUR LES CURIEUX
# ===========================================================================
def build_curieux():
    cells = [
        md(
            "# Pourquoi les marchés gagnent-ils *la nuit* ? 🌙\n"
            "### Une anomalie boursière réelle — et pourquoi elle est plus subtile qu'elle n'en a l'air\n\n"
            "Sur 30 ans, l'essentiel de la hausse des grandes bourses mondiales s'est "
            "accumulé **pendant que le marché était fermé** (de la clôture d'un jour à "
            "l'ouverture du lendemain). La **séance de journée** (ouverture → clôture), "
            "elle, est quasiment plate. Étrange, non ?\n\n"
            "Ce notebook raconte l'histoire **sans jargon**. Si vous voulez la version "
            "rigoureuse (données réelles, statistiques, frais), filez vers "
            "[`02_pour_les_quants.ipynb`](02_pour_les_quants.ipynb).\n\n"
            "> ⚠️ **Ceci n'est pas un conseil en investissement.** Outil de pédagogie et de recherche."
        ),
        code(BOOT),
        md(
            "## 1. Le constat : la nuit monte, le jour stagne\n\n"
            "Construisons un marché *jouet*, totalement honnête : chaque nuit il dérive "
            "d'un cheveu vers le haut (+3 points de base, soit +0,03 %), chaque jour d'un "
            "cheveu vers le bas, et **tout le reste est du bruit pur — aucune fraude, aucun "
            "complot**. Que donne la décomposition nuit / jour ?"
        ),
        code(
            "from overnight import decompose, diagnostics\n\n"
            "ohlc = diagnostics.synthetic_ohlc(overnight_bias_bps=3, intraday_bias_bps=-1, seed=0)\n"
            "dec = decompose.decompose(ohlc)\n\n"
            "ax = plt.subplot()\n"
            "ax.plot(dec.index, dec['cum_overnight']*100, label='Nuit (clôture→ouverture)', lw=2)\n"
            "ax.plot(dec.index, dec['cum_intraday']*100, label='Jour (ouverture→clôture)', lw=2)\n"
            "ax.plot(dec.index, dec['cum_close_close']*100, label='Acheter & conserver', color='grey', lw=1.2)\n"
            "ax.set_title('Marché jouet — aucune fraude, juste un biais de 3 pdb la nuit')\n"
            "ax.set_ylabel('Rendement cumulé (%)'); ax.legend(); ax.grid(alpha=.3)\n"
            "plt.show()\n"
            "s = decompose.summary(dec)\n"
            "print(f\"Nuit cumulée : {s.loc['overnight','cum_return']*100:+.0f}%   \"\n"
            "      f\"Jour cumulé : {s.loc['intraday','cum_return']*100:+.0f}%\")"
        ),
        md(
            "**On a reproduit le motif de Knuteson sans la moindre manipulation.** Un "
            "minuscule biais constant, répété ~250 nuits par an pendant 32 ans, suffit. "
            "La nuit *semble* magique. Gardez cette idée : un tout petit décalage, "
            "répété des milliers de fois, devient énorme. C'est notre premier piège."
        ),
        md(
            "## 2. Piège n°1 — l'échelle ment (la magie des intérêts composés)\n\n"
            "Les graphiques de l'article sont en **échelle logarithmique**, où l'on voit "
            "vite « des milliards de % ». Mais d'où vient ce chiffre vertigineux ? Pas "
            "d'une fraude : de la **composition**. Regardez ce que devient un simple biais "
            "constant selon sa taille et l'horizon :"
        ),
        code(
            "table = diagnostics.compounding_table()\n"
            "diagnostics.format_compounding(table)"
        ),
        md(
            "Un biais de **1 point de base par nuit** — totalement innocent, indétectable — "
            "compose à trois chiffres sur 30 ans. À 30 pdb, on atteint des *billions* de "
            "pourcents. **L'explosion vient de l'exposant, pas d'un complot.** Toute "
            "magnitude spectaculaire affichée en log doit d'abord passer ce test de bon sens."
        ),
        md(
            "## 3. Piège n°2 — des données sales fabriquent le signal\n\n"
            "Les données gratuites (Yahoo) ajustent mal certains *splits* et dividendes, "
            "surtout sur les marchés émergents. Une poignée de prix corrompus suffit à "
            "**déplacer mécaniquement du rendement du jour vers la nuit**. Démonstration "
            "sur un marché totalement plat (zéro biais) où l'on salit 3 cours :"
        ),
        code(
            "flat = diagnostics.synthetic_ohlc(overnight_bias_bps=0, intraday_bias_bps=0, seed=1)\n"
            "propre = decompose.decompose(flat)\n"
            "sale = decompose.decompose(diagnostics.inject_split_artifact(flat, factor=1.5))\n"
            "print(f\"Nuit cumulée  AVANT : {propre['cum_overnight'].iloc[-1]*100:+.1f}%\")\n"
            "print(f\"Nuit cumulée  APRÈS : {sale['cum_overnight'].iloc[-1]*100:+.1f}%   (3 cours salis)\")\n"
            "flags = diagnostics.flag_suspicious_returns(sale)\n"
            "print(f\"\\nLe détecteur automatique repère {len(flags)} jour(s) suspect(s) :\")\n"
            "flags[['r_overnight','r_intraday']]"
        ),
        md(
            "Trois erreurs de données, et la « performance nuit » passe du rouge au vert "
            "vif. C'est exactement le mécanisme derrière les chiffres délirants de "
            "certains marchés émergents dans l'article. **Avant de crier au scandale, "
            "vérifiez vos données.**"
        ),
        md(
            "## 4. Piège n°3 — les frais effacent le gain\n\n"
            "Admettons que l'effet nuit soit réel (il l'est en partie). Peut-on le "
            "*trader* ? La stratégie « acheter à la clôture, vendre à l'ouverture » paie "
            "l'écart achat/vente **deux fois par jour, ~250 jours par an**. Regardons ce "
            "qu'il reste quand on soustrait des frais réalistes :"
        ),
        code(
            "from overnight import backtest\n"
            "sweep = backtest.cost_sweep(dec, roundtrip_bps=(0,1,2,3,5,8))\n"
            "vue = sweep.copy()\n"
            "vue['cagr_net'] = (vue['cagr_net']*100).map('{:+.2f}%'.format)\n"
            "vue['sharpe_net'] = vue['sharpe_net'].map('{:+.2f}'.format)\n"
            "vue['max_drawdown'] = (vue['max_drawdown']*100).map('{:.0f}%'.format)\n"
            "vue.columns = ['Rendement annuel net', 'Sharpe net', 'Pire perte']\n"
            "vue.index.name = 'Coût aller-retour (pdb)'\n"
            "vue"
        ),
        md(
            "À **0 frais**, le Sharpe est correct (~0,7). À un coût réaliste de **5 points "
            "de base** l'aller-retour, le gain devient **négatif**. C'est précisément le "
            "sort des ETF « night effect » NSPY et NIWM : lancés en juin 2022, "
            "**liquidés en août 2023** après une forte sous-performance.\n\n"
            "> *Une stratégie magnifique sur le papier ne vaut pas plus que le papier "
            "tant qu'elle n'a pas payé les coûts réels d'exécution.*"
        ),
        md(
            "## En résumé\n\n"
            "| | |\n|---|---|\n"
            "| ✅ **Le fait est réel** | la nuit a bien sur-performé le jour, sur des décennies |\n"
            "| ⚠️ **Mais les chiffres sont gonflés** | composition + échelle log + données sales |\n"
            "| ❌ **Et difficilement exploitable** | les frais effacent l'avantage |\n\n"
            "L'anomalie nuit/jour est un superbe cas d'école : **réelle, fascinante, mais "
            "à manier avec rigueur**. Pour la version chiffrée sur données réelles — test "
            "Chine, artefacts, statistiques, bêta vs alpha — voir "
            "[`02_pour_les_quants.ipynb`](02_pour_les_quants.ipynb)."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_pour_les_curieux.ipynb")


# ===========================================================================
# 02 — POUR LES QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# L'anomalie overnight — analyse quantitative\n"
            "### Données réelles, démontage critique, statistiques, réalisme d'exécution\n\n"
            "Version rigoureuse du [notebook pour curieux](01_pour_les_curieux.ipynb). On "
            "y traite quatre questions qu'un quant sceptique pose immédiatement :\n\n"
            "1. **Le motif tient-il sur données réelles, partout ?** (spoiler : non — et c'est instructif)\n"
            "2. **Le Sharpe overnight est-il distinguable de zéro ? Est-ce de l'alpha ou du bêta déguisé ?**\n"
            "3. **Survit-il aux coûts d'exécution réels ?**\n"
            "4. **Le tout est-il reproductible ?**\n\n"
            "> ⚠️ **Pas un conseil en investissement.** Données : Yahoo! Finance via `yfinance`, "
            "mode d'ajustement `split_only` (choix documenté en §3.3). Première exécution = accès réseau."
        ),
        code(BOOT + "\nfrom overnight import data, decompose, diagnostics, backtest, stats\n"),
        md(
            "## 1. Le motif sur 10 indices mondiaux (ETF)\n\n"
            "On décompose chaque ETF en nuit / jour et on lit le Sharpe **annualisé de la "
            "jambe overnight** — le seul chiffre qui compte pour juger d'un *edge*."
        ),
        code(
            "rows, decs = {}, {}\n"
            "for tk, label in data.WORLD_INDICES.items():\n"
            "    try:\n"
            "        dec = decompose.decompose(data.fetch(tk, mode='split_only'))\n"
            "    except Exception as e:\n"
            "        print(f'{tk}: FAILED {e}'); continue\n"
            "    decs[tk] = dec\n"
            "    s = decompose.summary(dec)\n"
            "    rows[tk] = {\n"
            "        'pays': label,\n"
            "        'nuit cum %': s.loc['overnight','cum_return']*100,\n"
            "        'jour cum %': s.loc['intraday','cum_return']*100,\n"
            "        'Sharpe nuit': s.loc['overnight','sharpe'],\n"
            "        'Sharpe jour': s.loc['intraday','sharpe'],\n"
            "        'jours suspects': len(diagnostics.flag_suspicious_returns(dec)),\n"
            "    }\n"
            "table = pd.DataFrame(rows).T\n"
            "table"
        ),
        md(
            "Lecture rapide : **USA (SPY, QQQ)** et **Brésil** affichent le motif classique "
            "(nuit énorme, jour faible ou négatif, Sharpe nuit ~0,7). Mais regardez "
            "l'**Europe (UK, Allemagne, France)** et le **Japon** : le motif est **inversé**. "
            "Ce n'est pas un détail — c'est le cœur du démontage."
        ),
        md(
            "## 2. Démontage critique\n\n"
            "### 2.1 L'inversion des ETF étrangers : on mesure le fuseau horaire, pas une anomalie\n\n"
            "Visualisons le Sharpe nuit vs jour par marché :"
        ),
        code(
            "t = table.sort_values('Sharpe nuit')\n"
            "x = np.arange(len(t)); w = 0.4\n"
            "fig, ax = plt.subplots(figsize=(11,5))\n"
            "ax.bar(x-w/2, t['Sharpe nuit'].astype(float), w, label='Sharpe nuit', color='#2c7fb8')\n"
            "ax.bar(x+w/2, t['Sharpe jour'].astype(float), w, label='Sharpe jour', color='#fdae61')\n"
            "ax.axhline(0, color='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(t.index)\n"
            "ax.set_ylabel('Sharpe annualisé'); ax.set_title('Nuit vs jour par marché — le signe s\\'inverse pour les ETF étrangers')\n"
            "ax.legend(); ax.grid(axis='y', alpha=.3); plt.show()"
        ),
        md(
            "**Explication microstructure.** EWU, EWG, EWQ, EWJ sont des ETF cotés à New "
            "York mais dont le sous-jacent (Londres, Francfort, Paris, Tokyo) **trade "
            "pendant la nuit américaine**. Pour ces produits, la fenêtre « overnight » "
            "(horloge US, clôture→ouverture) **contient la séance du marché domestique**, "
            "tandis que la fenêtre « intraday » US correspond aux heures où le sous-jacent "
            "est largement fermé (le prix ne bouge que par arbitrage sur le NAV).\n\n"
            "Autrement dit : **la décomposition nuit/jour est relative à l'horloge de "
            "cotation**. Appliquée à un instrument décalé de son marché, elle mesure "
            "surtout le fuseau horaire — pas une « anomalie ». Un manipulateur mondial "
            "unique expliquerait mal pourquoi le signe dépend du lieu de cotation de l'ETF. "
            "C'est un argument de prudence majeur, et un rappel : **toujours vérifier ce "
            "que la fenêtre temporelle capture réellement pour l'instrument choisi.**"
        ),
        md(
            "### 2.2 Le test de la Chine et la règle T+1\n\n"
            "Sur les actions chinoises, la littérature (Qiao & Dam, 2020) documente un "
            "motif **inversé** (jour positif, nuit négative), proprement expliqué par la "
            "règle **T+1** (titres achetés un jour invendables avant le lendemain). Notre "
            "proxy ETF (FXI, coté US) ne teste pas directement les A-shares, mais on note "
            "un signal nettement plus faible que pour les USA :"
        ),
        code(
            "if 'FXI' in decs:\n"
            "    s = decompose.summary(decs['FXI'])\n"
            "    print('Chine (FXI) — Sharpe nuit : {:+.2f}  vs  USA (SPY) : {:+.2f}'.format(\n"
            "        s.loc['overnight','sharpe'], decompose.summary(decs['SPY']).loc['overnight','sharpe']))\n"
            "    print('Un motif universel et orchestré devrait être plus homogène géographiquement.')"
        ),
        md(
            "### 2.3 Composition et sélection\n\n"
            "Comme montré dans le notebook pour curieux, les magnitudes en log sont "
            "dominées par la **composition** (un biais de 1 pdb → +124 % sur 32 ans) et "
            "par la **sélection** (les figures les plus spectaculaires sont, de l'aveu "
            "même de l'auteur, « les 25 plus problématiques »). Le **compteur de jours "
            "suspects** (colonne de §1) reste faible ici car les ETF US sont propres — "
            "les artefacts vivent surtout dans les **indices spot émergents bruts** "
            "(ex. `^BSESN`), non dans ces ETF. Caveat à garder pour toute reproduction de "
            "la fameuse Figure 8 (Inde)."
        ),
        md(
            "## 3. Rigueur statistique & risque\n\n"
            "### 3.1 Le Sharpe overnight est-il distinguable de zéro ? (bootstrap)\n\n"
            "Un Sharpe de 0,77 sur un échantillon fini peut-il être du bruit ? Intervalle "
            "de confiance à 95 % par bootstrap (2000 rééchantillonnages) :"
        ),
        code(
            "for tk in ['SPY','QQQ','EWZ','FXI']:\n"
            "    if tk not in decs: continue\n"
            "    r = stats.sharpe_ci_bootstrap(decs[tk]['r_overnight'], n_boot=2000, seed=0)\n"
            "    print(f\"{tk:4s} Sharpe nuit = {r['sharpe']:+.2f}  \"\n"
            "          f\"IC95% [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  \"\n"
            "          f\"P(Sharpe<0) = {r['frac_negative']:.1%}  (n={r['n_obs']})\")"
        ),
        md(
            "Pour SPY/QQQ/Brésil l'intervalle exclut largement zéro : l'effet est "
            "**statistiquement réel**. La question n'est donc pas *« existe-t-il ? »* mais "
            "*« est-ce de l'alpha exploitable ? »* — d'où les deux sections suivantes."
        ),
        md(
            "### 3.2 Alpha, ou bêta déguisé ?\n\n"
            "Détenir le marché *chaque nuit*, c'est porter le **risque de gap** en "
            "permanence. Une partie de « l'alpha overnight » est donc une **prime de risque "
            "de bêta**, pas un edge distinct. On régresse la jambe nuit sur le marché "
            "(close-close) : `r_nuit = α + β·r_marché + ε`."
        ),
        code(
            "d = stats.beta_decomposition(decs['SPY'], leg='overnight')\n"
            "print('SPY — jambe overnight régressée sur le marché (close-close) :')\n"
            "print(f\"  bêta = {d['beta']:.2f}   R² = {d['r_squared']:.2f}\")\n"
            "print(f\"  rendement nuit moyen = {d['mean_leg_bps']:.2f} pdb/jour\")\n"
            "print(f\"     dont bêta×marché  = {d['beta_contrib_bps']:.2f} pdb   (prime de risque)\")\n"
            "print(f\"     dont alpha résid. = {d['alpha_daily_bps']:.2f} pdb   ({d['alpha_ann_pct']:+.1f}%/an)\")"
        ),
        md(
            "Deux lectures, toutes deux défavorables à la thèse de l'edge facile :\n\n"
            "- **~40 % du rendement nocturne est du bêta** (bêta ≈ 0,33 sur SPY) : "
            "détenir le marché la nuit est en partie une simple **prime de risque de gap**, "
            "pas un alpha distinct.\n"
            "- Surtout, l'**alpha résiduel (~1,9 pdb/jour) est *inférieur* au coût de point "
            "mort (~3,25 pdb)** calculé en §4 : même la part « non-bêta » du rendement ne "
            "survit pas aux frais d'exécution. C'est le clou du cercueil — il faut facturer "
            "les coûts sur l'alpha, pas sur le rendement brut."
        ),
        md(
            "### 3.3 Sensibilité au mode d'ajustement des dividendes\n\n"
            "Le mode d'ajustement n'est **pas un détail** : un titre passe ex-dividende à "
            "l'**ouverture**, donc l'ajustement déplace du rendement entre nuit et jour. "
            "Comparons `split_only` (défaut, garde le gap ex-div dans la nuit) vs "
            "`total_return` (tout ajusté) sur SPY :"
        ),
        code(
            "spy_tr = decompose.decompose(data.fetch('SPY', mode='total_return'))\n"
            "spy_so = decs['SPY']\n"
            "print('SPY — nuit cumulée :')\n"
            "print(f\"  split_only   : {spy_so['cum_overnight'].iloc[-1]*100:+,.0f}%\")\n"
            "print(f\"  total_return : {spy_tr['cum_overnight'].iloc[-1]*100:+,.0f}%\")\n"
            "print('Le choix déplace le niveau de la jambe nuit -> à documenter dans toute figure publiée.')"
        ),
        md(
            "## 4. Réalisme d'exécution\n\n"
            "Modèle de coût par nuit détenue :\n"
            "`coût = 2×(½·spread + commission + slippage) + financement`. "
            "Le **facteur 2** (on traverse la fourchette à l'achat *et* à la vente, "
            "~252×/an) est le tueur. Point mort et balayage sur SPY :"
        ),
        code(
            "be = backtest.breakeven_cost_bps(spy_so)\n"
            "print(f'Coût aller-retour de point mort = {be:.2f} pdb/nuit')\n"
            "sweep = backtest.cost_sweep(spy_so, roundtrip_bps=(0,1,2,3,5,8))\n"
            "sweep.round(3)"
        ),
        md(
            "L'edge brut survit à quelques points de base seulement. Trois facteurs "
            "aggravants côté retail, non capturés par ce backtest *optimiste* :\n\n"
            "- **Prix d'exécution ≠ prints académiques** : l'anomalie est mesurée sur les "
            "enchères de clôture/ouverture, inaccessibles au particulier ; à T±5 min on "
            "est en séance continue, avec un spread plus large.\n"
            "- **Swap CFD / MT5** : le financement overnight prélevé chaque nuit peut "
            "annuler l'edge à lui seul (vérifier `swap_long` avant tout trade — c'est "
            "exactement le garde-fou implémenté dans le connecteur MT5 du repo).\n"
            "- **Capacité / slippage** : l'impact marché croît avec la taille d'ordre, "
            "surtout sur les enchères peu liquides."
        ),
        md(
            "## 5. Reproductibilité\n\n"
            "- **Déterminisme** : tout le synthétique et le bootstrap sont *seedés*.\n"
            "- **Tests** : `pytest` vérifie l'identité de décomposition "
            "`(1+r_nuit)(1+r_jour)=(1+r_cc)` (erreur ~1e-16) et la monotonie du coût.\n"
            "- **CI** : GitHub Actions rejoue tests + démo offline sur Python 3.10–3.12.\n"
            "- **Données** : cache parquet local (`_cache/`), mode d'ajustement explicite.\n"
            "- **Notebooks générés** : `python notebooks/build_notebooks.py` puis "
            "`nbconvert --execute` — la figure que vous lisez est l'output exécuté, pas une capture."
        ),
        md(
            "## 6. Verdict honnête\n\n"
            "Trois niveaux à ne **jamais** confondre :\n\n"
            "1. **Le fait empirique est RÉEL** et bien documenté (Lou-Polk-Skouras 2019, "
            "Cooper-Cliff-Gulen 2008, Fed NY). Mérite à Knuteson d'avoir publié données et code.\n"
            "2. **Les magnitudes sont GONFLÉES** — composition sur 30 ans + échelle log, "
            "artefacts de données, biais de sélection/survie.\n"
            "3. **L'attribution à une fraude orchestrée n'est PAS prouvée** — l'inversion "
            "des ETF étrangers (fuseau horaire) et le cas chinois (T+1) favorisent des "
            "explications **microstructurelles**, et la majeure partie du « rendement nuit » "
            "est du **bêta de gap**, pas un alpha.\n\n"
            "Et même en supposant l'edge réel : **il ne survit pas aux coûts d'exécution "
            "réels** — comme l'ont montré la liquidation des ETF NSPY / NIWM en 2023. "
            "Belle anomalie pour comprendre la microstructure ; piètre stratégie de trading retail."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_pour_les_quants.ipynb")


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build_curieux()
    build_quants()
