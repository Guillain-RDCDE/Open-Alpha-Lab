# PROJECT BRIEF — Overnight Anomaly (vérification + implémentation)

> Document de reprise (handoff). Objectif : permettre de continuer le projet
> dans VS Code sans reperdre le contexte. À lire avec, dans le même dossier :
> le PDF de l'article (Knuteson, *Nothing to See Here*, SSRN 4619084) et la
> conversation initiale.

---

## 0. En une phrase

On a transformé un pamphlet financier (Knuteson) en **code reproductible** pour
(1) **vérifier soi-même** ses figures et son raisonnement, et (2) **implémenter
prudemment** la stratégie « acheter à la clôture, vendre à l'ouverture » — en
montrant chiffres à l'appui pourquoi l'edge brut survit rarement aux frais.

---

## 1. Contexte : que dit l'article ?

Knuteson observe que, sur ~30 ans, l'essentiel des gains des indices mondiaux
s'accumule **overnight** (clôture → ouverture suivante), tandis que la séance
**intraday** (ouverture → clôture) est plate à négative. Il en déduit — sur le
ton satirique d'un « guide pour commettre une fraude » — que ce motif serait la
signature d'une **manipulation de marché** par un grand hedge fund quantitatif
(la « Strategy » de la firme « M » : élargir le book quand le marché est
illiquide près de l'ouverture, le contracter quand il est liquide).

### Conclusion de notre analyse critique (à conserver)
Trois niveaux à **ne jamais confondre** :

1. **Le fait empirique est RÉEL** et bien documenté (Lou-Polk-Skouras 2019,
   Cooper-Cliff-Gulen 2008, Boyarchenko/Fed NY, etc.). Mérite de Knuteson :
   avoir alerté + publié données et code.
2. **Les magnitudes affichées sont GONFLÉES** par trois effets :
   - **composition** sur 30 ans + échelle log (un biais minuscule → « milliards de % ») ;
   - **artefacts de données** (splits/dividendes mal ajustés sur Yahoo, surtout
     marchés émergents → chiffres délirants de l'Inde, Figure 8) ;
   - **biais de sélection/survie** (Figure 2 = « les 25 plus problématiques », de son aveu).
3. **L'attribution à une fraude orchestrée par une firme identifiable n'est PAS
   prouvée** ni consensuelle. Points faibles :
   - **Test de la Chine** : motif **inversé** (jour positif, nuit négative),
     expliqué proprement par la règle **T+1** (Qiao & Dam 2020). Un manipulateur
     mondial unique l'expliquerait mal → faveur aux explications microstructurelles.
   - **Conflit d'intérêt / innuendo** : Knuteson est un ex-D.E. Shaw ; il cite la
     sanction SEC 2023 contre D.E. Shaw — mais celle-ci portait sur le **langage
     des accords whistleblower (Rule 21F-17)**, PAS sur une manipulation. Conflation trompeuse.
   - **Épistémologie glissante** : « ce point tient même si la cause est innocente »
     rend la thèse quasi non-réfutable (motte-and-bailey).

### Reality check terrain
Deux ETF « night effect » (NSPY, NIWM, AlphaTrAI) lancés juin 2022, **liquidés
août 2023** après forte sous-performance. Citation Morningstar (paraphrasée) :
une stratégie magnifique sur le papier ne vaut pas plus que le papier tant
qu'elle n'intègre pas les coûts réels d'exécution.

---

## 2. Ce qui a été construit (arborescence)

```
overnight-anomaly/
├── README.md               # présentation dual-audience (débutants + pros)
├── PROJECT_BRIEF.md        # CE fichier
├── requirements.txt        # numpy, pandas, matplotlib, yfinance, pyarrow
├── LICENSE                 # MIT + "not investment advice"
├── .gitignore
├── overnight/              # le package
│   ├── __init__.py
│   ├── decompose.py        # CŒUR : décomposition nuit/jour (exacte) + summary()
│   ├── data.py             # Yahoo! Finance + cache parquet + modes d'ajustement
│   ├── plots.py            # Figure 1(c) de Knuteson (log + chiffres en clair)
│   ├── diagnostics.py      # SOCLE 1 critique : composition, artefacts, pondération (OFFLINE)
│   ├── backtest.py         # SOCLE 2 : backtest avec coûts réels, point mort, balayage
│   └── brokers/
│       ├── __init__.py
│       ├── base.py         # interface broker abstraite (swappable)
│       └── mt5_connector.py# template MetaTrader 5 + planificateur overnight
├── examples/
│   ├── run_synthetic_demo.py    # tout, SANS réseau (validé, voir §3)
│   └── verify_world_indices.py  # reproduction monde (réseau Yahoo requis)
└── out_synthetic_decomposition.png  # figure générée par la démo
```

### Rôle de chaque module
- **`decompose.decompose(ohlc)`** → DataFrame avec `r_overnight`, `r_intraday`,
  `r_close_close` + courbes cumulées. Identité exacte : `(1+r_on)(1+r_id)=(1+r_cc)`.
- **`decompose.summary(dec)`** → rendements cumulés, bps/jour, vol annualisée,
  **Sharpe** (le chiffre qui compte), nb de jours.
- **`data.fetch(ticker, mode=...)`** → `'split_only'` (défaut), `'total_return'`,
  `'raw'`. Cache parquet dans `_cache/`. `data.WORLD_INDICES` = 10 indices.
- **`plots.plot_decomposition / plot_grid`** → format Figure 1(c)/3, option `linear`.
- **`diagnostics`** :
  - `synthetic_ohlc(...)` génère un marché avec biais nuit/jour paramétrable (offline) ;
  - `compounding_table()` montre l'explosion par composition ;
  - `inject_split_artifact()` fabrique un faux signal overnight ;
  - `flag_suspicious_returns(dec, threshold=0.40)` détecte les jours = artefacts probables ;
  - `portfolio_decompose(..., weights='equal'|'first')` montre l'effet de pondération.
- **`backtest`** :
  - `breakeven_cost_bps(dec)` = coût d'aller-retour qui annule l'edge ;
  - `cost_sweep(dec)` = Sharpe/CAGR nets selon le coût ;
  - `backtest_overnight(dec, spread_bps, commission_bps, timing_slippage_bps,
    financing_bps_per_night)` ; coût/jour = `2×(½spread+comm+slippage)+financement`.
- **`brokers/mt5_connector`** : `MT5Connector(dry_run=True)` + `run_overnight_loop(...)`
  (achat T-5 min clôture, vente T+5 min ouverture). Import MT5 différé dans `connect()`.

---

## 3. Résultats VALIDÉS (démo offline exécutée)

`python examples/run_synthetic_demo.py` a tourné avec succès. Chiffres clés :

**(A) Composition** — rendement cumulé d'un simple biais overnight constant :

| biais/nuit | ~10 ans | ~20 ans | ~32 ans |
|---|---|---|---|
| 1 bps | +29 % | +66 % | +123 % |
| 5 bps | +252 % | +1 142 % | +5 354 % |
| 10 bps | +1 141 % | +15 308 % | +296 807 % |
| 30 bps | +189 724 % | +360 M % | **+2 555 milliards %** |

Marché synthétique (nuit +3 bps, jour −1 bp, bruit pur) → overnight cumulé
**+494 %**, intraday **−83 %**, total **+3 %**. ⇒ le « clone » de Knuteson sans
aucune fraude. (Figure : `out_synthetic_decomposition.png`.)

**(B) Artefact** — 3 clôtures entachées sur un marché SANS dérive :
overnight passe de **−38,9 % → +388,5 %**, intraday se dégrade ; le détecteur
repère **6 jours suspects** automatiquement. ⇒ l'erreur déplace mécaniquement du
rendement du jour vers la nuit.

**(C) Frais** — point mort = **2,41 bps**/aller-retour. Balayage :

| coût AR (bps) | CAGR net | Sharpe net | maxDD net |
|---|---|---|---|
| 0 | +5,77 % | 0,64 | −33,6 % |
| 2 | +0,57 % | 0,11 | −44,4 % |
| 3 | −1,93 % | −0,16 | −56,4 % |
| 5 | −6,75 % | −0,68 | −89,7 % |

⇒ scénario réaliste (5 bps) : l'edge brut (Sharpe 0,64) devient **négatif** net.
C'est exactement le sort des ETF NSPY/NIWM.

### Validé vs NON exécuté (limite d'environnement)
- ✅ **Exécuté & validé** : toute la démo offline, imports de tout le package
  (y compris `mt5_connector` sans terminal MT5).
- ⚠️ **Écrit + import-testé mais NON exécuté de bout en bout** (réseau Yahoo
  bloqué dans le sandbox de génération) : `data.fetch()` et
  `examples/verify_world_indices.py`. À lancer chez toi en premier.
- ⚠️ **Template non testé** (requiert MT5 + compte broker) : connecteur live.

---

## 4. Points quant à NE PAS perdre

1. **Dividendes/splits = décision, pas détail.** Le mode d'ajustement déplace du
   rendement entre nuit et jour (un titre passe ex-dividende à l'ouverture).
   `'split_only'` par défaut ; documente ton choix dans tout post/figure.
2. **Sharpe > rendement brut.** Une partie de « l'alpha overnight » est une prime
   de risque de gap (on porte le risque action chaque nuit) → du bêta déguisé.
3. **Le facteur 2 du coût** (on traverse la fourchette à l'achat ET à la vente)
   × ~250 AR/an = le tueur de la stratégie.
4. **CFD/MT5 → swap overnight** prélevé la nuit = peut annuler l'edge à lui seul.
   Vérifier `mt5.symbol_info(sym).swap_long` avant tout.
5. **Prix d'exécution ≠ prints académiques.** L'anomalie est mesurée sur enchères
   clôture/ouverture, inaccessibles au retail. À T±5 min on est en séance continue.
6. **Ce qu'on n'implémente PAS, volontairement** : la « Strategy » de manipulation
   (pousser les prix en période illiquide). On ne fait que récolter passivement la
   prime overnight (price-taker). La version manipulation poserait des problèmes
   réglementaires et n'est pas l'objet.

---

## 5. Décisions ouvertes (à trancher)
- [ ] **Mode dividendes** retenu pour les figures publiées (`split_only` vs `total_return`).
- [ ] **Broker cible** : MT5 (CFD, swap) vs **Alpaca** (US equities, prints
      d'ouverture plus propres, paper-trading gratuit) vs IBKR.
- [ ] **Univers** : indices via ETF (SPY/QQQ…) vs indices spot (^FCHI…) vs titres individuels.
- [ ] **Second fournisseur de données** pour croiser Yahoo (qualité émergents).

## 6. Roadmap proposée
1. Lancer `verify_world_indices.py` en réel → confirmer le test Chine + compteur d'artefacts Inde.
2. **Notebook Jupyter narratif** pour LinkedIn (reprend A/B/C + figures).
3. **Connecteur Alpaca** testable en paper-trading (implémente `BrokerBase`).
4. **Robustesse** : tests unitaires (`pytest`) sur l'identité de décomposition et le coût ;
   calendrier de splits officiel pour durcir `flag_suspicious_returns`.
5. **Étude de capacité/slippage** réaliste par taille d'ordre.
6. Mode **market-neutral** (long/short) en price-taker pour isoler l'effet du bêta marché.

## 7. Commandes utiles
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python examples/run_synthetic_demo.py          # offline, validé
python examples/verify_world_indices.py        # nécessite Internet (Yahoo)
python -c "from overnight import decompose, data; print('ok')"   # smoke test
```

## 8. Références
- Knuteson, *Nothing to See Here* (2023) + arXiv 1612.06855, 1811.04994, 1912.01708,
  2010.01727, 2107.12516, 2201.00223 ; code/thread : bruceknuteson.github.io/spy-day-and-night
- Lou, Polk, Skouras (2019) *A Tug of War* ; Cooper, Cliff, Gulen (2008)
- Haghani et al. / Elm Wealth, *Night Moves* (2022) — discussion quant équilibrée
- Boyarchenko et al. (Fed NY) ; Qiao & Dam (2020, cas chinois T+1) ; Berkman et al. (2012)

## 9. Avertissement
Outil de **recherche et de pédagogie**. **Pas un conseil en investissement.** Le
backtest honnête montre précisément pourquoi la prudence s'impose avant tout
capital réel. Tester d'abord en **compte démo / paper-trading**.
