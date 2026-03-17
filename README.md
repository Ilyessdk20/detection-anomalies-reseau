# network-anomaly-detection

Projet Python / Jupyter orienté cybersécurité pour détecter des activités réseau anormales sur un jeu de données CSV.

## Objectif du projet

Ce projet montre une chaîne data complète, simple et crédible pour un niveau étudiant :

- chargement et nettoyage de logs réseau,
- analyse exploratoire,
- visualisation de variables clés,
- entraînement d'un modèle de détection d'anomalies,
- évaluation et interprétation des résultats.

Le projet est conçu pour être présenté dans un portfolio GitHub et valorisé pour une candidature en Master Cyber sécurité et Sciences des données.

## Technologies utilisées

- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn
- Jupyter Notebook

## Structure du projet

```text
network-anomaly-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── network_data.csv
├── notebooks/
│   └── anomaly_detection.ipynb
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   └── utils.py
└── results/
    ├── figures/
    └── reports/
```

## Description du dataset

Le fichier `data/network_data.csv` contient un exemple de connexions réseau avec les variables suivantes :

- `duration_sec` : durée de la connexion,
- `protocol` : protocole (`tcp`, `udp`, `icmp`),
- `service` : service ciblé (`http`, `ssh`, `dns`, etc.),
- `src_bytes` / `dst_bytes` : volume de trafic,
- `failed_logins`, `login_attempts` : signaux d'authentification,
- `connection_rate`, `same_srv_rate`, `dst_host_count` : indicateurs réseau,
- `label` : `normal` ou `anomaly`.

## Approche de modélisation

Le notebook sélectionne automatiquement l'approche :

- si une colonne `label` exploitable est présente : classification supervisée (Logistic Regression),
- sinon : détection non supervisée (Isolation Forest).

Dans l'exemple fourni, le mode supervisé est utilisé.

## Installation

```bash
cd network-anomaly-detection
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Exécution

### Option 1: Notebook (recommandée)

```bash
jupyter notebook notebooks/anomaly_detection.ipynb
```

### Option 2: Réutilisation des modules Python

Les fonctions de prétraitement et de modélisation sont disponibles dans `src/` pour être réutilisées dans d'autres scripts.

## Résultats produits

- Graphiques dans `results/figures/` :
  - distribution des classes,
  - histogramme de trafic,
  - nuage de points,
  - matrice de confusion.
- Rapport texte dans `results/reports/model_summary.txt`.

## Intérêt pour un profil Cyber + Data

Ce projet démontre des compétences utiles en cybersécurité data-driven :

- préparation de données réseau imparfaites,
- détection d'activités suspectes via machine learning,
- interprétation d'indicateurs de performance,
- communication claire des limites d'un modèle.

## Limites et pistes d'amélioration

- dataset d'exemple de petite taille (pédagogique),
- peu de features temporelles avancées,
- pas de validation croisée complète.

Améliorations possibles :

- ajout d'un dataset public plus volumineux (CIC-IDS, UNSW-NB15),
- comparaison entre Isolation Forest, LOF et modèles supervisés,
- intégration d'un tableau de bord de suivi.

