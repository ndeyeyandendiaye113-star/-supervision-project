# Console de supervision — Détection d'anomalies par auto-encodeur

Interface Django pour le **Sujet 7** : détection d'anomalies non supervisée
par auto-encodeur sur capteurs industriels (AI4I 2020 Predictive Maintenance).

Cette console répond aux 3 exigences du sujet :
1. **Score d'anomalie en continu** — chaque observation est scorée par
   l'auto-encodeur (erreur de reconstruction).
2. **Alerte au dépassement de seuil** — bandeau rouge/vert + badge par ligne
   dès que le score dépasse le seuil (97e percentile des erreurs sur données
   normales, calculé dans le notebook).
3. **Diagnostic des variables incriminées** — pour chaque anomalie, la
   contribution de chaque variable à l'erreur de reconstruction est affichée
   (barres triées par ordre décroissant).

L'auto-encodeur est **la seule approche utilisée ici**, conformément à la
problématique du sujet (« n'ayant appris que le fonctionnement normal »).
L'ANN supervisé du notebook reste un point de comparaison analytique, mais
n'a pas sa place dans cette console (il suppose des labels disponibles).

---

## 1. Récupérer les artefacts du modèle

Ce projet ne contient **pas** le modèle entraîné : il faut l'exporter depuis
`PROJET_DEEP_LEARNING_audite_corrige_FINAL.ipynb`. Les cellules d'export
existent déjà à la fin du notebook (section auto-encodeur) :

```python
model.save("autoencoder.keras")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")
joblib.dump({"threshold": float(threshold), "percentile": THRESHOLD_PERCENTILE}, "threshold.pkl")
```

Téléchargez les 4 fichiers générés (`autoencoder.keras`, `scaler.pkl`,
`feature_names.pkl`, `threshold.pkl`) depuis Colab, puis placez-les dans :

```
monitoring/ml_models/
```

**Important** : la variable `Type` (L/M/H) n'est **plus** utilisée par le
modèle dans la version finale du notebook (encodage one-hot testé puis
abandonné). Le modèle attend **10 features**. L'interface conserve un champ
"Type de machine" à titre informatif/traçabilité uniquement — il n'entre pas
dans le calcul du score.

---

## 2. Installation locale

```bash
python3 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt

python3 manage.py migrate
python3 manage.py createsuperuser   # optionnel, pour /admin/

python3 manage.py runserver
```

Ouvrez ensuite **http://127.0.0.1:8000/**.

Si les 4 fichiers ne sont pas dans `monitoring/ml_models/`, la page affiche
un message d'erreur explicite au lieu de planter — vous pouvez naviguer sur
le reste du site normalement.

---

## 3. Utilisation

- **Saisir une observation** : formulaire à gauche (température air/process,
  vitesse de rotation, couple, usure outil, type machine). Le calcul se fait
  côté serveur avec le *même* feature engineering que le notebook
  (`temperature_difference`, `mechanical_power`, `tool_stress`,
  `temperature_ratio`, `torque_speed_ratio`).
- **Simuler un cycle** : génère une observation aléatoire (≈15% dégradées)
  pour démontrer la console sans capteur réel — pratique pour la soutenance.
- **Flux continu en arrière-plan** (optionnel, pour une démo qui tourne
  toute seule pendant que vous présentez) :

  ```bash
  python3 manage.py simulate_stream --count 50 --interval 2 --anomaly-rate 0.15
  ```

  Laissez cette commande tourner dans un terminal pendant que vous
  rafraîchissez le dashboard dans le navigateur.

- **Réinitialiser** : efface l'historique (bouton rouge).
- **Panel admin** (`/admin/`) : consultation brute des `Reading` en base
  si besoin pendant la soutenance.

---

## 4. Architecture du projet

```
supervision_project/
├── manage.py
├── requirements.txt
├── supervision_project/        # config Django (settings, urls)
└── monitoring/                 # app principale
    ├── models.py                # modèle Reading (historique + diagnostic)
    ├── ml_utils.py               # pipeline d'inférence (fidèle au notebook)
    ├── forms.py                  # formulaire de saisie manuelle
    ├── views.py                  # dashboard, simulate, reset, chart-data (JSON)
    ├── urls.py
    ├── admin.py
    ├── management/commands/
    │   └── simulate_stream.py    # démo en flux continu (CLI)
    ├── ml_models/                 # <-- PLACEZ ICI les 4 fichiers exportés
    └── templates/monitoring/
        ├── base.html              # thème sombre, Chart.js via CDN
        └── dashboard.html
```

`ml_utils.py` est le seul fichier à modifier si le feature engineering
change encore dans le notebook : il lit dynamiquement l'ordre des colonnes
depuis `feature_names.pkl`, donc il n'y a normalement rien à toucher tant
que les noms de colonnes restent cohérents entre le notebook et ce module.

---

## 5. Déploiement

Projet Django standard, déployable sur Render, Railway, un VPS
(Gunicorn + Nginx), PythonAnywhere, etc. Points d'attention :

- **TensorFlow est lourd** : sur un tier gratuit avec peu de RAM, le
  chargement du modèle Keras peut échouer. Railway ou un VPS avec ≥1 Go de
  RAM sont plus fiables qu'un tier gratuit Render pour ce type de charge.
- Avant mise en ligne, définissez ces variables d'environnement (au lieu des
  valeurs par défaut de dev) :
  - `DJANGO_SECRET_KEY` — une clé secrète générée pour la prod
  - `DJANGO_DEBUG=False`
  - `DJANGO_ALLOWED_HOSTS` — domaine(s) séparés par des virgules
- Base de données : SQLite convient pour une démo/soutenance ; passez à
  PostgreSQL pour un usage persistant multi-utilisateurs.
- Fichiers statiques : `python3 manage.py collectstatic` avant déploiement
  si vous ajoutez des assets locaux (le projet actuel charge Chart.js via
  CDN, donc pas indispensable pour la démo).

---

## 6. Cohérence avec le notebook — points vérifiés

- ✅ Le scaler est appliqué de la même façon (`StandardScaler` ajusté
  uniquement sur les données normales d'entraînement dans le notebook).
- ✅ L'erreur de reconstruction est la MSE moyenne sur les features
  standardisées (pas les données brutes).
- ✅ Le seuil vient directement de `threshold.pkl` (97e percentile calculé
  dans le notebook), pas recalculé côté interface.
- ✅ L'ordre des features vient de `feature_names.pkl`, pas codé en dur —
  robuste si le notebook évolue encore.
- ✅ `Type` n'est plus une feature du modèle dans cette version finale (testé
  avec des artefacts factices de mêmes dimensions : le pipeline calcule bien
  10 features et ignore `Type` pour le scoring).
