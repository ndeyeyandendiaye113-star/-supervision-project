"""
Pipeline d'inférence de l'auto-encodeur de détection d'anomalies.

Ce module reproduit EXACTEMENT le feature engineering utilisé lors de
l'entraînement dans le notebook PROJET_DEEP_LEARNING_audite_corrige_FINAL.ipynb,
afin que le scaler et le modèle chargés ici donnent des résultats cohérents
avec le notebook.

Important (version finale auditée) :
    - La variable `Type` (L/M/H) N'EST PLUS utilisée comme feature du
      modèle (encodage one-hot testé puis abandonné dans le notebook).
      Elle est conservée dans l'interface à titre informatif seulement.
    - Le modèle attend 10 features (et non 11 comme dans une version
      précédente).
    - Le seuil est désormais sauvegardé comme un dict
      {"threshold": float, "percentile": int}.

Fichiers attendus dans monitoring/ml_models/ (exportés depuis le notebook
via les cellules `model.save(...)` / `joblib.dump(...)`) :
    - autoencoder.keras
    - scaler.pkl
    - feature_names.pkl   (ordre exact des colonnes utilisées à l'entraînement)
    - threshold.pkl       (dict {"threshold": float, "percentile": int})
"""

import os
import numpy as np
import joblib
from django.conf import settings

# Ordre de secours si feature_names.pkl est absent (doit rester synchronisé
# avec le notebook : X = df.drop(columns=["Machine failure","failure_status","Type"]))
FALLBACK_FEATURE_ORDER = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temperature_difference",
    "mechanical_power",
    "tool_stress",
    "temperature_ratio",
    "torque_speed_ratio",
]

MODEL_DIR = os.path.join(settings.BASE_DIR, "monitoring", "ml_models")

_model = None
_scaler = None
_threshold = None
_threshold_percentile = None
_feature_order = None
_load_error = None


def _load_artifacts():
    """Charge le modèle Keras, le scaler, l'ordre des features et le seuil
    une seule fois (paresseusement), et met en cache le résultat / l'erreur
    éventuelle."""
    global _model, _scaler, _threshold, _threshold_percentile
    global _feature_order, _load_error

    if _model is not None or _load_error is not None:
        return

    try:
        # Import tardif : tensorflow est lourd, on ne le charge que si besoin
        from tensorflow import keras

        model_path = os.path.join(MODEL_DIR, "autoencoder.keras")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        threshold_path = os.path.join(MODEL_DIR, "threshold.pkl")
        features_path = os.path.join(MODEL_DIR, "feature_names.pkl")

        for path in (model_path, scaler_path, threshold_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Fichier manquant : {path}. "
                    "Exportez les artefacts depuis le notebook final "
                    "(model.save('autoencoder.keras'), "
                    "joblib.dump(scaler, 'scaler.pkl'), "
                    "joblib.dump(list(X.columns), 'feature_names.pkl'), "
                    "joblib.dump({'threshold': float(threshold), "
                    "'percentile': THRESHOLD_PERCENTILE}, 'threshold.pkl')) "
                    "puis copiez-les dans monitoring/ml_models/."
                )

        _model = keras.models.load_model(model_path)
        _scaler = joblib.load(scaler_path)

        threshold_obj = joblib.load(threshold_path)
        if isinstance(threshold_obj, dict):
            _threshold = float(threshold_obj["threshold"])
            _threshold_percentile = threshold_obj.get("percentile")
        else:
            # Compatibilité avec une version antérieure où threshold.pkl
            # était un simple float.
            _threshold = float(threshold_obj)
            _threshold_percentile = None

        if os.path.exists(features_path):
            _feature_order = list(joblib.load(features_path))
        else:
            _feature_order = list(FALLBACK_FEATURE_ORDER)

    except Exception as exc:  # noqa: BLE001
        _load_error = str(exc)


def artifacts_ready():
    """True si le modèle/scaler/seuil sont chargés et utilisables."""
    _load_artifacts()
    return _load_error is None


def get_load_error():
    _load_artifacts()
    return _load_error


def get_threshold_cached():
    """Renvoie le seuil chargé (ou None si les artefacts ne sont pas prêts),
    sans lever d'exception — pratique pour l'affichage / le graphique."""
    _load_artifacts()
    return _threshold


def get_threshold_percentile():
    _load_artifacts()
    return _threshold_percentile


def get_feature_order():
    _load_artifacts()
    return _feature_order or list(FALLBACK_FEATURE_ORDER)


def build_feature_vector(raw: dict) -> np.ndarray:
    """Reconstruit les 10 features du modèle, dans l'ordre exact utilisé à
    l'entraînement (`feature_names.pkl`), à partir d'une observation brute.

    `raw` attend les clés :
        air_temperature (float, K)
        process_temperature (float, K)
        rotational_speed (float, rpm)
        torque (float, Nm)
        tool_wear (float, min)

    Note : `machine_type` (L/M/H) peut être présent dans `raw` pour être
    stocké/affiché dans l'interface, mais N'EST PAS utilisé par le modèle
    (cette version du notebook a abandonné cette variable).
    """
    air_temp = float(raw["air_temperature"])
    process_temp = float(raw["process_temperature"])
    rot_speed = float(raw["rotational_speed"])
    torque = float(raw["torque"])
    tool_wear = float(raw["tool_wear"])

    temperature_difference = process_temp - air_temp
    mechanical_power = torque * rot_speed * (2 * np.pi / 60)
    tool_stress = torque * tool_wear
    temperature_ratio = process_temp / air_temp
    torque_speed_ratio = torque / rot_speed

    computed = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "temperature_difference": temperature_difference,
        "mechanical_power": mechanical_power,
        "tool_stress": tool_stress,
        "temperature_ratio": temperature_ratio,
        "torque_speed_ratio": torque_speed_ratio,
    }

    feature_order = get_feature_order()
    try:
        feature_values = [computed[name] for name in feature_order]
    except KeyError as exc:
        raise RuntimeError(
            f"feature_names.pkl attend une colonne inconnue de l'interface : {exc}. "
            "Vérifiez que le notebook et ml_utils.py sont synchronisés."
        ) from exc

    return np.array(feature_values, dtype=float).reshape(1, -1)


def score_reading(raw: dict) -> dict:
    """Calcule le score d'anomalie (erreur de reconstruction), compare au
    seuil, et renvoie un diagnostic des variables les plus incriminées.

    Renvoie un dict :
        reconstruction_error, threshold, is_anomaly,
        feature_errors (liste triée [(nom_variable, erreur), ...]),
        top_variable, top_variable_error
    """
    _load_artifacts()
    if _load_error is not None:
        raise RuntimeError(_load_error)

    X = build_feature_vector(raw)
    X_scaled = _scaler.transform(X)

    X_reconstructed = _model.predict(X_scaled, verbose=0)

    squared_errors = (X_scaled - X_reconstructed) ** 2
    reconstruction_error = float(squared_errors.mean())

    feature_errors = list(zip(get_feature_order(), squared_errors[0].tolist()))
    feature_errors.sort(key=lambda item: item[1], reverse=True)

    top_variable, top_variable_error = feature_errors[0]
    is_anomaly = reconstruction_error > _threshold

    return {
        "reconstruction_error": reconstruction_error,
        "threshold": _threshold,
        "is_anomaly": is_anomaly,
        "feature_errors": feature_errors,
        "top_variable": top_variable,
        "top_variable_error": top_variable_error,
    }

