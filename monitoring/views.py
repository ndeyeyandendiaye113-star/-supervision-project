import random

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse

from . import ml_utils
from .forms import ReadingForm
from .models import Reading

HISTORY_LIMIT = 200  # nb de points conservés pour le graphique / la liste


def dashboard(request):
    """Console de supervision : formulaire de saisie, dernier statut,
    historique du score d'anomalie et diagnostic des variables."""
    form = ReadingForm()

    if request.method == "POST":
        form = ReadingForm(request.POST)
        if form.is_valid():
            _create_reading_from_form(form.cleaned_data)
            messages.success(request, "Observation traitée par l'auto-encodeur.")
            return redirect("monitoring:dashboard")

    readings = Reading.objects.all()[:HISTORY_LIMIT]
    latest = readings.first()

    ready = ml_utils.artifacts_ready()
    load_error = ml_utils.get_load_error()

    total_count = Reading.objects.count()
    anomaly_count = Reading.objects.filter(is_anomaly=True).count() if ready else 0

    context = {
        "form": form,
        "readings": readings,
        "latest": latest,
        "artifacts_ready": ready,
        "load_error": load_error,
        "anomaly_count": anomaly_count,
        "total_count": total_count,
    }
    return render(request, "monitoring/dashboard.html", context)


def chart_data(request):
    """Endpoint JSON consommé par Chart.js pour tracer le score dans le
    temps (le plus récent en dernier)."""
    readings = list(Reading.objects.all()[:HISTORY_LIMIT])
    readings.reverse()

    threshold = ml_utils.get_threshold_cached()

    data = {
        "labels": [r.timestamp.strftime("%H:%M:%S") for r in readings],
        "scores": [r.reconstruction_error for r in readings],
        "is_anomaly": [r.is_anomaly for r in readings],
        "threshold": threshold,
    }
    return JsonResponse(data)


def simulate_reading(request):
    """Génère une observation aléatoire (majoritairement normale, parfois
    dégradée) pour démontrer la console sans capteur réel connecté.
    Utile pour la soutenance / démo live."""
    if request.method != "POST":
        return redirect("monitoring:dashboard")

    degraded = random.random() < 0.15  # ~15% de cycles "dégradés" simulés

    if degraded:
        raw = {
            "machine_type": random.choice(["L", "M", "H"]),
            "air_temperature": random.uniform(300, 304),
            "process_temperature": random.uniform(311, 315),
            "rotational_speed": random.uniform(1150, 1350),
            "torque": random.uniform(58, 72),
            "tool_wear": random.uniform(200, 253),
        }
    else:
        raw = {
            "machine_type": random.choice(["L", "M", "H"]),
            "air_temperature": random.uniform(296, 302),
            "process_temperature": random.uniform(306, 312),
            "rotational_speed": random.uniform(1350, 1650),
            "torque": random.uniform(30, 50),
            "tool_wear": random.uniform(0, 200),
        }

    try:
        _create_reading_from_form(raw)
    except RuntimeError as exc:
        messages.error(request, f"Modèle indisponible : {exc}")

    return redirect("monitoring:dashboard")


def reset_history(request):
    if request.method == "POST":
        Reading.objects.all().delete()
        messages.info(request, "Historique réinitialisé.")
    return redirect("monitoring:dashboard")


def _create_reading_from_form(cleaned_data):
    result = ml_utils.score_reading(cleaned_data)

    Reading.objects.create(
        machine_type=cleaned_data["machine_type"],
        air_temperature=cleaned_data["air_temperature"],
        process_temperature=cleaned_data["process_temperature"],
        rotational_speed=cleaned_data["rotational_speed"],
        torque=cleaned_data["torque"],
        tool_wear=cleaned_data["tool_wear"],
        reconstruction_error=result["reconstruction_error"],
        threshold_used=result["threshold"],
        is_anomaly=result["is_anomaly"],
        top_variable=result["top_variable"],
        top_variable_error=result["top_variable_error"],
        feature_errors=result["feature_errors"],
    )
