from django.db import models


class Reading(models.Model):
    """Une observation capteur (un 'cycle' machine) et le résultat de
    l'auto-encodeur pour cette observation."""

    TYPE_CHOICES = [
        ("L", "L (faible qualité)"),
        ("M", "M (qualité moyenne)"),
        ("H", "H (haute qualité)"),
    ]

    timestamp = models.DateTimeField(auto_now_add=True)

    # --- Variables brutes saisies / reçues du capteur ---
    machine_type = models.CharField(max_length=1, choices=TYPE_CHOICES, default="M")
    air_temperature = models.FloatField(help_text="Air temperature [K]")
    process_temperature = models.FloatField(help_text="Process temperature [K]")
    rotational_speed = models.FloatField(help_text="Rotational speed [rpm]")
    torque = models.FloatField(help_text="Torque [Nm]")
    tool_wear = models.FloatField(help_text="Tool wear [min]")

    # --- Sortie de l'auto-encodeur ---
    reconstruction_error = models.FloatField()
    threshold_used = models.FloatField()
    is_anomaly = models.BooleanField(default=False)

    # --- Diagnostic : variable la plus incriminée ---
    top_variable = models.CharField(max_length=64, blank=True)
    top_variable_error = models.FloatField(default=0.0)
    feature_errors = models.JSONField(
        default=list,
        blank=True,
        help_text="Liste [[nom_variable, erreur], ...] triée par contribution décroissante",
    )

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        status = "ANOMALIE" if self.is_anomaly else "normal"
        return f"[{self.timestamp:%H:%M:%S}] err={self.reconstruction_error:.4f} ({status})"
