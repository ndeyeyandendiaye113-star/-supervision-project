"""
Simule un flux continu d'observations capteur, en insérant régulièrement
une nouvelle lecture scorée par l'auto-encodeur — pratique pour laisser
tourner la console de supervision en arrière-plan pendant une démo/soutenance.

Usage :
    python manage.py simulate_stream --count 50 --interval 2 --anomaly-rate 0.15
"""

import random
import time

from django.core.management.base import BaseCommand

from monitoring import ml_utils
from monitoring.models import Reading


class Command(BaseCommand):
    help = "Simule un flux d'observations capteur traitées par l'auto-encodeur."

    def add_arguments(self, parser):
        parser.add_argument("--count", type=int, default=30, help="Nombre de cycles à générer.")
        parser.add_argument("--interval", type=float, default=2.0, help="Secondes entre deux cycles.")
        parser.add_argument("--anomaly-rate", type=float, default=0.15, help="Proportion de cycles dégradés.")

    def handle(self, *args, **options):
        if not ml_utils.artifacts_ready():
            self.stderr.write(self.style.ERROR(
                f"Modèle indisponible : {ml_utils.get_load_error()}"
            ))
            return

        count = options["count"]
        interval = options["interval"]
        anomaly_rate = options["anomaly_rate"]

        for i in range(1, count + 1):
            degraded = random.random() < anomaly_rate

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

            result = ml_utils.score_reading(raw)

            Reading.objects.create(
                machine_type=raw["machine_type"],
                air_temperature=raw["air_temperature"],
                process_temperature=raw["process_temperature"],
                rotational_speed=raw["rotational_speed"],
                torque=raw["torque"],
                tool_wear=raw["tool_wear"],
                reconstruction_error=result["reconstruction_error"],
                threshold_used=result["threshold"],
                is_anomaly=result["is_anomaly"],
                top_variable=result["top_variable"],
                top_variable_error=result["top_variable_error"],
                feature_errors=result["feature_errors"],
            )

            status = "ANOMALIE" if result["is_anomaly"] else "normal"
            self.stdout.write(
                f"[{i}/{count}] err={result['reconstruction_error']:.5f} "
                f"seuil={result['threshold']:.5f} -> {status}"
            )

            if i < count:
                time.sleep(interval)

        self.stdout.write(self.style.SUCCESS(f"{count} cycles simulés."))
