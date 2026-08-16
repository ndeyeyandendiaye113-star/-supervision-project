from django.contrib import admin
from .models import Reading


@admin.register(Reading)
class ReadingAdmin(admin.ModelAdmin):
    list_display = (
        "timestamp", "machine_type", "reconstruction_error",
        "threshold_used", "is_anomaly", "top_variable",
    )
    list_filter = ("is_anomaly", "machine_type")
    ordering = ("-timestamp",)
