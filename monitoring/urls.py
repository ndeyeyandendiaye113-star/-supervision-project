from django.urls import path
from . import views

app_name = "monitoring"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("chart-data/", views.chart_data, name="chart_data"),
    path("simulate/", views.simulate_reading, name="simulate"),
    path("reset/", views.reset_history, name="reset"),
]
