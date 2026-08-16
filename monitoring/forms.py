from django import forms
from .models import Reading


class ReadingForm(forms.Form):
    machine_type = forms.ChoiceField(
        choices=Reading.TYPE_CHOICES,
        label="Type de machine",
        initial="M",
    )
    air_temperature = forms.FloatField(
        label="Air temperature [K]", initial=298.1,
        widget=forms.NumberInput(attrs={"step": "0.1"}),
    )
    process_temperature = forms.FloatField(
        label="Process temperature [K]", initial=308.6,
        widget=forms.NumberInput(attrs={"step": "0.1"}),
    )
    rotational_speed = forms.FloatField(
        label="Rotational speed [rpm]", initial=1500,
        widget=forms.NumberInput(attrs={"step": "1"}),
    )
    torque = forms.FloatField(
        label="Torque [Nm]", initial=40.0,
        widget=forms.NumberInput(attrs={"step": "0.1"}),
    )
    tool_wear = forms.FloatField(
        label="Tool wear [min]", initial=0,
        widget=forms.NumberInput(attrs={"step": "1"}),
    )

    def clean_rotational_speed(self):
        value = self.cleaned_data["rotational_speed"]
        if value == 0:
            raise forms.ValidationError(
                "La vitesse de rotation ne peut pas être nulle "
                "(utilisée au dénominateur du ratio couple/vitesse)."
            )
        return value

    def clean_air_temperature(self):
        value = self.cleaned_data["air_temperature"]
        if value == 0:
            raise forms.ValidationError(
                "La température d'air ne peut pas être nulle "
                "(utilisée au dénominateur du ratio de température)."
            )
        return value
