"""Central configuration for dataset columns and defaults."""

from __future__ import annotations

DEFAULT_RANDOM_STATE = 42

TARGET_COMPLICACIONES = "Complicaciones cardiovasculares"
TARGET_MORTALIDAD = "Mortalidad menor a 2 años"

COMPLETENESS_LEVELS = [1.0, 0.95, 0.9, 0.8]

ECHO_ZERO_TO_NAN_COLUMNS = [
    "ind vol final sistole",
    "índice E/E",
    "velocidad E´",
    "diametro AI",
    "TAPSE",
    "presion sVD",
    "índice de masa miocárdica",
    "indice vol final diastole",
]

ZERO_IMPUTE_COLUMNS = [
    "Años de fumador",
    "número de cigarrillos diarios",
    "IPA",
    "Mortalidad tiempo",
    "BB",
    "IECAs/ARAII",
]

OHE_COLUMNS = [
    "NYHA",
    "AHA",
    "estadocivil",
    "arearesidencia",
    "regimensalud",
    "educacion",
]

DROP_COMMON_COLUMNS = [
    "Primera dosis",
    "segunda dosis",
    "Tercera dosis",
    "Cuarta dosis",
    "Vacunación COVID",
    "Tipo vacuna",
    "Tipo vacuna.1",
    "Tipo vacuna.2",
    "Tipo vacuna.3",
    "Mortalidad tiempo",
    "Fuma actualmente",
    "número de cigarrillos diarios",
    "Años de fumador",
    "IMCat",
    "cod",
    "fechaingreso",
    "anioingreso",
    "Fecha aplicación",
    "Fecha aplicación.1",
    "Fecha aplicación.2",
    "Fecha aplicación.3",
    "Fecha mortalidad",
    "Fecha trasplante",
    "Fecha asistencia",
]

DROP_FOR_COMPLICACIONES = [
    "índice E/E",
    "velocidad E´",
    "presion sVD",
    "Complicaciones total",
    "Asistencia",
    "Trasplante",
    "ACV",
    "Mortalidad",
    "Mortalidad menor a 2 años",
]

DROP_FOR_MORTALIDAD = [
    "Complicaciones total",
    "Mortalidad",
    "Vacunación COVID",
    "Primera dosis",
    "segunda dosis",
    "Tercera dosis",
    "Cuarta dosis",
    "Fuma actualmente",
    "Años de fumador",
    "número de cigarrillos diarios",
    "presion sVD",
    "índice E/E",
    "velocidad E´",
]

SCALE_REQUIRED_MODELS = {"KNN", "SVM (RBF)"}

SAMPLING_METHODS = {None, "over", "under", "smote", "gm", "gmm", "dp-gnb"}
