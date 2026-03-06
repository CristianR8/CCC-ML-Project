"""Runnable pipelines wrapping the extracted methods."""

from .complicaciones import run_complicaciones_pipeline
from .mortalidad import run_mortalidad_pipeline
from .entrenamiento import run_entrenamiento_pipeline
from .exploracion import run_exploracion_pipeline

__all__ = [
    "run_complicaciones_pipeline",
    "run_mortalidad_pipeline",
    "run_entrenamiento_pipeline",
    "run_exploracion_pipeline",
]
