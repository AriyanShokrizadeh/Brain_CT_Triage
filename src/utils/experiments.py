"""Shared experiment setup and configuration context."""

from __future__ import annotations

from dataclasses import dataclass

from src.configs.registry import (
    load_augmentations_config,
    load_backbone_model_config,
    load_dataset_config,
    load_evaluation_config,
    load_inference_datamodule_config,
    load_labels_config,
    load_multitask_model_config,
    load_paths_config,
    load_runtime_config,
    load_tasks_config,
)
from src.configs.schemas import (
    AugmentationsConfig,
    BackboneConfig,
    DataModuleConfig,
    DatasetConfig,
    EvaluationConfig,
    LabelsConfig,
    MultitaskConfig,
    PathsConfig,
    RuntimeConfig,
    TasksConfig,
)
from src.utils.logging import configure_logging, log_runtime_summary
from src.utils.reproducibility import configure_reproducibility


@dataclass(frozen=True, slots=True)
class TrainingContext:
    """Configuration shared by all training pipelines."""

    paths: PathsConfig
    runtime: RuntimeConfig
    labels: LabelsConfig
    tasks: TasksConfig
    dataset: DatasetConfig
    augmentations: AugmentationsConfig
    backbone: BackboneConfig


def load_training_context() -> TrainingContext:
    """Load configuration shared by all training pipelines."""
    return TrainingContext(
        paths=load_paths_config(),
        runtime=load_runtime_config(),
        labels=load_labels_config(),
        tasks=load_tasks_config(),
        dataset=load_dataset_config(),
        augmentations=load_augmentations_config(),
        backbone=load_backbone_model_config(),
    )


@dataclass(frozen=True, slots=True)
class InferenceContext:
    """Configuration required by the inference pipeline."""

    paths: PathsConfig
    labels: LabelsConfig
    tasks: TasksConfig
    dataset: DatasetConfig
    backbone: BackboneConfig
    model: MultitaskConfig
    datamodule: DataModuleConfig
    evaluation: EvaluationConfig


def load_inference_context() -> InferenceContext:
    """Load inference configuration."""
    return InferenceContext(
        paths=load_paths_config(),
        labels=load_labels_config(),
        tasks=load_tasks_config(),
        dataset=load_dataset_config(),
        backbone=load_backbone_model_config(),
        model=load_multitask_model_config(),
        datamodule=load_inference_datamodule_config(),
        evaluation=load_evaluation_config(),
    )


def setup_experiment(
    *,
    name: str,
    paths: PathsConfig,
    runtime: RuntimeConfig,
) -> None:
    """Configure logging and reproducibility for an experiment."""
    configure_logging(
        log_dir=paths.artifacts.logs,
        experiment_name=name,
        runtime_config=runtime,
    )

    configure_reproducibility(runtime)
    log_runtime_summary(runtime)
