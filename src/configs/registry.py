"""Accessors for project configuration files."""

from __future__ import annotations

from src.configs.loader import PROJECT_ROOT, load_config
from src.configs.schemas import (
    AugmentationsConfig,
    BackboneConfig,
    DataModuleConfig,
    DatasetConfig,
    DinoConfig,
    EvaluationConfig,
    ExperimentConfig,
    LabelsConfig,
    MultitaskConfig,
    PathsConfig,
    RuntimeConfig,
    TasksConfig,
    TriageConfig,
)

# Data


def load_augmentations_config() -> AugmentationsConfig:
    """Load shared augmentation configuration."""
    return load_config(
        "data/augmentations.yaml",
        AugmentationsConfig,
    )


def load_dataset_config() -> DatasetConfig:
    """Load shared dataset preprocessing and split configuration."""
    return load_config(
        "data/dataset.yaml",
        DatasetConfig,
    )


def load_labels_config() -> LabelsConfig:
    """Load metadata and in-memory sample key configuration."""
    return load_config(
        "data/labels.yaml",
        LabelsConfig,
    )


def load_paths_config() -> PathsConfig:
    """Load project paths and resolve them against the project root."""
    return load_config(
        "data/paths.yaml",
        PathsConfig,
        path_root=PROJECT_ROOT,
    )


def load_tasks_config() -> TasksConfig:
    """Load supervised task target configuration."""
    return load_config(
        "data/tasks.yaml",
        TasksConfig,
    )


# DataModules


def load_dino_datamodule_config() -> DataModuleConfig:
    """Load DINO cache and DataLoader configuration."""
    return load_config(
        "datamodules/dino.yaml",
        DataModuleConfig,
    )


def load_probe_datamodule_config() -> DataModuleConfig:
    """Load linear-probe cache and DataLoader configuration."""
    return load_config(
        "datamodules/probe.yaml",
        DataModuleConfig,
    )


def load_multitask_datamodule_config() -> DataModuleConfig:
    """Load multitask cache and DataLoader configuration."""
    return load_config(
        "datamodules/multitask.yaml",
        DataModuleConfig,
    )


def load_inference_datamodule_config() -> DataModuleConfig:
    """Load inference and DataLoader configuration."""
    return load_config(
        "datamodules/inference.yaml",
        DataModuleConfig,
    )


# Models


def load_backbone_model_config() -> BackboneConfig:
    """Load shared backbone architecture configuration."""
    return load_config(
        "models/backbone.yaml",
        BackboneConfig,
    )


def load_dino_model_config() -> DinoConfig:
    """Load DINO model configuration."""
    return load_config(
        "models/dino.yaml",
        DinoConfig,
    )


def load_multitask_model_config() -> MultitaskConfig:
    """Load multitask model configuration."""
    return load_config(
        "models/multitask.yaml",
        MultitaskConfig,
    )


# Training


def load_dino_experiment_config() -> ExperimentConfig:
    """Load DINO training configuration."""
    return load_config(
        "training/dino.yaml",
        ExperimentConfig,
    )


def load_probe_experiment_config() -> ExperimentConfig:
    """Load linear-probe training configuration."""
    return load_config(
        "training/probe.yaml",
        ExperimentConfig,
    )


def load_multitask_experiment_config() -> ExperimentConfig:
    """Load multitask training configuration."""
    return load_config(
        "training/multitask.yaml",
        ExperimentConfig,
    )


# Runtime


def load_runtime_config() -> RuntimeConfig:
    """Load global runtime configuration."""
    return load_config(
        "runtime.yaml",
        RuntimeConfig,
    )


def load_triage_config() -> TriageConfig:
    """Load rule-based triage thresholds."""
    return load_config(
        "triage.yaml",
        TriageConfig,
    )


# Evaluation


def load_evaluation_config() -> EvaluationConfig:
    """Load post-processing and evaluation configuration."""
    return load_config(
        "evaluation.yaml",
        EvaluationConfig,
    )
