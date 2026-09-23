"""High-level CT series prediction API."""

from pathlib import Path

import torch

from src.data.series import load_series_info
from src.data.transforms.pipelines import build_inference_pipeline
from src.inference.engine import InferenceEngine
from src.inference.results import SeriesPrediction
from src.models.factory import load_multitask_model
from src.utils.checkpoints import experiment_weights_path
from src.utils.constants import MULTITASK_WEIGHTS
from src.utils.experiments import load_inference_context
from src.utils.io import require_file


class SeriesPredictor:
    """Predict intermediate outputs for complete CT series."""

    def __init__(self) -> None:
        context = load_inference_context()

        weights_path = require_file(
            experiment_weights_path(
                context.paths,
                context.paths.experiments.multitask,
                MULTITASK_WEIGHTS,
            ),
            description="Multitask model weights",
        )

        model = load_multitask_model(
            context.backbone,
            context.model,
            context.tasks,
            context.labels,
            weights_path,
        )

        transform = build_inference_pipeline(
            context.dataset,
            context.labels,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.engine = InferenceEngine(
            model,
            transform,
            device,
            labels_config=context.labels,
            tasks_config=context.tasks,
            datamodule_config=context.datamodule,
            evaluation_config=context.evaluation,
            paths_config=context.paths,
        )

    def predict(
        self,
        series_dir: str | Path,
    ) -> SeriesPrediction:
        """Predict one CT series."""
        slices = load_series_info(series_dir)
        return self.engine.predict(slices)
