"""Competition model API for Brain CT Triage inference."""

from pathlib import Path

from src.inference.predict import SeriesPredictor


class BrainCTTriageModel:
    """Predict the seven intermediate outputs for one CT series."""

    def __init__(self) -> None:
        self.predictor: SeriesPredictor | None = None

    def predict(self, study_dir: str | Path) -> dict[str, float]:
        """Predict one complete CT series."""
        if self.predictor is None:
            self.predictor = SeriesPredictor()

        prediction = self.predictor.predict(study_dir)

        return prediction.as_dict()
