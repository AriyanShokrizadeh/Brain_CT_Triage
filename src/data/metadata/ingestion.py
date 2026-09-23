"""Metadata loading, normalization, and partitioning."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from src.configs.schemas import DatasetConfig, LabelsConfig, PathsConfig
from src.utils.io import read_pickle


class MetadataIngestor:
    """Load, normalize, and split source metadata."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        labels_config: LabelsConfig,
        paths_config: PathsConfig,
    ) -> None:
        self.paths = paths_config
        self.cols = labels_config.metadata
        self.cv = dataset_config.cross_validation
        self.workers = dataset_config.processing.ingestion_workers

    @property
    def renames(self) -> dict[str, str]:
        """Map source columns to configured names."""
        sy, sx, st = self.cols.spacing

        return {
            "dicom_series.id": self.cols.series_id,
            "dicom_series.SOPInstanceUID": self.cols.slice_id,
            "dicom_series.PatientID": self.cv.group_column,
            "dicom_series.PixelSpacing0": sy,
            "dicom_series.PixelSpacing1": sx,
            "dicom_series.SliceThickness": st,
            "SkullFracture": self.cols.fracture_prob,
            "MidlineShiftMM": self.cols.MLS_mm,
            "RelativeAnnotationPath": self.cols.relative_path,
        }

    @property
    def areas(self) -> tuple[str, ...]:
        return tuple(f"{col}_Area" for col in self.cols.hemorrhage_columns)

    @property
    def unlabeled(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.cols.series_id,
                    self.cv.group_column,
                    self.cols.slice_id,
                    *self.cols.spacing,
                    self.cols.relative_path,
                )
            )
        )

    @property
    def labeled(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.unlabeled,
                    *self.cols.hemorrhage_columns,
                    *self.areas,
                    self.cols.fracture_prob,
                    self.cols.MLS_mm,
                    self.cv.stratify_column,
                    self.cols.triage_class,
                )
            )
        )

    @staticmethod
    def _norm_path(value: object) -> str:
        """Normalize a relative source path."""
        path = PureWindowsPath(str(value).strip())

        if path.anchor or ".." in path.parts:
            raise ValueError(f"Invalid relative path: {value!r}")

        if path.suffix.lower() in {".dcm", ".json"}:
            path = path.with_suffix("")

        return path.as_posix()

    @staticmethod
    def _series(root: Path) -> set[str]:
        """Return series directories."""
        if not root.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {root}")

        return {p.name for p in root.iterdir() if p.is_dir()}

    def _check_files(
        self,
        df: pd.DataFrame,
        root: Path,
        suffix: str,
    ) -> None:
        """Ensure referenced files exist."""
        paths = df[self.cols.relative_path].dropna().astype(str)

        missing = [p for p in paths if not (root / f"{p}{suffix}").is_file()]

        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} {suffix} files: {missing[:20]}"
            )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize source metadata."""
        df = df.copy()
        df.columns = df.columns.astype(str).str.strip()
        df.rename(columns=self.renames, inplace=True)

        missing = sorted(set(self.labeled) - set(df.columns))
        if missing:
            raise KeyError(f"Missing metadata columns: {missing}")

        df = df[list(self.labeled)].copy()

        ids = list(
            dict.fromkeys(
                (
                    self.cols.series_id,
                    self.cols.slice_id,
                    self.cv.group_column,
                )
            )
        )

        df[ids] = (
            df[ids]
            .astype("string")
            .apply(lambda col: col.str.strip())
            .replace("", pd.NA)
        )

        path = self.cols.relative_path
        df[path] = (
            df[path]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .map(self._norm_path, na_action="ignore")
        )

        floats = [
            *self.cols.spacing,
            *self.areas,
            self.cols.fracture_prob,
            self.cols.MLS_mm,
        ]
        ints = [
            *self.cols.hemorrhage_columns,
            self.cols.triage_class,
        ]

        df[floats] = df[floats].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        df[ints] = df[ints].apply(pd.to_numeric, errors="coerce")

        return df.reset_index(drop=True)

    def _split_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split train and prediction metadata."""
        col = self.cols.series_id
        train_ids = self._series(self.paths.data.train.dicoms)
        pred_ids = self._series(self.paths.data.predict.dicoms)

        if overlap := train_ids & pred_ids:
            raise ValueError(f"{len(overlap)} series exist in both train and predict.")

        ids = set(df[col].dropna().astype(str))

        if missing := ids - train_ids - pred_ids:
            raise FileNotFoundError(
                f"{len(missing)} metadata series have no DICOM directory."
            )

        train = df[df[col].isin(train_ids)].reset_index(drop=True)
        pred = df[df[col].isin(pred_ids)].reset_index(drop=True)

        logger.success(
            "Dataset split | train={:,} | predict={:,}",
            len(train),
            len(pred),
        )

        return train, pred

    def _has_annotations(self, series_id: str) -> bool:
        """Check whether every DICOM has an annotation."""
        dicom_dir = self.paths.data.train.dicoms / series_id
        ann_dir = self.paths.data.train.annotations / series_id

        if not dicom_dir.is_dir() or not ann_dir.is_dir():
            return False

        dicoms = {p.stem for p in dicom_dir.glob("*.dcm")}
        anns = {p.stem for p in ann_dir.glob("*.json")}

        return bool(dicoms) and dicoms == anns

    def _annotated(self, df: pd.DataFrame) -> set[str]:
        """Return fully annotated series."""
        ids = df[self.cols.series_id].dropna().astype(str).unique().tolist()

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            checks = pool.map(self._has_annotations, ids)

            result = {
                sid
                for sid, complete in tqdm(
                    zip(ids, checks, strict=True),
                    total=len(ids),
                    desc="Checking annotations",
                )
                if complete
            }

        logger.info(
            "Annotation check | annotated={:,}/{:,}",
            len(result),
            len(ids),
        )

        return result

    def _split_train(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split annotated and unannotated training data."""
        mask = df[self.cols.series_id].isin(self._annotated(df))

        annotated = df[mask].reset_index(drop=True)
        unannotated = df.loc[~mask, list(self.unlabeled)].reset_index(drop=True)

        logger.success(
            "Training split | annotated={:,} | unannotated={:,}",
            len(annotated),
            len(unannotated),
        )

        return annotated, unannotated

    def __call__(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load, normalize, validate, and split metadata."""
        df = self.clean(read_pickle(self.paths.data.source))
        train, pred = self._split_data(df)

        self._check_files(train, self.paths.data.train.dicoms, ".dcm")
        self._check_files(pred, self.paths.data.predict.dicoms, ".dcm")

        annotated, unannotated = self._split_train(train)

        self._check_files(annotated, self.paths.data.train.annotations, ".json")

        return annotated, unannotated, pred
