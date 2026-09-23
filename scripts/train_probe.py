"""Run linear probing across cross-validation folds."""

from loguru import logger

from src.configs import (
    load_evaluation_config,
    load_probe_datamodule_config,
    load_probe_experiment_config,
)
from src.data.datamodules.probe import LinearProbeDataModule
from src.models.factory import load_dino_backbone
from src.training.probe import LinearProbeLightningModule
from src.utils.checkpoints import experiment_weights_path
from src.utils.constants import DINO_WEIGHTS
from src.utils.experiments import load_training_context, setup_experiment
from src.utils.io import require_file
from src.utils.trainer import cleanup_training_resources, get_fold_trainer


def main() -> None:
    """Run linear probing across cross-validation folds."""
    context = load_training_context()
    datamodule_config = load_probe_datamodule_config()
    experiment_config = load_probe_experiment_config()
    evaluation_config = load_evaluation_config()

    experiment_name = context.paths.experiments.probe
    n_splits = context.dataset.cross_validation.n_splits

    dino_weights = require_file(
        experiment_weights_path(
            context.paths,
            context.paths.experiments.dino,
            DINO_WEIGHTS,
        ),
        description="DINO backbone weights",
    )

    setup_experiment(
        name=experiment_name,
        paths=context.paths,
        runtime=context.runtime,
    )

    logger.info(
        "Starting linear probing | folds={} | cache_backend={}",
        n_splits,
        datamodule_config.cache.backend,
    )

    for fold in range(n_splits):
        logger.info(
            "Starting probe fold {}/{}",
            fold + 1,
            n_splits,
        )

        datamodule = LinearProbeDataModule(
            augmentations_config=context.augmentations,
            dataset_config=context.dataset,
            labels_config=context.labels,
            datamodule_config=datamodule_config,
            paths_config=context.paths,
            runtime_config=context.runtime,
            fold=fold,
        )

        datamodule.setup(stage="fit")
        mls_mean, mls_std = datamodule.get_mls_stats()

        logger.info(
            "Fold {}/{} MLS | mean={:.6f} | std={:.6f}",
            fold + 1,
            n_splits,
            mls_mean,
            mls_std,
        )

        model = LinearProbeLightningModule(
            load_dino_backbone(context.backbone, dino_weights),
            mls_mean,
            mls_std,
            labels_config=context.labels,
            experiment_config=experiment_config,
            evaluation_config=evaluation_config,
        )

        trainer = get_fold_trainer(
            experiment_name=experiment_name,
            fold=fold,
            experiment_config=experiment_config,
            tensorboard_dir=context.paths.artifacts.tensorboard,
            checkpoints_dir=context.paths.artifacts.checkpoints,
        )

        trainer.fit(
            model=model,
            datamodule=datamodule,
        )

        logger.success(
            "Probe fold {}/{} finished",
            fold + 1,
            n_splits,
        )

        del trainer, model, datamodule

        cleanup_training_resources(
            cache_backend=datamodule_config.cache.backend,
            cache_dir=context.paths.artifacts.cache,
            experiment_name=experiment_name,
        )

    logger.success(
        "Linear probing finished | folds={}",
        n_splits,
    )


if __name__ == "__main__":
    main()
