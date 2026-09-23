"""Train the supervised multitask model across cross-validation folds."""

from loguru import logger

from src.configs import (
    load_multitask_datamodule_config,
    load_multitask_experiment_config,
    load_multitask_model_config,
)
from src.data.datamodules.multitask import MultiTaskDataModule
from src.models.factory import load_dino_backbone
from src.models.multitask.network import MultiTaskNet
from src.training.multitask import MultiTaskLightningModule
from src.utils.checkpoints import (
    BestCheckpoint,
    best_checkpoint,
    experiment_weights_path,
    export_checkpoint_weights,
    is_better_checkpoint,
)
from src.utils.constants import DINO_WEIGHTS, MULTITASK_WEIGHTS
from src.utils.experiments import load_training_context, setup_experiment
from src.utils.io import require_file
from src.utils.trainer import cleanup_training_resources, get_fold_trainer


def main() -> None:
    """Train all folds and export the best multitask model."""
    context = load_training_context()
    datamodule_config = load_multitask_datamodule_config()
    model_config = load_multitask_model_config()
    experiment_config = load_multitask_experiment_config()

    experiment_name = context.paths.experiments.multitask
    n_splits = context.dataset.cross_validation.n_splits

    dino_weights = require_file(
        experiment_weights_path(
            context.paths,
            context.paths.experiments.dino,
            DINO_WEIGHTS,
        ),
        description="DINO backbone weights",
    )

    output_weights = experiment_weights_path(
        context.paths,
        experiment_name,
        MULTITASK_WEIGHTS,
    )

    setup_experiment(
        name=experiment_name,
        paths=context.paths,
        runtime=context.runtime,
    )

    logger.info(
        "Starting multitask training | folds={} | " "cache_backend={} | batch_size={}",
        n_splits,
        datamodule_config.cache.backend,
        datamodule_config.dataloader.batch_size,
    )

    best_checkpoint_info: BestCheckpoint | None = None
    best_fold: int | None = None

    for fold in range(n_splits):
        logger.info(
            "Starting multitask fold {}/{}",
            fold + 1,
            n_splits,
        )

        network = MultiTaskNet(
            model_config,
            context.tasks,
            context.labels,
            backbone=load_dino_backbone(
                context.backbone,
                dino_weights,
            ),
        )

        model = MultiTaskLightningModule(
            model=network,
            labels_config=context.labels,
            task_config=context.tasks,
            experiment_config=experiment_config,
        )

        datamodule = MultiTaskDataModule(
            augmentations_config=context.augmentations,
            dataset_config=context.dataset,
            labels_config=context.labels,
            datamodule_config=datamodule_config,
            paths_config=context.paths,
            tasks_config=context.tasks,
            runtime_config=context.runtime,
            fold=fold,
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

        checkpoint = best_checkpoint(trainer)

        logger.info(
            "Fold {}/{} finished | score={:.6f} | mode={}",
            fold + 1,
            n_splits,
            checkpoint.score,
            checkpoint.mode,
        )

        if is_better_checkpoint(checkpoint, best_checkpoint_info):
            best_checkpoint_info = checkpoint
            best_fold = fold

            logger.success(
                "New best model | fold={}/{} | score={:.6f}",
                fold + 1,
                n_splits,
                checkpoint.score,
            )

        del trainer, model, network, datamodule

        cleanup_training_resources(
            cache_backend=datamodule_config.cache.backend,
            cache_dir=context.paths.artifacts.cache,
            experiment_name=experiment_name,
        )

    if best_checkpoint_info is None or best_fold is None:
        raise RuntimeError("No multitask model was selected.")

    exported_weights = export_checkpoint_weights(
        best_checkpoint_info.path,
        output_weights,
        prefix="model.",
    )

    logger.success(
        "Multitask training finished | best_fold={}/{} | "
        "score={:.6f} | mode={} | weights={}",
        best_fold + 1,
        n_splits,
        best_checkpoint_info.score,
        best_checkpoint_info.mode,
        exported_weights,
    )


if __name__ == "__main__":
    main()
