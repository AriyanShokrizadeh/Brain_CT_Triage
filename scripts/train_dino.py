"""Run DINO self-supervised pretraining."""

from loguru import logger

from src.configs import (
    load_dino_datamodule_config,
    load_dino_experiment_config,
    load_dino_model_config,
)
from src.data.datamodules.dino import DinoDataModule
from src.models.backbones.transunet import TransUNetBackbone2D
from src.training.dino import DinoLightningModule
from src.utils.checkpoints import (
    best_checkpoint,
    experiment_weights_path,
    export_checkpoint_weights,
)
from src.utils.constants import DINO_WEIGHTS
from src.utils.experiments import load_training_context, setup_experiment
from src.utils.trainer import cleanup_training_resources, get_trainer


def main() -> None:
    """Run DINO self-supervised pretraining."""
    context = load_training_context()

    datamodule_config = load_dino_datamodule_config()
    model_config = load_dino_model_config()
    experiment_config = load_dino_experiment_config()

    experiment_name = context.paths.experiments.dino

    setup_experiment(
        name=experiment_name,
        paths=context.paths,
        runtime=context.runtime,
    )

    datamodule = DinoDataModule(
        augmentations_config=context.augmentations,
        dataset_config=context.dataset,
        labels_config=context.labels,
        datamodule_config=datamodule_config,
        paths_config=context.paths,
        runtime_config=context.runtime,
    )

    model = DinoLightningModule(
        backbone=TransUNetBackbone2D(context.backbone),
        dino_config=model_config,
        augmentation_config=context.augmentations,
        labels_config=context.labels,
        experiment_config=experiment_config,
    )

    trainer = get_trainer(
        experiment_name=experiment_name,
        experiment_config=experiment_config,
        tensorboard_dir=context.paths.artifacts.tensorboard,
        checkpoints_dir=context.paths.artifacts.checkpoints,
    )

    logger.info("Starting DINO pretraining")

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )

    checkpoint = best_checkpoint(trainer)

    weights_path = export_checkpoint_weights(
        checkpoint.path,
        experiment_weights_path(
            context.paths,
            experiment_name,
            DINO_WEIGHTS,
        ),
        prefix="teacher.backbone.",
    )

    logger.success(
        "DINO pretraining finished | score={:.6f} | mode={} | weights={}",
        checkpoint.score,
        checkpoint.mode,
        weights_path,
    )

    del trainer, model, datamodule

    cleanup_training_resources(
        cache_backend=datamodule_config.cache.backend,
        cache_dir=context.paths.artifacts.cache,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    main()
