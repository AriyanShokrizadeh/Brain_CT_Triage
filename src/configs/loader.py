"""YAML configuration loading."""

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _resolve_paths(config: object, root: Path) -> None:
    """Resolve Path fields relative to a root directory."""
    if not is_dataclass(config):
        return

    for field in fields(config):
        value = getattr(config, field.name)

        if isinstance(value, Path):
            setattr(config, field.name, (root / value).resolve())
        else:
            _resolve_paths(value, root)


def load_config[T](
    path: str | Path,
    schema: type[T],
    *,
    overrides: dict[str, Any] | None = None,
    path_root: str | Path | None = None,
) -> T:
    """Load a YAML file into a structured configuration."""
    path = (CONFIG_ROOT / path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    config = OmegaConf.merge(
        OmegaConf.structured(schema),
        OmegaConf.load(path),
        overrides or {},
    )

    result = cast(T, OmegaConf.to_object(config))

    if path_root is not None:
        _resolve_paths(result, Path(path_root).resolve())

    return result
