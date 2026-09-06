"""Shared configuration, read on demand without creating files or directories."""

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_root: Path


def get_settings() -> Settings:
    """Read pdmdata.toml; resolve relative data paths beside that file.

    Set PDMDATA_CONFIG to select a different file.
    Configuration is reread on each call, including in an existing notebook.
    """
    override = os.environ.get("PDMDATA_CONFIG")
    if override is not None:
        if not override.strip():
            raise ValueError("PDMDATA_CONFIG must be a non-empty file path")
        config_path = Path(override).expanduser().resolve()
    else:
        # An editable checkout uses its own configuration from any working directory.
        base = PROJECT_ROOT if (PROJECT_ROOT / "pdmdata.toml").is_file() else Path.cwd()
        config_path = base / "pdmdata.toml"

    try:
        with config_path.open("rb") as stream:
            values = tomllib.load(stream)
    except FileNotFoundError:
        if override is not None:
            raise FileNotFoundError(f"Configuration file not found: {config_path}") from None
        values = {}

    unknown = values.keys() - {"data_root"}
    if unknown:
        raise ValueError(f"Unknown settings in {config_path}: {', '.join(sorted(unknown))}")
    data_root = values.get("data_root", "data/raw")
    if not isinstance(data_root, str) or not data_root.strip():
        raise ValueError(f"data_root in {config_path} must be a non-empty string")
    root = Path(data_root).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    return Settings(data_root=root.resolve())
