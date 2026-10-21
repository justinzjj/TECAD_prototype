"""Repository-relative paths shared by the prototype, experiments and plots."""

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    """Load defaults and optionally overlay a YAML file; resolve from repo root."""
    with (ROOT / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    if path is not None:
        custom = Path(path).expanduser()
        if not custom.is_absolute():
            custom = ROOT / custom
        with custom.open() as stream:
            override = yaml.safe_load(stream) or {}
        if not isinstance(override, dict):
            raise ValueError("Configuration must be a mapping")
        config["paths"].update(override.pop("paths", {}))
        config.update(override)
    for name, value in config["paths"].items():
        p = Path(value).expanduser()
        config["paths"][name] = (p if p.is_absolute() else ROOT / p).resolve()
    return config
