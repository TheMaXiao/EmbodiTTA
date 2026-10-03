import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.json"


def parse_configured_args(
    parser: argparse.ArgumentParser,
    section: str | Callable[[argparse.Namespace], str],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="JSON file containing defaults for this experiment",
    )
    preview, _ = parser.parse_known_args(argv)
    config_path = Path(preview.config)
    with config_path.open(encoding="utf-8") as config_file:
        config: Any = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a JSON object: {config_path}")

    section_name = section(preview) if callable(section) else section
    settings = config
    for part in section_name.split("."):
        if not isinstance(settings, dict) or part not in settings:
            raise ValueError(f"Missing configuration section '{section_name}' in {config_path}")
        settings = settings[part]
    if not isinstance(settings, dict):
        raise ValueError(f"Configuration section '{section_name}' must be a JSON object")

    valid_destinations = {action.dest for action in parser._actions}
    unknown = set(settings) - valid_destinations
    if unknown:
        raise ValueError(
            f"Unknown setting(s) in '{section_name}' of {config_path}: {', '.join(sorted(unknown))}"
        )
    parser.set_defaults(**settings)
    return parser.parse_args(argv)


def require_values(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    names: tuple[str, ...],
) -> None:
    missing = [name for name in names if getattr(args, name) in (None, "")]
    if missing:
        parser.error("set these values in the config or command line: " + ", ".join(missing))