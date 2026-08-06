from __future__ import annotations

from dataclasses import dataclass
from importlib.abc import Traversable
from importlib.resources import files
from pathlib import Path
from typing import Literal

from local_rag_chatbot.app_errors import ConfigurationError

ResourceKind = Literal["file", "dir"]


@dataclass(frozen=True)
class ConfigResource:
    location: Traversable | Path
    source: Literal["packaged", "external"]
    description: str
    kind: ResourceKind

    @property
    def is_external(self) -> bool:
        return self.source == "external"

    def __fspath__(self) -> str:
        if isinstance(self.location, Path):
            return str(self.location)
        raise TypeError(f"{self.description} is a packaged resource, not a filesystem path")

    def __str__(self) -> str:
        return str(self.location)


def packaged_resource(*parts: str, description: str, kind: ResourceKind) -> ConfigResource:
    location: Traversable = files("local_rag_chatbot.resources")
    for part in parts:
        location = location / part
    return ConfigResource(location=location, source="packaged", description=description, kind=kind)


def external_resource(path: Path | str, *, description: str, kind: ResourceKind) -> ConfigResource:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise ConfigurationError(
            f"The configured {description} does not exist:\n{resolved}"
        )
    if kind == "file" and not resolved.is_file():
        raise ConfigurationError(
            f"The configured {description} is not a file:\n{resolved}"
        )
    if kind == "dir" and not resolved.is_dir():
        raise ConfigurationError(
            f"The configured {description} is not a directory:\n{resolved}"
        )
    return ConfigResource(location=resolved, source="external", description=description, kind=kind)


def resource_from_optional_path(
    path: Path | str | ConfigResource | None,
    *,
    env_value: str | None,
    packaged_parts: tuple[str, ...],
    description: str,
    kind: ResourceKind,
) -> ConfigResource:
    if isinstance(path, ConfigResource):
        return path
    if path is not None:
        return external_resource(path, description=description, kind=kind)
    if env_value:
        return external_resource(env_value, description=description, kind=kind)
    return packaged_resource(*packaged_parts, description=description, kind=kind)
