"""Load a structured device-migration description.

Unknown ports and move orders are null. Callers label those gaps; this
module does not fill them in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MigrationError(ValueError):
    """The migration description is missing a required field or is the wrong type."""


@dataclass(frozen=True)
class Device:
    name: str
    os: str
    vendor: str | None = None
    nso_device_name: str | None = None
    nso_device_name_specified: bool = False


@dataclass(frozen=True)
class Link:
    name: str
    source_port: str | None
    target_port: str | None
    move_order: int | None
    note: str | None = None


@dataclass(frozen=True)
class Component:
    name: str
    role: str
    short_name: str | None = None


@dataclass(frozen=True)
class Controller:
    name: str
    workflow_manager: Component
    nso: Component
    short_name: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class Stage:
    order: int
    name: str
    template_kind: str
    summary: str | None = None


@dataclass(frozen=True)
class Network:
    label: str = "Network"
    note: str | None = None


@dataclass(frozen=True)
class Migration:
    source: Device
    target: Device
    links: tuple[Link, ...]
    controller: Controller
    stages: tuple[Stage, ...]
    title: str
    kind: str | None = None
    status: str | None = None
    blocked_on: tuple[str, ...] = ()
    network: Network = Network()


def load_migration(path: str | Path) -> Migration:
    """Load a migration description from a JSON file."""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    return migration_from_dict(data)


def migration_from_dict(data: Any) -> Migration:
    """Build a migration from a decoded JSON object."""
    root = _mapping(data, "migration")
    _check_keys(
        root,
        {
            "title",
            "kind",
            "status",
            "blocked_on",
            "network",
            "source",
            "target",
            "links",
            "controller",
            "stages",
        },
        "migration",
    )
    _require_keys(root, ("source", "target", "links", "controller", "stages"), "migration")

    source = _device(root["source"], "source")
    target = _device(root["target"], "target")
    links = _links(root["links"])
    controller = _controller(root["controller"])
    stages = _stages(root["stages"])
    network = _network(root["network"]) if "network" in root else Network()
    title = _optional_str(root.get("title"), "title") or f"{source.name} to {target.name}"

    return Migration(
        source=source,
        target=target,
        links=links,
        controller=controller,
        stages=stages,
        title=title,
        kind=_optional_str(root.get("kind"), "kind"),
        status=_optional_str(root.get("status"), "status"),
        blocked_on=_blocked_on(root.get("blocked_on", [])),
        network=network,
    )


def _device(value: Any, path: str) -> Device:
    obj = _mapping(value, path)
    _check_keys(obj, {"name", "os", "vendor", "nso_device_name"}, path)
    _require_keys(obj, ("name", "os"), path)
    specified = "nso_device_name" in obj
    return Device(
        name=_str(obj["name"], f"{path}.name"),
        os=_str(obj["os"], f"{path}.os"),
        vendor=_optional_str(obj.get("vendor"), f"{path}.vendor"),
        nso_device_name=_optional_str(obj["nso_device_name"], f"{path}.nso_device_name")
        if specified
        else None,
        nso_device_name_specified=specified,
    )


def _links(value: Any) -> tuple[Link, ...]:
    items = _list(value, "links")
    links: list[Link] = []
    for index, item in enumerate(items):
        path = f"links[{index}]"
        obj = _mapping(item, path)
        _check_keys(obj, {"name", "source_port", "target_port", "move_order", "note"}, path)
        _require_keys(obj, ("name", "source_port", "target_port", "move_order"), path)
        links.append(
            Link(
                name=_str(obj["name"], f"{path}.name"),
                source_port=_optional_str(obj["source_port"], f"{path}.source_port"),
                target_port=_optional_str(obj["target_port"], f"{path}.target_port"),
                move_order=_optional_int(obj["move_order"], f"{path}.move_order"),
                note=_optional_str(obj.get("note"), f"{path}.note"),
            )
        )
    orders = [link.move_order for link in links if link.move_order is not None]
    if len(orders) != len(set(orders)):
        raise MigrationError("links have duplicate move_order values")
    return tuple(links)


def _component(value: Any, path: str) -> Component:
    obj = _mapping(value, path)
    _check_keys(obj, {"name", "short_name", "role"}, path)
    _require_keys(obj, ("name", "role"), path)
    return Component(
        name=_str(obj["name"], f"{path}.name"),
        role=_str(obj["role"], f"{path}.role"),
        short_name=_optional_str(obj.get("short_name"), f"{path}.short_name"),
    )


def _controller(value: Any) -> Controller:
    obj = _mapping(value, "controller")
    _check_keys(obj, {"name", "short_name", "note", "workflow_manager", "nso"}, "controller")
    _require_keys(obj, ("name", "workflow_manager", "nso"), "controller")
    return Controller(
        name=_str(obj["name"], "controller.name"),
        short_name=_optional_str(obj.get("short_name"), "controller.short_name"),
        note=_optional_str(obj.get("note"), "controller.note"),
        workflow_manager=_component(obj["workflow_manager"], "controller.workflow_manager"),
        nso=_component(obj["nso"], "controller.nso"),
    )


def _stages(value: Any) -> tuple[Stage, ...]:
    items = _list(value, "stages")
    if not items:
        raise MigrationError("stages must contain at least one stage")
    stages: list[Stage] = []
    for index, item in enumerate(items):
        path = f"stages[{index}]"
        obj = _mapping(item, path)
        _check_keys(obj, {"order", "name", "template_kind", "summary"}, path)
        _require_keys(obj, ("order", "name", "template_kind"), path)
        stages.append(
            Stage(
                order=_int(obj["order"], f"{path}.order"),
                name=_str(obj["name"], f"{path}.name"),
                template_kind=_str(obj["template_kind"], f"{path}.template_kind"),
                summary=_optional_str(obj.get("summary"), f"{path}.summary"),
            )
        )
    orders = [stage.order for stage in stages]
    if len(orders) != len(set(orders)):
        raise MigrationError("stages have duplicate order values")
    return tuple(sorted(stages, key=lambda stage: stage.order))


def _network(value: Any) -> Network:
    obj = _mapping(value, "network")
    _check_keys(obj, {"label", "note"}, "network")
    label = _optional_str(obj.get("label"), "network.label") or "Network"
    return Network(label=label, note=_optional_str(obj.get("note"), "network.note"))


def _blocked_on(value: Any) -> tuple[str, ...]:
    items = _list(value, "blocked_on")
    return tuple(_str(item, f"blocked_on[{index}]") for index, item in enumerate(items))


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MigrationError(f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise MigrationError(f"{path} must be an array")
    return value


def _require_keys(obj: dict[str, Any], keys: tuple[str, ...], path: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise MigrationError(f"{path} is missing {', '.join(missing)}")


def _check_keys(obj: dict[str, Any], allowed: set[str], path: str) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        raise MigrationError(f"{path} has unknown field {extra[0]}")


def _str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MigrationError(f"{path} must be a non-empty string")
    return value.strip()


def _optional_str(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _str(value, path)


def _int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MigrationError(f"{path} must be a positive integer")
    return value


def _optional_int(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _int(value, path)
