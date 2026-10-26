"""Command line for the migration topology diagram."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from migration_topology.model import MigrationError, load_migration
from migration_topology.render import render_svg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="migration-topology",
        description="Render a device-migration topology diagram from a JSON description.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="render a migration description to SVG")
    render.add_argument("description", type=Path, help="JSON file describing the migration")
    render.add_argument(
        "-o",
        "--output",
        type=Path,
        help="SVG file to write. Omit to write the SVG to stdout.",
    )
    args = parser.parse_args(argv)

    try:
        migration = load_migration(args.description)
        svg = render_svg(migration)
        if args.output is None or str(args.output) == "-":
            sys.stdout.write(svg)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(svg, encoding="utf-8")
    except MigrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: {args.description}: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
