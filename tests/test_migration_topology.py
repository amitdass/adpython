"""Facts for the ASR 920 sample stay unlabeled, and the renderer follows the input."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from migration_topology.model import MigrationError, load_migration, migration_from_dict
from migration_topology.render import render_svg

SAMPLE = ROOT / "samples" / "asr920-to-ncs540.json"
SAMPLE_SVG = ROOT / "samples" / "asr920-to-ncs540.svg"

FORBIDDEN = (
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    r"GigabitEthernet",
    r"TenGigE",
    r"HundredGigE",
    r"Bundle-Ether",
    r"Loopback",
    r"\bGi\d",
    r"\bTe\d",
    r"\bHu\d",
    r"49\.\d{4}",
    r"prefix-sid",
    r"(?i)route-target",
    r"\bVRF\b",
    r"(?i)xconnect",
)


class SampleFactsTest(unittest.TestCase):
    def test_sample_encodes_only_known_facts(self) -> None:
        raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
        migration = load_migration(SAMPLE)

        self.assertEqual(migration.kind, "like-for-like")
        self.assertEqual(migration.status, "blocked")
        self.assertEqual(migration.source.name, "ASR 920")
        self.assertEqual(migration.source.vendor, "Cisco")
        self.assertEqual(migration.source.os, "IOS XE")
        self.assertIsNone(migration.source.nso_device_name)
        self.assertTrue(migration.source.nso_device_name_specified)
        self.assertEqual(migration.target.name, "NCS 540")
        self.assertEqual(migration.target.os, "IOS XR")
        self.assertIsNone(migration.target.nso_device_name)

        self.assertEqual(len(migration.links), 2)
        self.assertEqual(
            [link.name for link in migration.links],
            ["Ethernet uplink 1", "Ethernet uplink 2"],
        )
        for link, raw_link in zip(migration.links, raw["links"], strict=True):
            self.assertIsNone(link.source_port)
            self.assertIsNone(link.target_port)
            self.assertIsNone(link.move_order)
            self.assertIsNone(raw_link["source_port"])
            self.assertIsNone(raw_link["target_port"])
            self.assertIsNone(raw_link["move_order"])

        self.assertEqual(
            [stage.name for stage in migration.stages],
            ["Uplink move", "IS-IS", "SR-MPLS", "VPN services"],
        )
        self.assertEqual(
            [stage.template_kind for stage in migration.stages[:3]],
            ["device template", "device template", "device template"],
        )
        self.assertEqual(
            migration.stages[3].template_kind,
            "service template or L2VPN/L3VPN package",
        )
        self.assertEqual(migration.controller.name, "Cisco Crosswork Network Controller")
        self.assertEqual(migration.controller.short_name, "CNC")
        self.assertEqual(migration.controller.workflow_manager.short_name, "CWM")
        self.assertEqual(migration.controller.workflow_manager.name, "Crosswork Workflow Manager")
        self.assertEqual(migration.controller.nso.name, "NSO")
        self.assertEqual(migration.blocked_on, ("the real running-config", "the port map"))

    def test_sample_diagram_labels_gaps_and_invents_nothing(self) -> None:
        svg = render_svg(load_migration(SAMPLE))
        for pattern in FORBIDDEN:
            self.assertIsNone(re.search(pattern, svg), pattern)
        for phrase in (
            "ASR 920",
            "NCS 540",
            "IOS XE",
            "IOS XR",
            "Cisco Crosswork Network Controller",
            "CNC",
            "Crosswork Workflow Manager",
            "CWM",
            "NSO",
            "IS-IS",
            "SR-MPLS",
            "L2VPN",
            "L3VPN",
            "device template",
            "service template or",
            "L2VPN/L3VPN package",
            "unlabeled",
            "running-config",
            "port map",
            "not recorded",
            "CURRENT",
            "TARGET",
            "Ethernet uplink 1",
            "Ethernet uplink 2",
        ):
            self.assertIn(phrase, svg)
        self.assertGreaterEqual(svg.count(">unlabeled<"), 4)
        ET.fromstring(svg)

    def test_checked_in_sample_svg_matches_renderer(self) -> None:
        self.assertTrue(SAMPLE_SVG.is_file(), "generate samples/asr920-to-ncs540.svg from the sample input")
        generated = render_svg(load_migration(SAMPLE))
        self.assertEqual(SAMPLE_SVG.read_text(encoding="utf-8"), generated)


class RendererBehaviorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(SAMPLE.read_text(encoding="utf-8"))

    def test_supplied_port_is_shown_and_unknown_port_stays_unlabeled(self) -> None:
        self.data["links"] = [
            {
                "name": "Uplink A",
                "source_port": "SOURCE-PORT-TOKEN",
                "target_port": None,
                "move_order": 1,
            }
        ]
        svg = render_svg(migration_from_dict(self.data))
        self.assertIn("SOURCE-PORT-TOKEN", svg)
        self.assertIn(">unlabeled<", svg)
        self.assertIn("move order 1", svg)
        self.assertNotIn("move order not recorded", svg)

    def test_text_is_escaped(self) -> None:
        self.data["source"]["name"] = "A & B <C>"
        svg = render_svg(migration_from_dict(self.data))
        self.assertIn("A &amp; B &lt;C&gt;", svg)
        self.assertNotIn("A & B <C>", svg)

    def test_stages_follow_order_field(self) -> None:
        self.data["stages"] = list(reversed(self.data["stages"]))
        migration = migration_from_dict(self.data)
        self.assertEqual([stage.order for stage in migration.stages], [1, 2, 3, 4])
        svg = render_svg(migration)
        self.assertLess(svg.index(">1<"), svg.index(">2<"))
        self.assertLess(svg.index("Uplink move"), svg.index("SR-MPLS"))

    def test_duplicate_stage_order_is_rejected(self) -> None:
        self.data["stages"][1]["order"] = 1
        with self.assertRaises(MigrationError):
            migration_from_dict(self.data)

    def test_missing_port_field_is_rejected(self) -> None:
        del self.data["links"][0]["source_port"]
        with self.assertRaises(MigrationError):
            migration_from_dict(self.data)

    def test_boolean_move_order_is_rejected(self) -> None:
        self.data["links"][0]["move_order"] = True
        with self.assertRaises(MigrationError):
            migration_from_dict(self.data)

    def test_cli_writes_svg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.svg"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "migration_topology",
                    "render",
                    str(SAMPLE),
                    "-o",
                    str(out),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("<svg", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
