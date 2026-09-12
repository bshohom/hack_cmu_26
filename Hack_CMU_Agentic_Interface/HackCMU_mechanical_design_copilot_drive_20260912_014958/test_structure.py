"""StructureAgent must name load cases after GeometryOutput load regions."""

from __future__ import annotations

import unittest

from agents.geometry import GeometryAgent
from agents.structure import StructureAgent
from schemas import (
    AttachmentRegion,
    DesignEnvelope,
    EnvironmentGeometry,
    GeometryInput,
    GeometryOutput,
    LoadRegion,
    MassProvenance,
    ObjectGeometry,
    PartGeometry,
    StructureInput,
    UserRequirements,
)


def _geometry(load_name: str, kind: str = "strap") -> GeometryOutput:
    return GeometryOutput(
        is_mock=True,
        coordinate_frame="desk_edge_frame",
        environment=EnvironmentGeometry(desk_thickness_mm=20.0),
        payload_object=ObjectGeometry(kind=kind, bottle_diameter_mm=30.0, filled_mass_kg=5.0),
        design_envelope=DesignEnvelope(max_protrusion_mm=110.0, max_width_mm=60.0, max_height_mm=65.0),
        part=PartGeometry(shape="clamp_arm_hook", length_mm=110.0, width_mm=60.0, height_mm=65.0),
        attachment_regions=[AttachmentRegion(name="mount_contact", position_mm=(0.0, 0.0, 10.0))],
        load_regions=[LoadRegion(name=load_name, position_mm=(82.5, 0.0, -25.0))],
    )


def _run(geom: GeometryOutput):
    return StructureAgent().run(
        StructureInput(
            requirements=UserRequirements(),
            geometry=geom,
            payload_mass_kg=5.0,
            payload_mass_provenance=MassProvenance.USER_REQUIREMENTS,
        )
    )


class StructureLoadRegionTests(unittest.TestCase):
    def test_static_gravity_uses_strap_seat_for_hook(self) -> None:
        out = _run(_geometry("strap_seat", kind="strap"))
        self.assertEqual(out.load_cases[0].load_case_id, "static_gravity")
        self.assertEqual(out.load_cases[0].region_name, "strap_seat")
        self.assertEqual(out.load_regions[0].name, "strap_seat")
        self.assertNotEqual(out.load_cases[0].region_name, "cup_cavity")

    def test_static_gravity_uses_platform_for_shelf(self) -> None:
        out = _run(_geometry("platform", kind="box"))
        self.assertEqual(out.load_cases[0].region_name, "platform")

    def test_geometry_agent_strap_name_reaches_structure(self) -> None:
        req = UserRequirements()
        req.object_geometry.kind = "strap"
        req.object_geometry.bottle_diameter_mm = 30.0
        req.environment.desk_thickness_mm = 20.0
        req.design_envelope.max_protrusion_mm = 110.0
        req.payload.filled_mass_kg = 5.0
        geom = GeometryAgent().run(GeometryInput(requirements=req))
        self.assertEqual(geom.load_regions[0].name, "strap_seat")
        out = _run(geom)
        self.assertEqual(out.load_cases[0].region_name, "strap_seat")


if __name__ == "__main__":
    unittest.main()
