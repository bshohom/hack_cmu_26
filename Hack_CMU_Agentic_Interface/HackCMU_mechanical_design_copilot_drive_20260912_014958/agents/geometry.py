"""Geometry Agent: simplified engineering geometry from requirements.

Does not turn an image into exact CAD.
"""

from __future__ import annotations

from schemas import (
    AttachmentRegion,
    DesignEnvelope,
    EnvironmentGeometry,
    GeometryInput,
    GeometryOutput,
    LoadRegion,
    ObjectGeometry,
    PartGeometry,
)

PLA_DENSITY_KG_MM3 = 1.24e-6


class GeometryAgent:
    """Placeholder geometry interpretation.

    TODO(Yujie): replace mock environment/support/attachment/load regions
    with real geometry representation.
    TODO(Aman): consume registration results / common coordinate frame.
    """

    def run(self, inp: GeometryInput) -> GeometryOutput:
        req = inp.requirements
        diameter = req.object_geometry.bottle_diameter_mm or 0.0
        height = req.object_geometry.bottle_height_mm or 250.0
        desk_t = req.environment.desk_thickness_mm or 0.0
        protrusion = req.design_envelope.max_protrusion_mm or 0.0
        contact = req.attachment.allowed_contact_region or "unspecified"

        environment = EnvironmentGeometry(
            kind="desk_plane",
            desk_thickness_mm=desk_t,
            surface_normal=req.environment.surface_normal,
        )
        payload_object = ObjectGeometry(
            kind="cylinder",
            bottle_diameter_mm=diameter,
            bottle_height_mm=height,
            filled_mass_kg=req.payload.filled_mass_kg,
        )
        envelope = DesignEnvelope(
            max_protrusion_mm=protrusion,
            max_width_mm=diameter * 1.6 if diameter else None,
            max_height_mm=desk_t + height * 0.4 if desk_t else None,
        )

        part_length = protrusion
        part_width = max(diameter * 1.6, 1.0)
        part_height = desk_t + 20.0
        volume = max(part_length * part_width * part_height * 0.15, 1.0)
        part = PartGeometry(
            shape="clamp_arm_ring",
            length_mm=part_length,
            width_mm=part_width,
            height_mm=part_height,
            volume_mm3=volume,
            estimated_mass_kg=volume * PLA_DENSITY_KG_MM3,
        )

        attachment_regions = [
            AttachmentRegion(
                name="mount_contact",
                position_mm=(0.0, 0.0, desk_t / 2.0),
                normal=(1.0, 0.0, 0.0),
                area_mm2=part_width * desk_t,
                notes=f"clamp/contact on {contact}",
            )
        ]
        load_regions = [
            LoadRegion(
                name="cup_cavity",
                position_mm=(protrusion * 0.85, 0.0, desk_t + 10.0),
                direction=(0.0, 0.0, -1.0),
                notes="bottle weight applied in cup ring/base",
            )
        ]

        frame = "design_local_frame"
        if inp.registration is not None:
            frame = inp.registration.frame_id
        elif inp.registration_frame:
            frame = inp.registration_frame

        return GeometryOutput(
            is_mock=True,
            coordinate_frame=frame,
            environment=environment,
            payload_object=payload_object,
            design_envelope=envelope,
            part=part,
            material=req.manufacturing.material,
            attachment_regions=attachment_regions,
            load_regions=load_regions,
            notes=(
                "Simplified engineering geometry from measurements. "
                "Not exact CAD from an image."
            ),
        )
