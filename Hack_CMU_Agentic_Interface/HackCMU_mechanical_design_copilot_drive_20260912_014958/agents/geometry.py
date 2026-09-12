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

# Per payload kind: (part shape, load region name, how the payload bears on the part).
PAYLOAD_KINDS = {
    "cylinder": ("clamp_arm_ring", "cup_cavity", "payload weight in the cup ring/base"),
    "strap": ("clamp_arm_hook", "strap_seat", "strap weight distributed over the hook arm contact patch"),
    "box": ("clamp_shelf", "platform", "payload weight distributed over the flat platform"),
}


class GeometryAgent:
    """Placeholder geometry interpretation.

    TODO(Yujie): replace mock environment/support/attachment/load regions
    with real geometry representation.
    TODO(Aman): consume registration results / common coordinate frame.
    """

    def run(self, inp: GeometryInput) -> GeometryOutput:
        req = inp.requirements
        kind = req.object_geometry.kind if req.object_geometry.kind in PAYLOAD_KINDS else "cylinder"
        shape, load_name, load_note = PAYLOAD_KINDS[kind]
        size = req.object_geometry.bottle_diameter_mm or 0.0
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
            kind=kind,
            bottle_diameter_mm=size,
            bottle_height_mm=height,
            filled_mass_kg=req.payload.filled_mass_kg,
        )

        # Where the payload bears on the part, and how far the part reaches, depend on the
        # payload kind: a cup sits above the desk, a bag hangs below it, a shelf lifts above it.
        if kind == "strap":
            load_x = protrusion * 0.75
            load_z = -max(25.0, desk_t)
            part_height = desk_t + abs(load_z) + 20.0
            part_width = max(size * 2.0, 40.0)
        elif kind == "box":
            load_x = protrusion * 0.45
            load_z = desk_t + max(40.0, height)
            part_height = load_z + 10.0
            part_width = max(size * 1.4, 60.0)
        else:
            load_x = protrusion * 0.85
            load_z = desk_t + 10.0
            part_height = desk_t + 20.0
            part_width = max(size * 1.6, 1.0)

        envelope = DesignEnvelope(
            max_protrusion_mm=protrusion,
            max_width_mm=part_width if size else None,
            max_height_mm=part_height if desk_t else None,
        )
        volume = max(protrusion * part_width * part_height * 0.15, 1.0)
        part = PartGeometry(
            shape=shape,
            length_mm=protrusion,
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
                name=load_name,
                position_mm=(load_x, 0.0, load_z),
                direction=(0.0, 0.0, -1.0),
                notes=load_note,
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
                f"Simplified engineering geometry from measurements ({kind} payload). "
                "Not exact CAD from an image."
            ),
        )
