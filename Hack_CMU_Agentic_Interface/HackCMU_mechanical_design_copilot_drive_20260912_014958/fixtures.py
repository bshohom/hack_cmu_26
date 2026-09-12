"""Load typed cup-holder integration fixtures.

These files are hypothetical teammate outputs, not algorithm implementations.
"""

from __future__ import annotations

import json
from pathlib import Path

from schemas import (
    AnalysisOutput,
    CadOutput,
    GeometryOutput,
    IntegrationFixtures,
    RegistrationOutput,
    RequirementsUpdate,
    TopologyOutput,
    VerificationResult,
)

EXAMPLE_DIR = Path(__file__).resolve().parent / "examples" / "cup_holder"


def _read(name: str) -> str:
    return (EXAMPLE_DIR / name).read_text()


def load_user_request() -> str:
    payload = json.loads(_read("00_user_request.json"))
    return payload["message"]


def load_clarifications() -> RequirementsUpdate:
    return RequirementsUpdate.model_validate_json(_read("01_user_clarifications.json"))


def load_integration_fixtures() -> IntegrationFixtures:
    return IntegrationFixtures(
        registration=RegistrationOutput.model_validate_json(_read("02_registration_output.json")),
        geometry=GeometryOutput.model_validate_json(_read("03_geometry_output.json")),
        analysis=AnalysisOutput.model_validate_json(_read("05_analysis_output.json")),
        topology=TopologyOutput.model_validate_json(_read("06_topology_output.json")),
        cad=CadOutput.model_validate_json(_read("08_cad_output.json")),
    )


def load_expected_structure():
    from schemas import StructureOutput

    return StructureOutput.model_validate_json(_read("04_structure_output.json"))


def load_expected_verification() -> VerificationResult:
    return VerificationResult.model_validate_json(_read("07_verification_output.json"))
