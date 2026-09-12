import pytest
import yaml
from pydantic import ValidationError

from to_agent.contracts import TOProblem, load_problem, save_problem

MINIMAL = """
design_domain: {type: box, min: [0, 0, 0], max: [10, 5, 5]}
supports:
  - {id: wall, region: {type: box, min: [-0.1, -1, -1], max: [0.1, 6, 6]}}
load_cases:
  - {id: tip, region: {type: box, min: [9.9, -1, -1], max: [10.1, 6, 0.1]}, force_N: [0, 0, -1]}
void:
  - type: difference
    a: {type: cylinder, center: [5, 2.5, 0], axis: z, r_max: 1.0, along: [0, 5]}
    b: {type: sphere, center: [5, 2.5, 2.5], radius: 0.5}
"""


def test_yaml_roundtrip(tmp_path):
    problem = TOProblem.model_validate(yaml.safe_load(MINIMAL))
    assert problem.void[0].type == "difference"
    assert problem.void[0].a.type == "cylinder"
    path = tmp_path / "p.yaml"
    save_problem(problem, path, header="hello")
    again = load_problem(path)
    assert again == problem


def test_relative_paths_resolved(tmp_path):
    (tmp_path / "cloud.xyz").write_text("0 0 0\n1 1 1\n")
    text = MINIMAL + "warm_start:\n  - {type: near_points, path: cloud.xyz, tol: 0.5}\n"
    path = tmp_path / "p.yaml"
    path.write_text(text)
    problem = load_problem(path)
    assert problem.warm_start[0].path == str((tmp_path / "cloud.xyz").resolve())


def test_validation_messages():
    data = yaml.safe_load(MINIMAL)
    data["supports"] = []
    with pytest.raises(ValidationError, match="at least one support"):
        TOProblem.model_validate(data)
    data = yaml.safe_load(MINIMAL)
    data["void"][0]["type"] = "blob"
    with pytest.raises(ValidationError, match="type"):
        TOProblem.model_validate(data)
