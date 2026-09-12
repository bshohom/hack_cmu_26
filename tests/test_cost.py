from to_agent.contracts import BoxRegion, LoadCase, Support, TOProblem
from to_agent.cost import estimate_cost, fit_power_law


def problem(h: float) -> TOProblem:
    return TOProblem(
        design_domain=BoxRegion(min=(0, 0, 0), max=(100, 50, 50)),
        supports=[Support(id="a", region=BoxRegion(min=(0, 0, 0), max=(1, 50, 50)))],
        load_cases=[LoadCase(id="b", region=BoxRegion(min=(99, 0, 0), max=(100, 50, 1)), force_N=(0, 0, -1))],
        target_element_size=h,
    )


def test_estimate_monotone_and_levels():
    coarse = estimate_cost(problem(5.0), device="cpu")
    fine = estimate_cost(problem(2.0), device="cpu")
    tiny = estimate_cost(problem(0.4), device="cpu")
    assert coarse.n_elem < fine.n_elem < tiny.n_elem
    assert coarse.total_sec < fine.total_sec < tiny.total_sec
    assert coarse.level == "ok"
    assert tiny.level == "too_big"
    assert "target_element_size" in tiny.message


def test_fit_power_law():
    a, b = fit_power_law([1000, 10000, 100000], [0.01, 0.2, 4.0])
    assert 1.2 < b < 1.4
    assert a > 0
