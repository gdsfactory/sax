import jax.numpy as jnp
import pytest

import sax


@pytest.mark.parametrize("backend", ["cuda", "default", "klu", "filipsson_gunnar"])
def test_backend(backend: str) -> None:
    if backend not in sax.backends.circuit_backends:
        pytest.skip(f"{backend} backend not available")

    instances = {
        "lft": {"component": "coupler"},
        "top": {"component": "wg"},
        "rgt": {"component": "mmi"},
    }
    nets = [
        {"p1": "lft,out0", "p2": "rgt,in0"},
        {"p1": "lft,out1", "p2": "top,in0"},
        {"p1": "top,out0", "p2": "rgt,in1"},
    ]
    ports = {"in0": "lft,in0", "out0": "rgt,out0"}
    models = {
        "wg": lambda: {
            ("in0", "out0"): -0.99477 - 0.10211j,
            ("out0", "in0"): -0.99477 - 0.10211j,
        },
        "mmi": lambda: {
            ("in0", "out0"): 0.7071067811865476,
            ("in0", "out1"): 0.7071067811865476j,
            ("in1", "out0"): 0.7071067811865476j,
            ("in1", "out1"): 0.7071067811865476,
            ("out0", "in0"): 0.7071067811865476,
            ("out1", "in0"): 0.7071067811865476j,
            ("out0", "in1"): 0.7071067811865476j,
            ("out1", "in1"): 0.7071067811865476,
        },
        "coupler": lambda: (
            jnp.array(
                [
                    [
                        5.19688622e-06 - 1.19777138e-05j,
                        6.30595625e-16 - 1.48061189e-17j,
                        -3.38542541e-01 - 6.15711852e-01j,
                        5.80662654e-03 - 1.11068866e-02j,
                        -3.38542542e-01 - 6.15711852e-01j,
                        -5.80662660e-03 + 1.11068866e-02j,
                    ],
                    [
                        8.59445189e-16 - 8.29783014e-16j,
                        -2.08640825e-06 + 8.17315497e-06j,
                        2.03847666e-03 - 2.10649131e-03j,
                        5.30509661e-01 + 4.62504708e-01j,
                        -2.03847666e-03 + 2.10649129e-03j,
                        5.30509662e-01 + 4.62504708e-01j,
                    ],
                    [
                        -3.38542541e-01 - 6.15711852e-01j,
                        2.03847660e-03 - 2.10649129e-03j,
                        7.60088070e-06 + 9.07340423e-07j,
                        2.79292426e-09 + 2.79093547e-07j,
                        5.07842364e-06 + 2.16385350e-06j,
                        -6.84244232e-08 - 5.00486817e-07j,
                    ],
                    [
                        5.80662707e-03 - 1.11068869e-02j,
                        5.30509661e-01 + 4.62504708e-01j,
                        2.79291895e-09 + 2.79093540e-07j,
                        -4.55645798e-06 + 1.50570403e-06j,
                        6.84244128e-08 + 5.00486817e-07j,
                        -3.55812153e-06 + 4.59781091e-07j,
                    ],
                    [
                        -3.38542541e-01 - 6.15711852e-01j,
                        -2.03847672e-03 + 2.10649131e-03j,
                        5.07842364e-06 + 2.16385349e-06j,
                        6.84244230e-08 + 5.00486816e-07j,
                        7.60088070e-06 + 9.07340425e-07j,
                        -2.79292467e-09 - 2.79093547e-07j,
                    ],
                    [
                        -5.80662607e-03 + 1.11068863e-02j,
                        5.30509662e-01 + 4.62504708e-01j,
                        -6.84244296e-08 - 5.00486825e-07j,
                        -3.55812153e-06 + 4.59781093e-07j,
                        -2.79293217e-09 - 2.79093547e-07j,
                        -4.55645798e-06 + 1.50570403e-06j,
                    ],
                ]
            ),
            {"in0": 0, "out0": 2, "out1": 4},
        ),
    }

    (
        analyze_instances,
        analyze_circuit,
        evaluate_circuit,
    ) = sax.backends.circuit_backends[backend]

    analyzed_instances = analyze_instances(instances, models)
    analyzed_circuit = analyze_circuit(analyzed_instances, nets, ports)
    sdict_backend = sax.sdict(
        evaluate_circuit(
            analyzed_circuit,
            {k: models[v["component"]]() for k, v in instances.items()},
        )
    )

    analyzed_instances = sax.backends.analyze_instances_klu(instances, models)  # type: ignore[arg-type]
    analyzed_circuit = sax.backends.analyze_circuit_klu(analyzed_instances, nets, ports)  # type: ignore[arg-type]
    sdict_klu = sax.sdict(
        sax.backends.evaluate_circuit_klu(
            analyzed_circuit,
            {k: models[v["component"]]() for k, v in instances.items()},
        )
    )

    # Compare to klu backend as source of truth
    for k in sdict_klu:
        val_klu = sdict_klu[k]
        val_backend = sdict_backend[k]
        assert abs(val_klu - val_backend) < 1e-5
