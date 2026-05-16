from functools import partial

import sax

SAMPLE_NETLIST = {
    "instances": {
        "lft": "coupler",
        "top": "waveguide",
        "btm": "waveguide",
        "rgt": "coupler",
    },
    "connections": {
        "lft,out0": "btm,in0",
        "btm,out0": "rgt,in0",
        "lft,out1": "top,in0",
        "top,out0": "rgt,in1",
    },
    "ports": {
        "in0": "lft,in0",
        "in1": "lft,in1",
        "out0": "rgt,out0",
        "out1": "rgt,out1",
    },
}

SAMPLE_NETLIST2 = {
    "instances": {
        "lft": sax.models.coupler_ideal,
        "top": partial(sax.models.straight, length=10.0),
        "btm": partial(sax.models.straight, length=30.0),
        "rgt": sax.models.coupler_ideal,
    },
    "connections": {
        "lft,out0": "btm,in0",
        "btm,out0": "rgt,in0",
        "lft,out1": "top,in0",
        "top,out0": "rgt,in1",
    },
    "ports": {
        "in0": "lft,in0",
        "in1": "lft,in1",
        "out0": "rgt,out0",
        "out1": "rgt,out1",
    },
}


def test_netlist_function_from_dict() -> None:
    net = sax.netlist(SAMPLE_NETLIST)
    assert "top_level" in net


def test_netlist_using_instance_functions() -> None:
    net = sax.netlist(SAMPLE_NETLIST2)
    assert "top_level" in net
