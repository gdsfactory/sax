"""Test S-matrix convention: SDense S[i,j] = output at i for input at j."""

import jax.numpy as jnp
import numpy as np

import sax

# Non-reciprocal reference: forward transmission only (in->out), no reverse
SDICT_REF: sax.SDict = {
    ("in", "in"): jnp.array(0.0 + 0.0j),
    ("in", "out"): jnp.array(1.0 + 0.0j),  # from in to out
    ("out", "in"): jnp.array(0.0 + 0.0j),  # NO reverse path
    ("out", "out"): jnp.array(0.0 + 0.0j),
}


# --- SDense convention ---


def test_sdense_convention() -> None:
    """S[out_idx, in_idx] should hold the in->out transmission."""
    S, pm = sax.sdense(SDICT_REF)
    assert float(S[pm["out"], pm["in"]].real) == 1.0
    assert float(S[pm["in"], pm["out"]].real) == 0.0


# --- Round-trips: every (source, target) pair ---


def test_sdict_to_scoo_to_sdict() -> None:
    result = sax.sdict(sax.scoo(SDICT_REF))
    for k, v in SDICT_REF.items():
        np.testing.assert_allclose(float(result[k].real), float(complex(v).real))


def test_sdict_to_sdense_to_sdict() -> None:
    result = sax.sdict(sax.sdense(SDICT_REF))
    for k, v in SDICT_REF.items():
        np.testing.assert_allclose(float(result[k].real), float(complex(v).real))


def test_scoo_to_sdict_to_scoo() -> None:
    scoo1 = sax.scoo(SDICT_REF)
    scoo2 = sax.scoo(sax.sdict(scoo1))
    np.testing.assert_array_equal(scoo1[0], scoo2[0])  # Si
    np.testing.assert_array_equal(scoo1[1], scoo2[1])  # Sj
    np.testing.assert_array_almost_equal(scoo1[2], scoo2[2])  # Sx


def test_scoo_to_sdense_to_scoo() -> None:
    scoo1 = sax.scoo(SDICT_REF)
    sdense = sax.sdense(scoo1)
    scoo2 = sax.scoo(sdense)
    # Reconstruct dense from both and compare
    S1, _ = sax.sdense(scoo1)
    S2, _ = sax.sdense(scoo2)
    np.testing.assert_array_almost_equal(S1, S2)


def test_sdense_to_sdict_to_sdense() -> None:
    sdense1 = sax.sdense(SDICT_REF)
    sdense2 = sax.sdense(sax.sdict(sdense1))
    np.testing.assert_array_almost_equal(sdense1[0], sdense2[0])


def test_sdense_to_scoo_to_sdense() -> None:
    sdense1 = sax.sdense(SDICT_REF)
    sdense2 = sax.sdense(sax.scoo(sdense1))
    np.testing.assert_array_almost_equal(sdense1[0], sdense2[0])


# --- Full chain: sdict -> scoo -> sdense -> sdict ---


def test_full_chain_sdict_scoo_sdense_sdict() -> None:
    scoo = sax.scoo(SDICT_REF)
    sdense = sax.sdense(scoo)
    result = sax.sdict(sdense)
    for k, v in SDICT_REF.items():
        np.testing.assert_allclose(float(result[k].real), float(complex(v).real))


# --- SCoo convention ---


def test_scoo_convention() -> None:
    """SCoo Si = row = output port index, Sj = col = input port index."""
    Si, Sj, _Sx, pm = sax.scoo({("in", "out"): jnp.array(1.0)})
    # The single entry should have Si = out index (row), Sj = in index (col)
    assert int(Si[0]) == pm["out"]
    assert int(Sj[0]) == pm["in"]
