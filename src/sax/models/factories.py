"""SAX Default Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@validate_call
def model_2port(p1: sax.Name, p2: sax.Name) -> sax.SDictModel:
    """Generate a general 2-port model with 100% transmission.

    ```{svgbob}
     p1 *--------* p2
    ```

    Args:
        p1: Name of the first port (typically in0 or o1).
        p2: Name of the second port (typically out0 or o2).

    Returns:
        A 2-port model
    """

    @jax.jit
    @validate_call
    def model_2port(wl: sax.FloatArrayLike = 1.5) -> sax.SDict:
        wl = jnp.asarray(wl)
        return sax.reciprocal({(p1, p2): jnp.ones_like(wl)})

    return model_2port


@validate_call
def model_3port(p1: sax.Name, p2: sax.Name, p3: sax.Name) -> sax.SDictModel:
    """Generate a general 3-port model (1x2 splitter).

    ```{svgbob}
            +-----+--* p2
     p1 *---|     |
            +-----+--* p3
    ```

    Args:
        p1: Name of the input port (typically o1 or in0).
        p2: Name of the first output port (typically o2 or out1).
        p3: Name of the second output port (typically o3 or out0).

    Returns:
        A 3-port model
    """

    @jax.jit
    @validate_call
    def model_3port(wl: sax.FloatArrayLike = 1.5) -> sax.SDict:
        wl = jnp.asarray(wl)
        thru = jnp.ones_like(wl) / jnp.sqrt(2)
        return sax.reciprocal(
            {
                (p1, p2): thru,
                (p1, p3): thru,
            }
        )

    return model_3port


@validate_call
def model_4port(
    p1: sax.Name, p2: sax.Name, p3: sax.Name, p4: sax.Name
) -> sax.SDictModel:
    """Generate a general 4-port model (2x2 coupler).

    ```{svgbob}
    p2 *---+-----+--* p3
           |     |
    p1 *---+-----+--* p4
    ```

    Args:
        p1: Name of the first input port (typically o1 or in0).
        p2: Name of the second input port (typically o2 or in1).
        p3: Name of the first output port (typically o3 or out1).
        p4: Name of the second output port (typicall o4 or out0).

    Returns:
        A 4-port model
    """

    @jax.jit
    @validate_call
    def model_4port(wl: sax.FloatArrayLike = 1.5) -> sax.SDict:
        wl = jnp.asarray(wl)
        thru = jnp.ones_like(wl) / jnp.sqrt(2)
        cross = 1j * thru
        return sax.reciprocal(
            {
                (p1, p4): thru,
                (p2, p3): thru,
                (p1, p3): cross,
                (p2, p4): cross,
            }
        )

    return model_4port


@validate_call
def unitary(
    num_inputs: int,
    num_outputs: int,
    *,
    reciprocal: bool = True,
    diagonal: bool = False,
) -> sax.SCooModel:
    """Generate a unitary N×M optical device model.

    ```{svgbob}
    o_ni -1 *---+-----+--* o_ni
                |     |
                  .
                  .
                  .
                |     |
         o2 *---+-----+--* o_N -2
        in1     |     |
                |     |
         o1 *---+-----+--* o_N -1
        in0
    ```

    Args:
        num_inputs: Number of input ports for the device.
        num_outputs: Number of output ports for the device.
        reciprocal: If True, the device exhibits reciprocal behavior (S = S^T).
            This is typical for passive optical devices. Defaults to True.
        diagonal: If True, creates a diagonal coupling matrix (each input
            couples to only one output). If False, creates full coupling
            between all input-output pairs. Defaults to False.

    Returns:
        A unitary model
    """
    # let's create the squared S-matrix:
    N = max(num_inputs, num_outputs)
    S = jnp.zeros((2 * N, 2 * N), dtype=float)

    if not diagonal:
        S = S.at[:N, N:].set(1)
    else:
        r = jnp.arange(
            N,
            dtype=int,
        )  # reciprocal only works if num_inputs == num_outputs!
        S = S.at[r, N + r].set(1)

    if reciprocal:
        if not diagonal:
            S = S.at[N:, :N].set(1)
        else:
            r = jnp.arange(
                N,
                dtype=int,
            )  # reciprocal only works if num_inputs == num_outputs!
            S = S.at[N + r, r].set(1)

    # Now we need to normalize the squared S-matrix
    U, s, V = jnp.linalg.svd(S, full_matrices=False)
    S = jnp.sqrt(U @ jnp.diag(jnp.where(s > sax.EPS, 1, 0)) @ V)

    # Now create subset of this matrix we're interested in:
    r = jnp.concatenate(
        [jnp.arange(num_inputs, dtype=int), N + jnp.arange(num_outputs, dtype=int)],
        0,
    )
    S = S[r, :][:, r]
    S = S.T  # convert from (from, to) to standard (to, from) convention

    # let's convert it in SCOO format:
    Si, Sj = jnp.where(S > sax.EPS)
    Sx = S[Si, Sj]

    # the last missing piece is a port map:
    p = sax.PortNamer(num_inputs, num_outputs)
    pm = {
        **{p[i]: i for i in range(num_inputs)},
        **{p[i + num_inputs]: i + num_inputs for i in range(num_outputs)},
    }

    @validate_call
    def func(*, wl: sax.FloatArrayLike = 1.5) -> sax.SCoo:
        wl_ = jnp.asarray(wl)
        Sx_ = jnp.broadcast_to(Sx, (*wl_.shape, *Sx.shape))
        return Si, Sj, Sx_, pm

    func.__name__ = f"unitary_{num_inputs}_{num_outputs}"
    func.__qualname__ = f"unitary_{num_inputs}_{num_outputs}"
    return jax.jit(func)


@validate_call
def copier(
    num_inputs: int,
    num_outputs: int,
    *,
    reciprocal: bool = True,
    diagonal: bool = False,
) -> sax.SCooModel:
    """Copy 100% of the power at the input ports to all output ports.

    ```{svgbob}
    o_ni -1 *---+-----+--* o_ni
                |     |
                  .
                  .
                  .
                |     |
         o2 *---+-----+--* o_N -2
        in1     |     |
                |     |
         o1 *---+-----+--* o_N -1
        in0
    ```

    Args:
        num_inputs: Number of input ports for the device.
        num_outputs: Number of output ports for the device.
        reciprocal: If True, the device exhibits reciprocal behavior where
            forward and reverse transmissions are equal. Defaults to True.
        diagonal: If True, creates diagonal coupling (each input couples to
            only one output). If False, creates full coupling between all
            input-output pairs. Defaults to False.

    Returns:
        A copier model

    Examples:
        A 1×4 optical amplifier/splitter:

        ```python
        import sax

        amp_splitter = sax.models.copier(1, 4, reciprocal=False)
        Si, Sj, Sx, port_map = amp_splitter(wl=1.55)
        # Single input amplified and split to 4 outputs
        ```

    """
    # let's create the squared S-matrix:
    S = jnp.zeros((num_inputs + num_outputs, num_inputs + num_outputs), dtype=float)

    if not diagonal:
        S = S.at[:num_inputs, num_inputs:].set(1)
    else:
        r = jnp.arange(
            num_inputs,
            dtype=int,
        )  # == range(num_outputs) # reciprocal only works if num_inputs == num_outputs!
        S = S.at[r, num_inputs + r].set(1)

    if reciprocal:
        if not diagonal:
            S = S.at[num_inputs:, :num_inputs].set(1)
        else:
            # reciprocal only works if num_inputs == num_outputs!
            r = jnp.arange(num_inputs, dtype=int)  # == range(num_outputs)
            S = S.at[num_inputs + r, r].set(1)

    S = S.T  # convert from (from, to) to standard (to, from) convention

    # let's convert it in SCOO format:
    Si, Sj = jnp.where(jnp.sqrt(sax.EPS) < S)
    Sx = S[Si, Sj]

    # the last missing piece is a port map:
    p = sax.PortNamer(num_inputs, num_outputs)
    pm = {
        **{p[i]: i for i in range(num_inputs)},
        **{p[i + num_inputs]: i + num_inputs for i in range(num_outputs)},
    }

    @validate_call
    def func(wl: sax.FloatArrayLike = 1.5) -> sax.SCoo:
        wl_ = jnp.asarray(wl)
        Sx_ = jnp.broadcast_to(Sx, (*wl_.shape, *Sx.shape))
        return Si, Sj, Sx_, pm

    func.__name__ = f"unitary_{num_inputs}_{num_outputs}"
    func.__qualname__ = f"unitary_{num_inputs}_{num_outputs}"
    return jax.jit(func)


@validate_call
def passthru(
    num_links: int,
    *,
    reciprocal: bool = True,
) -> sax.SCooModel:
    """Copy 100% of the power at each input port to its corresponding output port.

    ```{svgbob}
    o_ni -1 *---+-----+--* o_ni
                |     |
                  .
                  .
                  .
                |     |
         o2 *---+-----+--* o_N -2
        in1     |     |
                |     |
         o1 *---+-----+--* o_N -1
        in0
    ```

    Args:
        num_links: Number of independent pass-through links (input-output pairs).
            This creates a device with num_links inputs and num_links outputs.
        reciprocal: If True, the device exhibits reciprocal behavior where
            transmission is identical in both directions. This is typical for
            passive devices like fibers and waveguides. Defaults to True.

    Returns:
        A passthru model

    Examples:
        Create an 8×8 optical switch (straight-through state):

        ```python
        switch_thru = sax.models.passthru(8, reciprocal=True)
        Si, Sj, Sx, port_map = switch_thru(wl=1.55)
        # Each input passes straight to corresponding output
        ```
    """
    passthru = unitary(
        num_links,
        num_links,
        reciprocal=reciprocal,
        diagonal=True,
    )
    passthru.__name__ = f"passthru_{num_links}_{num_links}"
    passthru.__qualname__ = f"passthru_{num_links}_{num_links}"
    return jax.jit(passthru)
