# Component models and physical conventions

## Callable contract

A model accepts keyword settings and returns `SDict`, `SCoo`, or `SDense` using the
[input/output convention](s-parameters.md). Circuit analysis calls it with no
arguments: every simulation parameter needs a usable default. Model validation
rejects required parameters, positional-only parameters, `*args`, and `**kwargs`;
ordinary defaulted positional-or-keyword parameters are allowed. Generated circuit
functions expose synthetic signatures for inspection.

Factories create models rather than S-matrices and are distinguished by callable
return annotations. Instantiate factories before using their returned model in a
circuit. Do not change the model's topology as a function of traced numerical values.
Use JAX operations on the numerical path when JIT or gradients are required.

Evidence: [`saxtypes/singlemode.py`](../src/sax/saxtypes/singlemode.py)
(`val_sax_callable`, `val_model`, `val_model_factory`),
[`utils.py`](../src/sax/utils.py) (`get_settings`, `replace_kwargs`),
[`models/factories.py`](../src/sax/models/factories.py).
Tests: [`00_typing.ipynb`](../src/tests/nbs/00_typing.ipynb).

## Optical models

Wavelengths and geometric lengths generally use **micrometers**. Loss conventions
are model-specific; do not interchange `loss`, `loss_dB_cm`, and insertion-loss
parameters. Complex entries are amplitude coefficients; power ratios use `abs(S)**2`
where the model's normalization permits it.

Representative contracts:

- `straight`: linearized effective index
  `n(wl) = neff - (wl-wl0)*(ng-neff)/wl0`; reciprocal transmission is
  `10**(-1e-4*loss_dB_cm*length/20) * exp(1j*2*pi*n(wl)*length/wl)`.
- `attenuator`: reciprocal amplitude `10**(-loss/20)`, shaped by wavelength.
- `coupler_ideal`: through amplitude `sqrt(1-coupling)` and cross amplitude
  `1j*sqrt(coupling)`, with reciprocal entries. The documented physical domain is
  `0 <= coupling <= 1`; the implementation is not a universal bounds checker.
- `coupler`: dispersive coupling angle with cosine through amplitude and negative
  sine quadrature cross amplitude. Its `coupling0` is not interchangeable with the
  ideal model's power-coupling parameter.
- `phase_shifter`: adds `pi*voltage` to propagation phase; `loss` is dB per
  micrometer, giving total loss `loss*length`. Historical numerical behavior is
  retained; `attenuator` is the separate lumped-loss model.
  `test_phase_shifter.py` verifies length scaling, voltage phase, and gradients.
- `ideal_probe` copies signals for measurement and is explicitly nonunitary.
- `isolator` and `circulator` provide directional/nonreciprocal behavior and use
  fixed `o*` names rather than the global `PortNamer` convention.

The model library also includes bends, crossings, grating couplers, ideal and
parameterized MMIs, splitters, reflectors/mirrors, terminators, and factories
`model_2port`, `model_3port`, `model_4port`, `unitary`, and `copier`.
Factory names do not establish physical validity for every parameter combination;
`copier` and probes intentionally need not conserve power.

Evidence: [`models/straight.py`](../src/sax/models/straight.py),
[`couplers.py`](../src/sax/models/couplers.py),
[`isolators.py`](../src/sax/models/isolators.py),
[`probes.py`](../src/sax/models/probes.py), and other modules in
[`models/`](../src/sax/models/). Public model inventory:
[`models/__init__.py`](../src/sax/models/__init__.py).
Optical regression coverage is mainly circuits/probes and notebooks; there is no
claim here that every primitive's physics was independently validated.

## RF models

[`models/rf.py`](../src/sax/models/rf.py) provides loads, tee, impedance/admittance,
resistor/capacitor/inductor, open/short, LC shunt, CPW, and microstrip models, plus
permittivity/impedance/thickness/propagation helpers.

- Frequency `f` is in **Hz**; resistance/reference impedance in **ohms**,
  capacitance in **farads**, inductance in **henries**.
- High-level `coplanar_waveguide` and `microstrip` geometry is in **micrometers**;
  they convert to meters internally. `transmission_line_s_params` uses length in
  **meters** and propagation constant in **1/m**.
- CPW/microstrip models use `o1`, `o2`, reshape results to the supplied frequency
  shape, and produce equal forward/reverse transmission and equal reflections.
- `ep_r`, `ep_eff`, and `tand` are real dielectric inputs. Complex inputs are
  rejected, including complex dtype with zero imaginary part. Use real `tand`
  to express dielectric loss rather than an imaginary permittivity.
- `propagation_constant` returns `alpha_d + 1j*beta`, with
  `beta = 2*pi*f*sqrt(ep_eff)/C_M_S`. Its vacuum-substrate special case suppresses
  dielectric attenuation when `ep_r` is effectively 1.
- `transmission_line_s_params(gamma, z0, length, z_ref=None)` returns `(S11, S21)`
  from ABCD conversion. Default `z_ref=z0` is matched: `S11=0`,
  `S21=exp(-gamma*length)`. High-level CPW/microstrip use this matched convention,
  not an implicit 50-ohm reference. Zero length gives zero reflection/unit transmission.

Optical straight propagation uses a positive phase exponential; matched RF lines
use `exp(-gamma*length)`. Do not silently unify these signs in a refactor.

Tests: [`test_rf_models.py`](../src/tests/test_rf_models.py),
[`test_rf_cpw.py`](../src/tests/test_rf_cpw.py),
[`test_rf_microstrip.py`](../src/tests/test_rf_microstrip.py), and
[`test_rf_dielectric_inputs.py`](../src/tests/test_rf_dielectric_inputs.py).
These cover formulas/limits, frequency shapes, selected JIT paths, dielectric
rejection, attenuation, matching, and selected passivity/reciprocity properties.

## Constants and helpers

[`constants.py`](../src/sax/constants.py) defines `C_M_S=299792458`,
`C_UM_S=1e6*C_M_S`, optical band limits/centers and wavelength-array helpers,
`DEFAULT_MODE="TE"`, and `DEFAULT_MODES=("TE", "TM")`. Keep unit conversions
explicit when moving between RF frequency and optical wavelength data.
