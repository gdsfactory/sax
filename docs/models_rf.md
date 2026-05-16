# Radio frequency (RF) Models

For more information on these RF models, see Ref. [@pozar-2012].

## Coplanar Waveguides (CPW) and Microstrips

Sax includes JAX-jittable functions for computing the characteristic impedance, effective permittivity, and propagation constant of coplanar waveguides and microstrip lines. All results are obtained analytically so the functions compose freely with JAX transformations (`jit`, `grad`, `vmap`, etc.).

### CPW Theory

The quasi-static CPW analysis follows the conformal-mapping approach described by Simons [@simonsCoplanarWaveguideCircuits2001] (ch. 2) and Ghione & Naldi [@ghioneAnalyticalFormulasCoplanar1984]. Conductor thickness corrections use the first-order formulae of Gupta, Garg, Bahl & Bhartia [@guptaMicrostripLinesSlotlines1996] (§7.3, Eqs. 7.98-7.100).

### Microstrip Theory

The microstrip analysis uses the Hammerstad-Jensen [@hammerstadAccurateModelsMicrostrip1980] closed-form expressions for effective permittivity and characteristic impedance, as presented in Pozar [@pozar-2012] (ch. 3, §3.8).

### General

The ABCD-to-S-parameter conversion is the standard microwave-network relation from Pozar [@pozar-2012] (ch. 4).

The implementation was cross-checked against the Qucs-S model (see Qucs technical documentation [@qucs_technical_papers], §12 for CPW, §11 for microstrip) and the `scikit-rf` CPW class.

::: sax.models.rf
