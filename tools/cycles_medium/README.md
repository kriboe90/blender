# Tabulated Volume tools

Tools for the Cycles **Tabulated Volume** shader node. The node renders a volume whose
scattering coefficient, absorption coefficient and phase function come from a *medium table*,
for example computed with Mie theory. Each RGB channel gets its own phase function.

## Medium table format

The table is plain text, with values separated by whitespace and comments starting with `#`.
The full specification is in `intern/cycles/scene/medium_table.h`.

```
CYCLES_MEDIUM_TABLE 1
WAVELENGTHS_NM 31          # or: CHANNELS RGB (three rows, in the order R, G, B)
400 410 ... 700
ANGLES_DEG 1801            # any spacing, increasing, must cover 0..180
0 0.1 ... 180
SIGMA_S                    # per unit Density, per scene length unit (1/m)
...
SIGMA_A
...
PHASE                      # one row per wavelength, any normalization
...
CHANNEL_WEIGHTS            # optional: R, G and B weight per wavelength
...
```

**What to take from a Mie calculation.** For each wavelength, average over the particle size
distribution and provide:
- `SIGMA_S` and `SIGMA_A`: the scattering and absorption coefficients per unit concentration,
  for example mass-specific cross sections in m²/kg when Density is the soot density in kg/m³;
- `PHASE`: the unpolarized phase function `(|S1|² + |S2|²) / 2`, with polydisperse averaging
  weighted by the scattering cross section.

**Band averaging to RGB.** With wavelengths, Cycles averages to RGB using `CHANNEL_WEIGHTS`. The
weights are the product of the camera response and the light spectrum, for example the
emission spectrum of the LED that dominates each channel. Without weights, it uses flat bands:
B 400–500, G 500–600, R 600–700 nm.
- The coefficients are averaged with these weights.
- The phase functions are additionally weighted by `SIGMA_S`.

## Writing tables

`medium_table.py` needs only the Python standard library. NumPy arrays work as input.

```python
import medium_table
medium_table.write("soot.txt", angles_deg=theta, phase=p, sigma_s=k_sca, sigma_a=k_abs,
                   wavelengths_nm=wl, channel_weights=[led_r, led_g, led_b])
```

To write a Henyey–Greenstein table with a different anisotropy per channel, for testing:

```bash
python3 tools/cycles_medium/medium_table.py hg_test.txt --g 0.9 0.3 0.0
```

## Mie example

`mie_example.py` computes a table with Mie theory for spheres with a log-normal size distribution.
It requires NumPy and `miepython`. The coefficients are mass-specific (m²/kg), so the node's
Density input is the particle mass concentration in kg/m³, for example FDS `SOOT DENSITY`. The
`--led` option builds the channel weights from Gaussian LED spectra.

```bash
python3 tools/cycles_medium/mie_example.py soot.txt --m 1.75 0.6 --dg 0.15 --sigma-g 1.6 \
    --particle-density 1800 --led 630 525 465 --led-fwhm 25
```

Soot is made of fractal aggregates, for which equal-volume spheres are only a rough model. Any
other method, such as RDG-FA or T-matrix, can write the same table with `medium_table.write()`.

## Using the node in Blender

The table is read from a text data-block, which is saved in the `.blend` file.

```python
node = material.node_tree.nodes.new("ShaderNodeVolumeTabulated")
node.table = bpy.data.texts.load("/path/to/soot.txt")
material.node_tree.links.new(attribute_node.outputs["Fac"], node.inputs["Density"])
material.node_tree.links.new(node.outputs["Volume"], output_node.inputs["Volume"])
```

The resulting coefficients are the table values × Density × the object's volume density.

**Engine support:**
- **Cycles with SVM:** full support on CPU, Metal, CUDA, OptiX, HIP and oneAPI.
- **Cycles with OSL:** approximates the phase function with Henyey–Greenstein, using the
  table's mean cosine.
- **EEVEE:** not supported; the volume is not shown.

## Render validation

`validate_render.py` renders analytic Henyey–Greenstein closures and the same phase functions as
tables. It checks that the images match within the Monte Carlo noise. This covers:
- a single phase function;
- one phase function per channel;
- colored scattering and absorption.

```bash
blender -b --factory-startup --python tools/cycles_medium/validate_render.py -- \
    --device OPTIX --samples 256 --output /tmp/medium_validation
```

`--device` accepts `CPU`, `METAL`, `CUDA`, `OPTIX`, `HIP` or `ONEAPI`. The script exits with
code 1 when a comparison fails.
