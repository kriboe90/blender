# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Example: compute a medium table with Mie theory for spheres with a log-normal size distribution.

Requires NumPy and miepython (``pip install miepython``). For each wavelength, the coefficients are
averaged over the size distribution and divided by the mean particle mass, so they are mass-specific
(m^2/kg). The Density input of the Tabulated Volume node is then the particle mass concentration in
kg/m^3, for example the FDS output quantity SOOT DENSITY. The phase function is averaged weighted
by the scattering cross section.

Soot-like spheres, with channel weights from narrowband LEDs::

    python3 mie_example.py soot.txt --m 1.75 0.6 --dg 0.15 --sigma-g 1.6 --particle-density 1800 \\
        --led 630 525 465 --led-fwhm 25

Water droplets::

    python3 mie_example.py fog.txt --m 1.33 0.0 --dg 10 --sigma-g 1.2 --particle-density 1000

Note that soot consists of fractal aggregates, for which equal-volume spheres are only a rough
model. Any other method that provides the same quantities per wavelength can write the table with
`medium_table.write()` instead.
"""

import argparse
import math
import os
import sys

import numpy as np

try:
    import miepython
except ImportError:
    sys.exit("This example requires miepython: pip install miepython")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import medium_table  # noqa: E402


def scattering_angles():
    """Scattering angles in degrees, finer near the forward peak."""
    return np.unique(np.concatenate([
        np.arange(0.0, 10.0, 0.02),
        np.arange(10.0, 180.0, 0.2),
        [180.0],
    ]))


def compute_medium(m, d_g_um, sigma_g, particle_density, wavelengths_nm, num_sizes=60):
    """
    Mie scattering for a log-normal number distribution of sphere diameters.

    Returns angles in degrees, mass-specific scattering and absorption coefficients in m^2/kg per
    wavelength, the phase function per wavelength (normalized over the sphere) and the asymmetry
    parameter per wavelength.
    """
    # Diameters equally spaced in log(d), covering +-3.5 geometric standard deviations.
    ln_sigma = math.log(sigma_g)
    ln_d = np.linspace(math.log(d_g_um) - 3.5 * ln_sigma, math.log(d_g_um) + 3.5 * ln_sigma, num_sizes)
    d_um = np.exp(ln_d)
    number = np.exp(-0.5 * ((ln_d - math.log(d_g_um)) / ln_sigma) ** 2)
    number /= number.sum()

    d_m = d_um * 1e-6
    area = math.pi * d_m ** 2 / 4.0
    mean_mass = np.sum(number * particle_density * math.pi * d_m ** 3 / 6.0)

    angles = scattering_angles()
    mu = np.cos(np.radians(angles))

    sigma_s = []
    sigma_a = []
    phase = []
    g = []
    for lambda_nm in wavelengths_nm:
        lambda_um = lambda_nm * 1e-3
        qext, qsca, _qback, g_d = miepython.efficiencies(m, d_um, lambda_um)
        c_sca = number * qsca * area
        c_abs = number * (qext - qsca) * area
        sigma_s.append(c_sca.sum() / mean_mass)
        sigma_a.append(c_abs.sum() / mean_mass)

        p = np.zeros_like(mu)
        for weight, diameter in zip(c_sca, d_um):
            x = math.pi * diameter / lambda_um
            p += weight * miepython.i_unpolarized(m, x, mu, norm="one")
        phase.append(p / c_sca.sum())
        g.append(np.sum(c_sca * g_d) / c_sca.sum())

    return angles, np.array(sigma_s), np.array(sigma_a), np.array(phase), np.array(g)


def gaussian_weights(wavelengths_nm, centers_nm, fwhm_nm):
    """Channel weights for narrowband LEDs with Gaussian spectra, in the order R, G, B."""
    s = fwhm_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return [np.exp(-0.5 * ((np.asarray(wavelengths_nm) - c) / s) ** 2) for c in centers_nm]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="Medium table to write")
    parser.add_argument("--m", type=float, nargs=2, default=(1.75, 0.6), metavar=("N", "K"),
                        help="Complex refractive index n + ik of the particles")
    parser.add_argument("--dg", type=float, default=0.15, help="Geometric mean diameter in micrometers")
    parser.add_argument("--sigma-g", type=float, default=1.6, help="Geometric standard deviation")
    parser.add_argument("--particle-density", type=float, default=1800.0, help="Particle material density in kg/m^3")
    parser.add_argument("--wavelengths", type=float, nargs=3, default=(400.0, 700.0, 5.0),
                        metavar=("MIN", "MAX", "STEP"), help="Wavelength grid in nm")
    parser.add_argument("--led", type=float, nargs=3, metavar=("R", "G", "B"),
                        help="Center wavelengths in nm of the LEDs that dominate the R, G and B channels")
    parser.add_argument("--led-fwhm", type=float, default=25.0, help="LED spectral width (FWHM) in nm")
    args = parser.parse_args()

    lo, hi, step = args.wavelengths
    wavelengths = np.arange(lo, hi + 0.5 * step, step)
    m = complex(args.m[0], args.m[1])

    angles, sigma_s, sigma_a, phase, g = compute_medium(
        m, args.dg, args.sigma_g, args.particle_density, wavelengths)

    weights = gaussian_weights(wavelengths, args.led, args.led_fwhm) if args.led else None
    medium_table.write(
        args.output,
        angles_deg=angles,
        phase=phase,
        sigma_s=sigma_s,
        sigma_a=sigma_a,
        wavelengths_nm=wavelengths,
        channel_weights=weights,
        comment=(
            "Mie spheres, m = {:g} + {:g}i, log-normal d_g = {:g} um, sigma_g = {:g}\n"
            "Coefficients in m^2/kg for particle density {:g} kg/m^3"
        ).format(args.m[0], args.m[1], args.dg, args.sigma_g, args.particle_density),
    )

    print("Wrote", args.output)
    print("lambda [nm]  sigma_s [m2/kg]  sigma_a [m2/kg]  albedo   g")
    for i in range(0, len(wavelengths), max(1, len(wavelengths) // 6)):
        k_ext = sigma_s[i] + sigma_a[i]
        print("{:8.0f}  {:15.1f}  {:15.1f}  {:7.3f}  {:6.3f}".format(
            wavelengths[i], sigma_s[i], sigma_a[i], sigma_s[i] / k_ext, g[i]))


if __name__ == "__main__":
    main()
