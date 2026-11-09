# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Write and read medium tables for the Cycles "Tabulated Volume" shader node.

A medium table describes one particle population: scattering and absorption coefficients and the
phase function, per wavelength or per RGB channel. See ``intern/cycles/scene/medium_table.h`` for
the format. Typical use with the output of a Mie calculation, per wavelength and averaged over the
particle size distribution::

    import medium_table
    medium_table.write(
        "soot.txt",
        angles_deg=theta_deg,            # 0 .. 180, any spacing
        phase=p,                         # [n_wavelengths][n_angles], any normalization
        sigma_s=k_sca,                   # per unit density, e.g. m^2/kg times kg/m^3 -> 1/m
        sigma_a=k_abs,
        wavelengths_nm=wavelengths,
        channel_weights=[led_r, led_g, led_b],  # optional, per wavelength
    )

In Blender, load the file into a text data-block and assign it to the node::

    node.table = bpy.data.texts.load("//soot.txt")

The module only needs the Python standard library; NumPy arrays work as inputs.
"""

__all__ = (
    "henyey_greenstein",
    "read",
    "write",
)

import math


def henyey_greenstein(theta_deg, g):
    """Henyey-Greenstein phase function, normalized over the sphere."""
    cos_theta = math.cos(math.radians(theta_deg))
    denom = 1.0 + g * g - 2.0 * g * cos_theta
    return (1.0 - g * g) / (4.0 * math.pi * denom * math.sqrt(denom))


def _format_row(values):
    return " ".join("{:.9g}".format(float(v)) for v in values)


def write(
        filepath,
        *,
        angles_deg,
        phase,
        sigma_s,
        sigma_a,
        wavelengths_nm=None,
        channel_weights=None,
        comment="",
):
    """
    Write a medium table.

    :arg angles_deg: Scattering angles in degrees, increasing, covering 0 to 180.
    :arg phase: One row of phase function values per wavelength (or per R, G, B channel), any
       normalization.
    :arg sigma_s: Scattering coefficient per wavelength (or channel), per unit density.
    :arg sigma_a: Absorption coefficient per wavelength (or channel), per unit density.
    :arg wavelengths_nm: Wavelengths in nanometers, increasing. When omitted, the rows are the R, G
       and B channels.
    :arg channel_weights: Optional R, G and B weights per wavelength, e.g. the emission spectrum
       of the LED that dominates each camera channel times the camera response. Without weights,
       Cycles uses flat bands: B 400-500, G 500-600, R 600-700 nm.
    :arg comment: Optional comment written at the top of the file.
    """
    angles_deg = [float(a) for a in angles_deg]
    phase = [[float(v) for v in row] for row in phase]
    sigma_s = [float(v) for v in sigma_s]
    sigma_a = [float(v) for v in sigma_a]

    num_rows = 3 if wavelengths_nm is None else len(wavelengths_nm)
    if len(phase) != num_rows or len(sigma_s) != num_rows or len(sigma_a) != num_rows:
        raise ValueError("phase, sigma_s and sigma_a need one entry per wavelength or channel")
    for row in phase:
        if len(row) != len(angles_deg):
            raise ValueError("every phase row needs one value per angle")
    if channel_weights is not None:
        if wavelengths_nm is None:
            raise ValueError("channel_weights requires wavelengths_nm")
        if len(channel_weights) != 3 or any(len(w) != num_rows for w in channel_weights):
            raise ValueError("channel_weights needs 3 rows (R, G, B) with one value per wavelength")

    lines = ["CYCLES_MEDIUM_TABLE 1"]
    for line in comment.splitlines():
        lines.append("# " + line)
    if wavelengths_nm is None:
        lines.append("CHANNELS RGB")
    else:
        lines.append("WAVELENGTHS_NM {:d}".format(num_rows))
        lines.append(_format_row(wavelengths_nm))
    lines.append("ANGLES_DEG {:d}".format(len(angles_deg)))
    lines.append(_format_row(angles_deg))
    lines.append("SIGMA_S")
    lines.append(_format_row(sigma_s))
    lines.append("SIGMA_A")
    lines.append(_format_row(sigma_a))
    lines.append("PHASE")
    lines.extend(_format_row(row) for row in phase)
    if channel_weights is not None:
        lines.append("CHANNEL_WEIGHTS")
        lines.extend(_format_row(row) for row in channel_weights)

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def read(filepath):
    """
    Read a medium table into a dictionary with the keys used by :func:`write`.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        tokens = []
        for line in fh:
            tokens.extend(line.split("#", 1)[0].split())

    if tokens[:2] != ["CYCLES_MEDIUM_TABLE", "1"]:
        raise ValueError("not a medium table (version 1)")

    result = {"wavelengths_nm": None, "channel_weights": None}
    pos = 2
    num_rows = 0

    def take(count):
        nonlocal pos
        values = [float(t) for t in tokens[pos:pos + count]]
        if len(values) != count:
            raise ValueError("unexpected end of table")
        pos += count
        return values

    while pos < len(tokens):
        keyword = tokens[pos]
        pos += 1
        if keyword == "CHANNELS":
            pos += 1
            num_rows = 3
        elif keyword == "WAVELENGTHS_NM":
            num_rows = int(tokens[pos])
            pos += 1
            result["wavelengths_nm"] = take(num_rows)
        elif keyword == "ANGLES_DEG":
            count = int(tokens[pos])
            pos += 1
            result["angles_deg"] = take(count)
        elif keyword == "SIGMA_S":
            result["sigma_s"] = take(num_rows)
        elif keyword == "SIGMA_A":
            result["sigma_a"] = take(num_rows)
        elif keyword == "PHASE":
            count = len(result["angles_deg"])
            result["phase"] = [take(count) for _ in range(num_rows)]
        elif keyword == "CHANNEL_WEIGHTS":
            result["channel_weights"] = [take(num_rows) for _ in range(3)]
        else:
            raise ValueError("unknown keyword " + keyword)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Write a Henyey-Greenstein medium table per RGB channel, e.g. for testing.",
    )
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--g", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("R", "G", "B"))
    parser.add_argument("--sigma-s", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("R", "G", "B"))
    parser.add_argument("--sigma-a", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("R", "G", "B"))
    parser.add_argument("--step", type=float, default=0.1, help="Angle step in degrees")
    args = parser.parse_args()

    num_angles = int(round(180.0 / args.step)) + 1
    angles = [180.0 * i / (num_angles - 1) for i in range(num_angles)]
    write(
        args.output,
        angles_deg=angles,
        phase=[[henyey_greenstein(a, g) for a in angles] for g in args.g],
        sigma_s=args.sigma_s,
        sigma_a=args.sigma_a,
        comment="Henyey-Greenstein, g = {:g} {:g} {:g}".format(*args.g),
    )


if __name__ == "__main__":
    main()
