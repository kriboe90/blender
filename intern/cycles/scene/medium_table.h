/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/string.h"
#include "util/types.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

/* Tabulated participating medium for the Tabulated Volume node, e.g. computed with Mie theory.
 *
 * The text format describes one particle population, either per wavelength, which is then
 * band-averaged to RGB, or directly per RGB channel. Values are separated by whitespace, and
 * comments run from '#' to the end of the line:
 *
 *   CYCLES_MEDIUM_TABLE 1
 *   WAVELENGTHS_NM <n>       (or "CHANNELS RGB", which means n = 3 rows in the order R, G, B)
 *   <n wavelengths in nanometers, increasing>
 *   ANGLES_DEG <m>
 *   <m scattering angles in degrees, increasing, covering 0 to 180>
 *   SIGMA_S
 *   <n scattering coefficients per unit density, per scene unit length>
 *   SIGMA_A
 *   <n absorption coefficients per unit density, per scene unit length>
 *   PHASE
 *   <n rows of m phase function values, in any normalization>
 *   CHANNEL_WEIGHTS          (optional, only with wavelengths)
 *   <3 rows of n weights for the R, G and B channels>
 *
 * The sections after WAVELENGTHS_NM or CHANNELS can be in any order.
 *
 * Without CHANNEL_WEIGHTS, the channels are flat bands: B 400-500 nm, G 500-600 nm and
 * R 600-700 nm. The weights are the product of the camera response and the spectrum of the light,
 * e.g. the emission spectrum of an LED that dominates a channel.
 *
 * Band averaging per channel c with weights w_c(lambda):
 *   sigma_c = integral(sigma(lambda) * w_c) / integral(w_c)
 *   p_c(theta) = integral(sigma_s(lambda) * p(theta, lambda) * w_c) /
 *                integral(sigma_s(lambda) * w_c)
 */

struct MediumTable {
  /* RGB scattering and absorption coefficients per unit density. */
  float3 sigma_s = zero_float3();
  float3 sigma_a = zero_float3();
  /* Mean cosine of the phase function, averaged over the channels. */
  float g = 0.0f;
  /* Number of angle samples, and the packed table as described for phase_tabulated_eval(). */
  int num_angles = 0;
  vector<float> data;
};

/* Number of angle samples of the packed table, about 0.09 degree apart. */
#define MEDIUM_TABLE_NUM_ANGLES 2048

/* Parse a medium table from text and build the packed table for the kernel. On failure, returns
 * false and describes the problem in `error`. */
bool medium_table_build(const string &text,
                        MediumTable &table,
                        string &error,
                        const int num_angles = MEDIUM_TABLE_NUM_ANGLES);

CCL_NAMESPACE_END
