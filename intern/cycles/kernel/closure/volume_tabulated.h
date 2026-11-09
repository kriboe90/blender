/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/globals.h"
#include "kernel/types.h"

#include "kernel/closure/volume_util.h"

#include "kernel/util/colorspace.h"

CCL_NAMESPACE_BEGIN

/* TABULATED CLOSURE
 *
 * Phase function from a table with one entry per color channel, see phase_tabulated_eval(). The
 * table is stored in the SVM node array, directly after the NODE_TABULATED_VOLUME node. */

struct TabulatedVolume {
  SHADER_CLOSURE_VOLUME_BASE;

  /* Offset of the table in the SVM node array. */
  int table_offset;
  int num_angles;
  /* Mean cosine of the average over the channels, for path guiding and ray differentials. */
  float g;
};
static_assert(sizeof(ShaderVolumeClosure) >= sizeof(TabulatedVolume),
              "TabulatedVolume is too large!");

ccl_device int volume_tabulated_setup(ccl_private TabulatedVolume *volume)
{
  volume->type = CLOSURE_VOLUME_TABULATED_ID;
  return SR_SCATTER;
}

ccl_device_inline const ccl_global float *volume_tabulated_table(
    KernelGlobals kg, const ccl_private TabulatedVolume *volume)
{
  return reinterpret_cast<const ccl_global float *>(
      &kernel_data_fetch(svm_nodes, volume->table_offset));
}

ccl_device Spectrum volume_tabulated_eval(KernelGlobals kg,
                                          const ccl_private ShaderData *sd,
                                          const ccl_private ShaderVolumeClosure *svc,
                                          const float3 wo,
                                          ccl_private float *pdf)
{
  const ccl_private TabulatedVolume *volume = (const ccl_private TabulatedVolume *)svc;

  /* Note that wi points towards the viewer. The angle is computed from both sine and cosine, to
   * stay precise for strongly forward scattering phase functions. */
  const float3 D = -sd->wi;
  const float cos_theta = dot(D, wo);
  const float sin_theta = len(cross(D, wo));
  const float theta = atan2f(sin_theta, cos_theta);

  return rgb_to_spectrum(phase_tabulated_eval(
      volume_tabulated_table(kg, volume), volume->num_angles, theta, sin_theta, pdf));
}

ccl_device int volume_tabulated_sample(KernelGlobals kg,
                                       const ccl_private ShaderData *sd,
                                       const ccl_private ShaderVolumeClosure *svc,
                                       const float2 rand,
                                       ccl_private Spectrum *eval,
                                       ccl_private float3 *wo,
                                       ccl_private float *pdf)
{
  const ccl_private TabulatedVolume *volume = (const ccl_private TabulatedVolume *)svc;
  const ccl_global float *table = volume_tabulated_table(kg, volume);

  const float theta = phase_tabulated_sample_theta(table, volume->num_angles, rand.x);
  const float sin_theta = sinf(theta);
  const float cos_theta = cosf(theta);

  /* Note that wi points towards the viewer and so is used negated. */
  *wo = phase_sample_direction(-sd->wi, cos_theta, rand.y);
  *eval = rgb_to_spectrum(phase_tabulated_eval(table, volume->num_angles, theta, sin_theta, pdf));

  return LABEL_VOLUME_SCATTER;
}

CCL_NAMESPACE_END
