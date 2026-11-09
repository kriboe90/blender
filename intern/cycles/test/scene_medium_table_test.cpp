/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>

#include "scene/medium_table.h"

#include "kernel/closure/volume_util.h"

#include "util/math.h"

CCL_NAMESPACE_BEGIN

namespace {

/* Henyey-Greenstein phase function, normalized over the sphere. */
double hg(const double cos_theta, const double g)
{
  const double denom = 1.0 + g * g - 2.0 * g * cos_theta;
  return (1.0 - g * g) / (4.0 * M_PI * denom * std::sqrt(denom));
}

/* Medium table text with one phase function row per entry of `g`, sampled every 0.1 degree. */
string make_table_text(const string &rows_header,
                       const vector<double> &g,
                       const vector<double> &sigma_s,
                       const vector<double> &sigma_a,
                       const string &extra = "")
{
  std::ostringstream text;
  text.precision(9);
  text << "CYCLES_MEDIUM_TABLE 1\n# Test table\n" << rows_header << "\n";
  const int num_angles = 1801;
  text << "ANGLES_DEG " << num_angles << "\n";
  for (int i = 0; i < num_angles; i++) {
    text << i * 0.1 << " ";
  }
  text << "\nSIGMA_S\n";
  for (const double s : sigma_s) {
    text << s << " ";
  }
  text << "\nSIGMA_A\n";
  for (const double a : sigma_a) {
    text << a << " ";
  }
  text << "\nPHASE\n";
  for (const double row_g : g) {
    for (int i = 0; i < num_angles; i++) {
      text << hg(std::cos(i * 0.1 * M_PI / 180.0), row_g) << " ";
    }
    text << "\n";
  }
  text << extra;
  return text.str();
}

float3 eval_table(const MediumTable &table, const double theta, float *pdf)
{
  return phase_tabulated_eval(
      table.data.data(), table.num_angles, float(theta), float(std::sin(theta)), pdf);
}

}  // namespace

TEST(MediumTable, ParseErrors)
{
  MediumTable table;
  string error;

  EXPECT_FALSE(medium_table_build("", table, error));
  EXPECT_FALSE(error.empty());

  EXPECT_FALSE(medium_table_build("CYCLES_MEDIUM_TABLE 2", table, error));
  EXPECT_NE(error.find("version"), string::npos);

  EXPECT_FALSE(medium_table_build("CYCLES_MEDIUM_TABLE 1\nCHANNELS RGB\nFOO", table, error));
  EXPECT_NE(error.find("FOO"), string::npos);

  /* Too few values for PHASE. */
  EXPECT_FALSE(medium_table_build(
      "CYCLES_MEDIUM_TABLE 1\nCHANNELS RGB\nANGLES_DEG 2\n0 180\nSIGMA_S 1 1 1\nSIGMA_A 0 0 0\n"
      "PHASE 1 1 1 1 1",
      table,
      error));
  EXPECT_NE(error.find("PHASE"), string::npos);

  /* Angles must cover the sphere. */
  EXPECT_FALSE(medium_table_build(
      "CYCLES_MEDIUM_TABLE 1\nCHANNELS RGB\nANGLES_DEG 2\n0 90\nSIGMA_S 1 1 1\nSIGMA_A 0 0 0\n"
      "PHASE 1 1 1 1 1 1",
      table,
      error));

  /* Missing SIGMA_A. */
  EXPECT_FALSE(medium_table_build(
      "CYCLES_MEDIUM_TABLE 1\nCHANNELS RGB\nANGLES_DEG 2\n0 180\nSIGMA_S 1 1 1\n"
      "PHASE 1 1 1 1 1 1",
      table,
      error));

  /* Negative values. */
  EXPECT_FALSE(medium_table_build(
      "CYCLES_MEDIUM_TABLE 1\nCHANNELS RGB\nANGLES_DEG 2\n0 180\nSIGMA_S 1 -1 1\nSIGMA_A 0 0 0\n"
      "PHASE 1 1 1 1 1 1",
      table,
      error));
}

TEST(MediumTable, IsotropicRGB)
{
  MediumTable table;
  string error;
  ASSERT_TRUE(
      medium_table_build("CYCLES_MEDIUM_TABLE 1\n"
                         "CHANNELS RGB  # comment\n"
                         "ANGLES_DEG 2\n0 180\n"
                         "SIGMA_S 1 2 3\nSIGMA_A 0.5 0.25 0.125\n"
                         "PHASE\n1 1\n2 2\n7 7\n",
                         table,
                         error))
      << error;

  EXPECT_EQ(table.num_angles, MEDIUM_TABLE_NUM_ANGLES);
  EXPECT_EQ(table.data.size(), size_t(PHASE_TABULATED_STRIDE * MEDIUM_TABLE_NUM_ANGLES));
  EXPECT_FLOAT_EQ(table.sigma_s.x, 1.0f);
  EXPECT_FLOAT_EQ(table.sigma_s.y, 2.0f);
  EXPECT_FLOAT_EQ(table.sigma_s.z, 3.0f);
  EXPECT_FLOAT_EQ(table.sigma_a.x, 0.5f);
  EXPECT_FLOAT_EQ(table.sigma_a.z, 0.125f);
  EXPECT_NEAR(table.g, 0.0f, 1e-5f);

  /* Every channel is normalized to 1 / (4 pi), including at the poles. */
  for (const double theta : {0.0, 1e-4, 0.3, M_PI_2, 2.5, M_PI - 1e-4, M_PI}) {
    float pdf;
    const float3 p = eval_table(table, theta, &pdf);
    EXPECT_NEAR(p.x, M_1_4PI_F, 1e-4f) << theta;
    EXPECT_NEAR(p.y, M_1_4PI_F, 1e-4f) << theta;
    EXPECT_NEAR(p.z, M_1_4PI_F, 1e-4f) << theta;
    EXPECT_NEAR(pdf, M_1_4PI_F, 1e-4f) << theta;
  }
}

TEST(MediumTable, HenyeyGreenstein)
{
  for (const double g : {-0.5, 0.0, 0.5, 0.7, 0.9}) {
    MediumTable table;
    string error;
    ASSERT_TRUE(medium_table_build(
        make_table_text("CHANNELS RGB", {g, g, g}, {1, 1, 1}, {0, 0, 0}), table, error))
        << error;

    EXPECT_NEAR(table.g, g, 2e-3) << g;

    /* The tabulated phase function matches the analytic one. */
    for (int i = 0; i <= 180; i++) {
      const double theta = (i + 0.37) * M_PI / 181.0;
      float pdf;
      const float3 p = eval_table(table, theta, &pdf);
      const double expected = hg(std::cos(theta), g);
      EXPECT_NEAR(p.x / expected, 1.0, 5e-3) << "g " << g << " theta " << theta;
      EXPECT_NEAR(pdf / expected, 1.0, 5e-3) << "g " << g << " theta " << theta;
    }
  }
}

TEST(MediumTable, SamplingMatchesPdf)
{
  /* Different phase function per channel, so the PDF is a mixture. */
  MediumTable table;
  string error;
  ASSERT_TRUE(medium_table_build(
      make_table_text("CHANNELS RGB", {0.9, 0.3, 0.0}, {1, 1, 1}, {0, 0, 0}), table, error))
      << error;

  /* Every channel integrates to one over the sphere. */
  const int num_steps = 200000;
  double integral[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < num_steps; i++) {
    const double theta = (i + 0.5) * M_PI / num_steps;
    float pdf;
    const float3 p = eval_table(table, theta, &pdf);
    const double solid_angle = 2.0 * M_PI * std::sin(theta) * M_PI / num_steps;
    integral[0] += p.x * solid_angle;
    integral[1] += p.y * solid_angle;
    integral[2] += p.z * solid_angle;
  }
  for (int c = 0; c < 3; c++) {
    EXPECT_NEAR(integral[c], 1.0, 1e-3) << c;
  }

  /* Histogram of sampled angles against the probability from the PDF. Stratified random numbers
   * make the histogram follow the inverse CDF closely. */
  const int num_bins = 90;
  const int num_samples = 900000;
  vector<double> histogram(num_bins, 0.0);
  for (int i = 0; i < num_samples; i++) {
    const float rand = (i + 0.5f) / num_samples;
    const float theta = phase_tabulated_sample_theta(table.data.data(), table.num_angles, rand);
    ASSERT_GE(theta, 0.0f);
    ASSERT_LE(theta, M_PI_F);
    const int bin = std::min(int(theta / M_PI * num_bins), num_bins - 1);
    histogram[bin] += 1.0 / num_samples;
  }

  for (int bin = 0; bin < num_bins; bin++) {
    double expected = 0.0;
    const int sub_steps = 2000;
    for (int s = 0; s < sub_steps; s++) {
      const double theta = (bin + (s + 0.5) / sub_steps) * M_PI / num_bins;
      float pdf;
      eval_table(table, theta, &pdf);
      expected += pdf * 2.0 * M_PI * std::sin(theta) * M_PI / (num_bins * sub_steps);
    }
    EXPECT_NEAR(histogram[bin], expected, 1e-4 + 1e-3 * expected) << "bin " << bin;
  }
}

TEST(MediumTable, SpectralBandAveraging)
{
  /* One wavelength per default band: B 400-500, G 500-600 and R 600-700 nm. */
  MediumTable table;
  string error;
  ASSERT_TRUE(medium_table_build(
      make_table_text(
          "WAVELENGTHS_NM 3\n450 550 650", {0.0, 0.5, 0.9}, {1.0, 2.0, 3.0}, {0.1, 0.2, 0.3}),
      table,
      error))
      << error;

  EXPECT_FLOAT_EQ(table.sigma_s.x, 3.0f);
  EXPECT_FLOAT_EQ(table.sigma_s.y, 2.0f);
  EXPECT_FLOAT_EQ(table.sigma_s.z, 1.0f);
  EXPECT_FLOAT_EQ(table.sigma_a.x, 0.3f);
  EXPECT_FLOAT_EQ(table.sigma_a.z, 0.1f);

  float pdf;
  const float3 p = eval_table(table, 0.2, &pdf);
  EXPECT_NEAR(p.x / hg(std::cos(0.2), 0.9), 1.0, 5e-3);
  EXPECT_NEAR(p.y / hg(std::cos(0.2), 0.5), 1.0, 5e-3);
  EXPECT_NEAR(p.z / hg(std::cos(0.2), 0.0), 1.0, 5e-3);

  /* Custom channel weights mix wavelengths, weighting the phase functions by sigma_s. All three
   * wavelengths in the red channel, with trapezoidal wavelength weights 50, 100 and 50. */
  ASSERT_TRUE(medium_table_build(make_table_text("WAVELENGTHS_NM 3\n450 550 650",
                                                 {0.0, 0.5, 0.9},
                                                 {1.0, 2.0, 3.0},
                                                 {0.1, 0.2, 0.3},
                                                 "CHANNEL_WEIGHTS\n1 1 1\n0 1 0\n1 0 0\n"),
                                 table,
                                 error))
      << error;

  EXPECT_NEAR(table.sigma_s.x, (50.0 * 1.0 + 100.0 * 2.0 + 50.0 * 3.0) / 200.0, 1e-5);
  EXPECT_FLOAT_EQ(table.sigma_s.y, 2.0f);
  EXPECT_FLOAT_EQ(table.sigma_s.z, 1.0f);

  const float3 p_mix = eval_table(table, 0.2, &pdf);
  const double mix_expected = (50.0 * 1.0 * hg(std::cos(0.2), 0.0) +
                               100.0 * 2.0 * hg(std::cos(0.2), 0.5) +
                               50.0 * 3.0 * hg(std::cos(0.2), 0.9)) /
                              (50.0 * 1.0 + 100.0 * 2.0 + 50.0 * 3.0);
  EXPECT_NEAR(p_mix.x / mix_expected, 1.0, 5e-3);

  /* A band without wavelengths is an error. */
  EXPECT_FALSE(medium_table_build(
      make_table_text("WAVELENGTHS_NM 2\n450 550", {0.0, 0.5}, {1.0, 2.0}, {0.1, 0.2}),
      table,
      error));
  EXPECT_NE(error.find("channel R"), string::npos);
}

CCL_NAMESPACE_END
