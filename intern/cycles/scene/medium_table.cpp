/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/medium_table.h"

#include <cmath>
#include <locale>
#include <sstream>

#include "kernel/closure/volume_util.h"

#include "util/math.h"

CCL_NAMESPACE_BEGIN

namespace {

/* Parsed text, before resampling and band averaging. */
struct MediumTableText {
  bool is_rgb = false;
  vector<double> wavelengths;
  vector<double> angles;
  vector<double> sigma_s;
  vector<double> sigma_a;
  /* Phase function and channel weights, row-major. */
  vector<double> phase;
  vector<double> weights;
};

/* Whitespace separated tokens, with comments removed. Numbers are parsed independent of the
 * locale. */
class TokenStream {
 public:
  explicit TokenStream(const string &text)
  {
    std::istringstream lines(text);
    string line;
    while (std::getline(lines, line)) {
      const size_t comment = line.find('#');
      if (comment != string::npos) {
        line.resize(comment);
      }
      std::istringstream words(line);
      string word;
      while (words >> word) {
        tokens_.push_back(word);
      }
    }
  }

  bool at_end() const
  {
    return pos_ >= tokens_.size();
  }

  string next()
  {
    return at_end() ? string() : tokens_[pos_++];
  }

  bool read_number(double &value)
  {
    if (at_end()) {
      return false;
    }
    std::istringstream stream(tokens_[pos_]);
    stream.imbue(std::locale::classic());
    stream >> value;
    if (stream.fail() || !stream.eof() || !std::isfinite(value)) {
      return false;
    }
    pos_++;
    return true;
  }

  bool read_count(int &count)
  {
    double value;
    if (!read_number(value) || value < 1.0 || value > 1e7 || value != std::floor(value)) {
      return false;
    }
    count = int(value);
    return true;
  }

  bool read_numbers(const size_t count, vector<double> &values)
  {
    values.resize(count);
    for (size_t i = 0; i < count; i++) {
      if (!read_number(values[i])) {
        return false;
      }
    }
    return true;
  }

  /* Current token, for error messages. */
  string current() const
  {
    return at_end() ? string("end of text") : "'" + tokens_[pos_] + "'";
  }

 private:
  vector<string> tokens_;
  size_t pos_ = 0;
};

bool is_increasing(const vector<double> &values)
{
  for (size_t i = 1; i < values.size(); i++) {
    if (!(values[i] > values[i - 1])) {
      return false;
    }
  }
  return true;
}

bool is_non_negative(const vector<double> &values)
{
  for (const double value : values) {
    if (value < 0.0) {
      return false;
    }
  }
  return true;
}

bool parse_medium_table(const string &text, MediumTableText &medium, string &error)
{
  TokenStream tokens(text);

  int version = 0;
  if (tokens.next() != "CYCLES_MEDIUM_TABLE" || !tokens.read_count(version)) {
    error = "text must start with \"CYCLES_MEDIUM_TABLE 1\"";
    return false;
  }
  if (version != 1) {
    error = "unsupported version " + std::to_string(version);
    return false;
  }

  size_t num_rows = 0;
  bool has_sigma_s = false;
  bool has_sigma_a = false;
  bool has_phase = false;

  while (!tokens.at_end()) {
    const string keyword = tokens.next();

    if (keyword == "CHANNELS" || keyword == "WAVELENGTHS_NM") {
      if (num_rows != 0) {
        error = "only one CHANNELS or WAVELENGTHS_NM section is allowed";
        return false;
      }
      if (keyword == "CHANNELS") {
        if (tokens.next() != "RGB") {
          error = "CHANNELS must be followed by RGB";
          return false;
        }
        medium.is_rgb = true;
        num_rows = 3;
      }
      else {
        int count = 0;
        if (!tokens.read_count(count) || !tokens.read_numbers(count, medium.wavelengths)) {
          error = "WAVELENGTHS_NM must be followed by a count and that many wavelengths, at " +
                  tokens.current();
          return false;
        }
        if (!is_increasing(medium.wavelengths)) {
          error = "wavelengths must be increasing";
          return false;
        }
        num_rows = medium.wavelengths.size();
      }
    }
    else if (keyword == "ANGLES_DEG") {
      int count = 0;
      if (!tokens.read_count(count) || count < 2 || !tokens.read_numbers(count, medium.angles)) {
        error = "ANGLES_DEG must be followed by a count of at least 2 and that many angles, at " +
                tokens.current();
        return false;
      }
      if (!is_increasing(medium.angles)) {
        error = "angles must be increasing";
        return false;
      }
      if (medium.angles.front() < 0.0 || medium.angles.front() > 0.5 ||
          medium.angles.back() > 180.0 || medium.angles.back() < 179.5)
      {
        error = "angles must cover 0 to 180 degrees";
        return false;
      }
    }
    else if (keyword == "SIGMA_S" || keyword == "SIGMA_A" || keyword == "PHASE" ||
             keyword == "CHANNEL_WEIGHTS")
    {
      if (num_rows == 0) {
        error = keyword + " must come after CHANNELS or WAVELENGTHS_NM";
        return false;
      }
      vector<double> *values = nullptr;
      size_t count = num_rows;
      if (keyword == "SIGMA_S") {
        values = &medium.sigma_s;
        has_sigma_s = true;
      }
      else if (keyword == "SIGMA_A") {
        values = &medium.sigma_a;
        has_sigma_a = true;
      }
      else if (keyword == "PHASE") {
        if (medium.angles.empty()) {
          error = "PHASE must come after ANGLES_DEG";
          return false;
        }
        values = &medium.phase;
        count = num_rows * medium.angles.size();
        has_phase = true;
      }
      else {
        if (medium.is_rgb) {
          error = "CHANNEL_WEIGHTS can only be used with WAVELENGTHS_NM";
          return false;
        }
        values = &medium.weights;
        count = 3 * num_rows;
      }
      if (!tokens.read_numbers(count, *values)) {
        error = keyword + " expects " + std::to_string(count) + " numbers, at " + tokens.current();
        return false;
      }
      if (!is_non_negative(*values)) {
        error = keyword + " values must not be negative";
        return false;
      }
    }
    else {
      error = "unknown keyword '" + keyword + "'";
      return false;
    }
  }

  if (num_rows == 0 || medium.angles.empty() || !has_sigma_s || !has_sigma_a || !has_phase) {
    error =
        "missing section, CHANNELS or WAVELENGTHS_NM, ANGLES_DEG, SIGMA_S, SIGMA_A and PHASE "
        "are required";
    return false;
  }

  return true;
}

/* Resample a phase function row given at arbitrary angles to the uniform grid of the kernel
 * table, as density over theta normalized to integrate to one. Interpolates linearly in log space
 * where possible, which suits phase functions spanning several orders of magnitude. */
bool resample_phase(const vector<double> &angles_deg,
                    const double *phase,
                    const int num_angles,
                    vector<double> &density)
{
  const double dtheta = M_PI / (num_angles - 1);
  const size_t num_input = angles_deg.size();

  density.resize(num_angles);
  size_t k = 0;
  for (int i = 0; i < num_angles; i++) {
    const double theta_deg = i * dtheta * (180.0 / M_PI);
    while (k + 2 < num_input && angles_deg[k + 1] < theta_deg) {
      k++;
    }

    double p;
    if (theta_deg <= angles_deg.front()) {
      p = phase[0];
    }
    else if (theta_deg >= angles_deg.back()) {
      p = phase[num_input - 1];
    }
    else {
      const double t = (theta_deg - angles_deg[k]) / (angles_deg[k + 1] - angles_deg[k]);
      const double p0 = phase[k];
      const double p1 = phase[k + 1];
      p = (p0 > 0.0 && p1 > 0.0) ? std::exp((1.0 - t) * std::log(p0) + t * std::log(p1)) :
                                   (1.0 - t) * p0 + t * p1;
    }

    density[i] = p * std::sin(i * dtheta);
  }

  /* The density goes to zero at the poles. */
  density.front() = 0.0;
  density.back() = 0.0;

  double integral = 0.0;
  for (int i = 0; i + 1 < num_angles; i++) {
    integral += 0.5 * (density[i] + density[i + 1]) * dtheta;
  }
  if (!(integral > 0.0)) {
    return false;
  }
  for (double &value : density) {
    value /= integral;
  }
  return true;
}

/* Weights of the RGB channels per wavelength, including the integration weights of the
 * wavelength grid. */
bool channel_weights(const MediumTableText &medium, vector<double> &weights, string &error)
{
  const size_t num_wavelengths = medium.wavelengths.size();
  weights.assign(3 * num_wavelengths, 0.0);

  /* Default flat bands for R, G and B. */
  const double band_min[3] = {600.0, 500.0, 400.0};
  const double band_max[3] = {700.0, 600.0, 500.0};
  const char *channel_names[3] = {"R", "G", "B"};

  for (int c = 0; c < 3; c++) {
    double sum = 0.0;
    for (size_t j = 0; j < num_wavelengths; j++) {
      const double lambda = medium.wavelengths[j];
      double w;
      if (!medium.weights.empty()) {
        w = medium.weights[c * num_wavelengths + j];
      }
      else {
        w = (lambda >= band_min[c] && lambda <= band_max[c]) ? 1.0 : 0.0;
      }

      /* Trapezoidal integration weight of the wavelength grid. */
      double dlambda = 1.0;
      if (num_wavelengths > 1) {
        const double lambda_prev = medium.wavelengths[(j > 0) ? j - 1 : j];
        const double lambda_next = medium.wavelengths[(j + 1 < num_wavelengths) ? j + 1 : j];
        dlambda = 0.5 * (lambda_next - lambda_prev);
      }

      weights[c * num_wavelengths + j] = w * dlambda;
      sum += w * dlambda;
    }

    if (!(sum > 0.0)) {
      error = string("channel ") + channel_names[c] + " has no wavelength with a nonzero weight";
      if (medium.weights.empty()) {
        error += string(", the default band is ") + std::to_string(int(band_min[c])) + "-" +
                 std::to_string(int(band_max[c])) + " nm, add CHANNEL_WEIGHTS to change it";
      }
      return false;
    }
  }

  return true;
}

}  // namespace

bool medium_table_build(const string &text,
                        MediumTable &table,
                        string &error,
                        const int num_angles)
{
  MediumTableText medium;
  if (!parse_medium_table(text, medium, error)) {
    return false;
  }

  /* Resample and normalize every row. */
  const size_t num_rows = medium.is_rgb ? 3 : medium.wavelengths.size();
  const size_t num_input_angles = medium.angles.size();
  vector<vector<double>> row_density(num_rows);
  for (size_t r = 0; r < num_rows; r++) {
    if (!resample_phase(
            medium.angles, &medium.phase[r * num_input_angles], num_angles, row_density[r]))
    {
      error = "phase function row " + std::to_string(r + 1) + " is zero";
      return false;
    }
  }

  /* Combine rows into the density and coefficients of each channel. */
  vector<double> channel_density[3];
  double sigma_s[3];
  double sigma_a[3];

  if (medium.is_rgb) {
    for (int c = 0; c < 3; c++) {
      channel_density[c] = row_density[c];
      sigma_s[c] = medium.sigma_s[c];
      sigma_a[c] = medium.sigma_a[c];
    }
  }
  else {
    vector<double> weights;
    if (!channel_weights(medium, weights, error)) {
      return false;
    }

    for (int c = 0; c < 3; c++) {
      const double *w = &weights[c * num_rows];

      double sum_w = 0.0;
      double sum_sigma_s = 0.0;
      double sum_sigma_a = 0.0;
      for (size_t j = 0; j < num_rows; j++) {
        sum_w += w[j];
        sum_sigma_s += w[j] * medium.sigma_s[j];
        sum_sigma_a += w[j] * medium.sigma_a[j];
      }
      sigma_s[c] = sum_sigma_s / sum_w;
      sigma_a[c] = sum_sigma_a / sum_w;

      /* Weight phase functions by how much they scatter. Without any scattering in the band, the
       * phase function does not matter, use the plain weights. */
      const bool use_sigma_s = sum_sigma_s > 0.0;
      const double sum_phase_w = use_sigma_s ? sum_sigma_s : sum_w;

      channel_density[c].assign(num_angles, 0.0);
      for (size_t j = 0; j < num_rows; j++) {
        const double phase_w = w[j] * (use_sigma_s ? medium.sigma_s[j] : 1.0) / sum_phase_w;
        if (phase_w == 0.0) {
          continue;
        }
        for (int i = 0; i < num_angles; i++) {
          channel_density[c][i] += phase_w * row_density[j][i];
        }
      }
    }
  }

  /* Pack the table, see phase_tabulated_eval(). */
  const double dtheta = M_PI / (num_angles - 1);

  table.num_angles = num_angles;
  table.data.resize(size_t(PHASE_TABULATED_STRIDE) * num_angles);
  table.sigma_s = make_float3(float(sigma_s[0]), float(sigma_s[1]), float(sigma_s[2]));
  table.sigma_a = make_float3(float(sigma_a[0]), float(sigma_a[1]), float(sigma_a[2]));

  double cdf = 0.0;
  double g = 0.0;
  double prev_mean = 0.0;
  for (int i = 0; i < num_angles; i++) {
    const double mean = (channel_density[0][i] + channel_density[1][i] + channel_density[2][i]) /
                        3.0;
    if (i > 0) {
      cdf += 0.5 * (prev_mean + mean) * dtheta;
    }
    prev_mean = mean;

    /* Trapezoidal rule for the mean cosine. */
    const double end_weight = (i == 0 || i == num_angles - 1) ? 0.5 : 1.0;
    g += end_weight * mean * std::cos(i * dtheta) * dtheta;

    float *entry = &table.data[size_t(PHASE_TABULATED_STRIDE) * i];
    entry[0] = float(channel_density[0][i]);
    entry[1] = float(channel_density[1][i]);
    entry[2] = float(channel_density[2][i]);
    entry[3] = float(cdf);
  }

  /* Normalize away round-off, the sampling code relies on the CDF ending at exactly one. */
  for (int i = 0; i < num_angles; i++) {
    float &entry_cdf = table.data[size_t(PHASE_TABULATED_STRIDE) * i + 3];
    entry_cdf = float(std::min(double(entry_cdf) / cdf, 1.0));
  }
  table.data[size_t(PHASE_TABULATED_STRIDE) * (num_angles - 1) + 3] = 1.0f;

  table.g = float(g);

  return true;
}

CCL_NAMESPACE_END
