/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup shdnodes
 */

#include "node_shader_util.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "BKE_node_runtime.hh"

namespace blender {

namespace nodes::node_shader_volume_tabulated_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  const bNodeTree *ntree = b.tree_or_null();
  const bool is_gpu_internal = ntree && (ntree->flag & NTREE_IS_GPU_SHADER_INTERNAL);

  b.add_input<decl::Float>("Density"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .max(1000000.0f)
      .description(
          "Multiplier for the scattering and absorption coefficients of the medium table, for "
          "example the particle concentration when the table has coefficients per unit "
          "concentration");
  b.add_input<decl::Float>("Weight"_ustr).available(is_gpu_internal);
  b.add_output<decl::Shader>("Volume"_ustr).translation_context(BLT_I18NCONTEXT_ID_ID);
}

static void node_shader_buts_tabulated(ui::Layout &layout, bContext * /*C*/, PointerRNA *ptr)
{
  layout.prop(ptr, "table", ui::ITEM_R_SPLIT_EMPTY_NAME, "", ICON_TEXT);
}

}  // namespace nodes::node_shader_volume_tabulated_cc

/* node type definition */
void register_node_type_sh_volume_tabulated()
{
  namespace file_ns = nodes::node_shader_volume_tabulated_cc;

  static bke::bNodeType ntype;

  sh_node_type_base(&ntype, "ShaderNodeVolumeTabulated"_ustr, SH_NODE_VOLUME_TABULATED);
  ntype.ui_name = "Tabulated Volume";
  ntype.ui_description =
      "Scatter and absorb light with coefficients and per-channel phase functions from a medium "
      "table in a text data-block, for example computed with Mie theory (Cycles only)";
  ntype.enum_name_legacy = "VOLUME_TABULATED";
  ntype.nclass = NODE_CLASS_SHADER;
  ntype.declare = file_ns::node_declare;
  ntype.gather_link_search_ops = search_link_ops_for_shader_bsdf_node;
  ntype.add_ui_poll = object_shader_nodes_poll;
  ntype.draw_buttons = file_ns::node_shader_buts_tabulated;
  ntype.default_width = bke::NodeWidth::_160;

  bke::node_register_type(ntype);
}

}  // namespace blender
