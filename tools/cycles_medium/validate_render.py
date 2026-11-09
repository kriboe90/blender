# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Render validation for the Cycles "Tabulated Volume" node.

Renders volumes with analytic Henyey-Greenstein closures and the same phase functions written as
medium tables, and checks that the images match within the Monte Carlo noise. The noise level is
measured by rendering the reference twice with different seeds.

Cases:
- hg: Volume Scatter (g = 0.7) against a table with the same phase function.
- chromatic: three Volume Scatter nodes, each scattering one color channel with its own
  anisotropy, against one table with a phase function per channel. This also checks that phase
  functions are mixed per color channel.
- coefficients: Volume Coefficients with colored scattering and absorption (g = -0.4) against a
  table with the same coefficients.

Usage::

    blender -b --factory-startup --python tools/cycles_medium/validate_render.py -- \\
        --device METAL --samples 256 --output /tmp/medium_validation

Devices: CPU, METAL, CUDA, OPTIX, HIP, ONEAPI. Exits with code 1 when a comparison fails.
"""

import argparse
import math
import os
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import medium_table  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="CPU", help="CPU, METAL, CUDA, OPTIX, HIP or ONEAPI")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=160, help="Image width in pixels")
    parser.add_argument("--output", default="", help="Directory for the rendered images")
    parser.add_argument("--cases", default="hg,chromatic,coefficients")
    return parser.parse_args(argv)


def setup_device(scene, device):
    if device == "CPU":
        scene.cycles.device = "CPU"
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = device
    if hasattr(prefs, "refresh_devices"):
        prefs.refresh_devices()
    else:
        prefs.get_devices()
    found = False
    for dev in prefs.devices:
        dev.use = (dev.type == device)
        found |= dev.use
    if not found:
        raise RuntimeError("No {:s} device found".format(device))
    scene.cycles.device = "GPU"


def setup_scene(args):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    setup_device(scene, args.device)

    cycles = scene.cycles
    cycles.samples = args.samples
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = False
    cycles.sample_clamp_direct = 0.0
    cycles.sample_clamp_indirect = 0.0
    cycles.max_bounces = 16
    cycles.volume_bounces = 8
    cycles.transparent_max_bounces = 16

    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution * 3 // 4
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"

    world = bpy.data.worlds.new("Black")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scene.world = world

    # Sun light traveling along +X.
    light_data = bpy.data.lights.new("Sun", type="SUN")
    light_data.energy = 4.0
    light_data.angle = 0.0
    light = bpy.data.objects.new("Sun", light_data)
    light.rotation_euler = (0.0, -math.pi / 2.0, 0.0)
    scene.collection.objects.link(light)

    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera

    bpy.ops.mesh.primitive_cube_add(size=2.0)
    volume = bpy.context.active_object
    volume.name = "Volume"

    return scene, camera, volume


def place_camera(camera, angle_deg):
    """Place the camera so that light reaching it scattered by roughly angle_deg."""
    angle = math.radians(angle_deg)
    camera.location = (8.0 * math.cos(angle), 8.0 * math.sin(angle), 1.5)
    direction = Vector((0.0, 0.0, 0.0)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def new_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.remove(nodes["Principled BSDF"])
    return mat, nodes, mat.node_tree.links, nodes["Material Output"]


def material_scatter(g):
    mat, nodes, links, output = new_material("Scatter")
    node = nodes.new("ShaderNodeVolumeScatter")
    node.phase = "HENYEY_GREENSTEIN"
    node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    node.inputs["Density"].default_value = 1.0
    node.inputs["Anisotropy"].default_value = g
    links.new(node.outputs["Volume"], output.inputs["Volume"])
    return mat


def material_chromatic_scatter(g_rgb):
    mat, nodes, links, output = new_material("ChromaticScatter")
    shaders = []
    for c, g in enumerate(g_rgb):
        node = nodes.new("ShaderNodeVolumeScatter")
        node.phase = "HENYEY_GREENSTEIN"
        color = [0.0, 0.0, 0.0, 1.0]
        color[c] = 1.0
        node.inputs["Color"].default_value = color
        node.inputs["Density"].default_value = 1.0
        node.inputs["Anisotropy"].default_value = g
        shaders.append(node.outputs["Volume"])
    add1 = nodes.new("ShaderNodeAddShader")
    add2 = nodes.new("ShaderNodeAddShader")
    links.new(shaders[0], add1.inputs[0])
    links.new(shaders[1], add1.inputs[1])
    links.new(add1.outputs[0], add2.inputs[0])
    links.new(shaders[2], add2.inputs[1])
    links.new(add2.outputs[0], output.inputs["Volume"])
    return mat


def material_coefficients(sigma_s, sigma_a, g):
    mat, nodes, links, output = new_material("Coefficients")
    node = nodes.new("ShaderNodeVolumeCoefficients")
    node.phase = "HENYEY_GREENSTEIN"
    node.inputs["Scatter Coefficients"].default_value = sigma_s
    node.inputs["Absorption Coefficients"].default_value = sigma_a
    node.inputs["Anisotropy"].default_value = g
    node.inputs["Emission Coefficients"].default_value = (0.0, 0.0, 0.0)
    links.new(node.outputs["Volume"], output.inputs["Volume"])
    return mat


def material_tabulated(name, g_rgb, sigma_s, sigma_a, output_dir):
    path = os.path.join(output_dir, name + ".txt")
    angles = [0.1 * i for i in range(1801)]
    medium_table.write(
        path,
        angles_deg=angles,
        phase=[[medium_table.henyey_greenstein(a, g) for a in angles] for g in g_rgb],
        sigma_s=sigma_s,
        sigma_a=sigma_a,
        comment=name,
    )
    text = bpy.data.texts.load(path)

    mat, nodes, links, output = new_material("Tabulated")
    node = nodes.new("ShaderNodeVolumeTabulated")
    node.table = text
    node.inputs["Density"].default_value = 1.0
    links.new(node.outputs["Volume"], output.inputs["Volume"])
    return mat


def render(scene, volume, material, seed, path):
    volume.data.materials.clear()
    volume.data.materials.append(material)
    scene.cycles.seed = seed
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(path)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(image.size[1], image.size[0], 4)[:, :, :3].astype(np.float64)
    bpy.data.images.remove(image)
    return pixels


def block_means(pixels, block=8):
    h = pixels.shape[0] // block * block
    w = pixels.shape[1] // block * block
    p = pixels[:h, :w]
    return p.reshape(h // block, block, w // block, block, 3).mean(axis=(1, 3))


def relative_rms(a, b, mask):
    d = (a - b)[mask] / b[mask]
    return float(np.sqrt(np.mean(d * d)))


def compare(reference, reference2, test):
    """Return (ok, report). The test must match as well as two independent references do."""
    b_ref = block_means(reference)
    b_ref2 = block_means(reference2)
    b_test = block_means(test)
    mask = b_ref > 0.05 * b_ref.max()

    noise = relative_rms(b_ref2, b_ref, mask)
    diff = relative_rms(b_test, b_ref, mask)

    mean_ref = reference.mean(axis=(0, 1))
    mean_noise = np.abs(reference2.mean(axis=(0, 1)) / mean_ref - 1.0)
    mean_diff = np.abs(test.mean(axis=(0, 1)) / mean_ref - 1.0)

    ok = diff <= 2.0 * noise + 0.01 and bool(np.all(mean_diff <= 3.0 * mean_noise + 0.005))
    report = (
        "mean RGB ref {:s} | rel. mean diff {:s} (noise {:s}) | block rms diff {:.4f} "
        "(noise {:.4f})".format(
            np.array2string(mean_ref, precision=4),
            np.array2string(mean_diff, precision=4),
            np.array2string(mean_noise, precision=4),
            diff,
            noise,
        )
    )
    return ok, report


def main():
    args = parse_args()
    output_dir = args.output or tempfile.mkdtemp(prefix="medium_validation_")
    os.makedirs(output_dir, exist_ok=True)

    scene, camera, volume = setup_scene(args)

    cases = {
        "hg": (
            lambda: material_scatter(0.7),
            lambda: material_tabulated("hg", (0.7, 0.7, 0.7), (1, 1, 1), (0, 0, 0), output_dir),
        ),
        "chromatic": (
            lambda: material_chromatic_scatter((0.9, 0.3, 0.0)),
            lambda: material_tabulated("chromatic", (0.9, 0.3, 0.0), (1, 1, 1), (0, 0, 0), output_dir),
        ),
        "coefficients": (
            lambda: material_coefficients((0.6, 1.0, 1.4), (0.3, 0.2, 0.1), -0.4),
            lambda: material_tabulated(
                "coefficients", (-0.4, -0.4, -0.4), (0.6, 1.0, 1.4), (0.3, 0.2, 0.1), output_dir),
        ),
    }

    failures = 0
    for name in args.cases.split(","):
        make_reference, make_test = cases[name]
        reference_mat = make_reference()
        test_mat = make_test()
        for view in (35, 120):
            place_camera(camera, view)
            prefix = os.path.join(output_dir, "{:s}_{:s}_view{:d}".format(args.device, name, view))
            reference = render(scene, volume, reference_mat, 0, prefix + "_reference.exr")
            reference2 = render(scene, volume, reference_mat, 1, prefix + "_reference_seed1.exr")
            test = render(scene, volume, test_mat, 0, prefix + "_tabulated.exr")
            ok, report = compare(reference, reference2, test)
            failures += not ok
            print("[{:s}] {:s} view {:d}: {:s}".format("PASS" if ok else "FAIL", name, view, report))

    print("Images in", output_dir)
    print("{:d} comparison(s) failed".format(failures) if failures else "All comparisons passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
