#!/usr/bin/env python3
"""
MCP server for a Blender build, over stdio, using only the Python standard library (3.9+).

The server starts Blender itself (headless, or with its user interface) and controls it through two
private pipes, see blender_bridge.py. Nothing listens on the network.

Tools:
  launch, status, stop        start and manage the Blender session
  exec_python                 run bpy code in the session
  scene_info                  summary of the open scene
  render                      render the current scene, returns a preview image
  save_blend                  save the open file
  set_tabulated_volume        assign a Tabulated Volume material (medium table file)
  run_headless                run code in a separate, short-lived headless Blender
  read_log                    end of the session's Blender output

Headless runs use their own user folders in ~/.cache/blender-mcp/user, so they never change the
user's preferences.

Register with Claude Code:
  claude mcp add blender -s user -- /usr/bin/python3 /path/to/blender_mcp_server.py \\
      --blender /path/to/Blender
"""

import argparse
import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time

SERVER_INFO = {"name": "blender", "version": "0.1.0"}
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE_SCRIPT = os.path.join(HERE, "blender_bridge.py")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "blender-mcp")
USER_DIR = os.path.join(CACHE_DIR, "user")
RENDER_DIR = os.path.join(CACHE_DIR, "renders")
LOG_FILE = os.path.join(CACHE_DIR, "blender.log")
MAX_TEXT = 20000

INSTRUCTIONS = """\
Controls a Blender build. Start a session with `launch` (headless by default, mode="gui" shows the
Blender window), then use `scene_info`, `exec_python` (bpy code; assign to `result` to return data),
`render` (returns a preview image), `save_blend` and `stop`. `run_headless` runs a one-off script
in a fresh headless Blender, independent of the session.

This build may include the Cycles "Tabulated Volume" shader node (ShaderNodeVolumeTabulated):
scattering and absorption coefficients and per-channel phase functions from a medium table text
file (format: CYCLES_MEDIUM_TABLE 1, WAVELENGTHS_NM or CHANNELS RGB, ANGLES_DEG, SIGMA_S, SIGMA_A,
PHASE, optional CHANNEL_WEIGHTS). `set_tabulated_volume` assigns one to an object; the node's
Density input scales the table coefficients (e.g. soot density in kg/m^3 for mass-specific tables).

Headless sessions use isolated user folders and never change the user's preferences. With
mode="gui", Blender uses the user's normal preferences and startup file."""


class ToolError(Exception):
    """Reported to the client as a failed tool call."""


def truncate(text, limit=MAX_TEXT):
    if len(text) <= limit:
        return text
    return "[... {} characters omitted ...]\n".format(len(text) - limit) + text[-limit:]


def isolated_env():
    """Environment with Blender user folders in the cache, to never touch the user's settings."""
    env = dict(os.environ)
    folders = (
        ("BLENDER_USER_RESOURCES", ""),
        ("BLENDER_USER_CONFIG", "config"),
        ("BLENDER_USER_SCRIPTS", "scripts"),
        ("BLENDER_USER_DATAFILES", "datafiles"),
        ("BLENDER_USER_EXTENSIONS", "extensions"),
    )
    for variable, sub_folder in folders:
        path = os.path.join(USER_DIR, sub_folder) if sub_folder else USER_DIR
        os.makedirs(path, exist_ok=True)
        env[variable] = path
    return env


class Session:
    """A Blender process started by this server, controlled through two private pipes."""

    def __init__(self, blender, mode, blend_file):
        os.makedirs(CACHE_DIR, exist_ok=True)
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()

        env = isolated_env() if mode == "background" else dict(os.environ)
        env["BLENDER_MCP_IN_FD"] = str(request_read)
        env["BLENDER_MCP_OUT_FD"] = str(response_write)

        args = [blender]
        if mode == "background":
            args.append("-b")
        if blend_file:
            args.append(blend_file)
        args += ["--python", BRIDGE_SCRIPT]

        self.mode = mode
        self.log = open(LOG_FILE, "w")
        try:
            self.process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                env=env,
                pass_fds=(request_read, response_write),
                # A window the user works in must outlive this server.
                start_new_session=(mode == "gui"),
            )
        except OSError as ex:
            for fd in (request_read, request_write, response_read, response_write):
                os.close(fd)
            self.log.close()
            raise ToolError("Cannot start Blender ({}): {}".format(blender, ex))
        os.close(request_read)
        os.close(response_write)

        self.requests = os.fdopen(request_write, "w", encoding="utf-8")
        self.responses = queue.Queue()
        self.next_id = 1
        threading.Thread(
            target=self._read_responses,
            args=(os.fdopen(response_read, "r", encoding="utf-8"),),
            daemon=True,
        ).start()

    def _read_responses(self, stream):
        for line in stream:
            try:
                self.responses.put(json.loads(line))
            except ValueError:
                continue
        self.responses.put(None)

    def alive(self):
        return self.process.poll() is None

    def call(self, request, timeout):
        if not self.alive():
            raise ToolError("The Blender session has exited. Use `launch` to start a new one.")
        request = dict(request, id=self.next_id)
        self.next_id += 1
        try:
            self.requests.write(json.dumps(request) + "\n")
            self.requests.flush()
        except OSError:
            raise ToolError("The Blender session no longer accepts requests. See `read_log`.")

        deadline = time.time() + timeout
        while True:
            try:
                response = self.responses.get(timeout=0.5)
            except queue.Empty:
                if not self.alive():
                    raise ToolError("Blender exited while handling the request. See `read_log`.")
                if time.time() > deadline:
                    raise ToolError("Timed out after {} s waiting for Blender.".format(timeout))
                continue
            if response is None:
                raise ToolError("The Blender session closed the connection. See `read_log`.")
            if response.get("id") == request["id"]:
                return response
            # A late response to an earlier request that timed out.

    def exec(self, code, timeout=600):
        response = self.call({"cmd": "exec", "code": code}, timeout)
        if not response.get("ok"):
            raise ToolError(truncate((response.get("output") or "") +
                                     (response.get("error") or "Unknown error")))
        return response

    def stop(self, timeout=15):
        if self.alive():
            try:
                self.call({"cmd": "stop"}, timeout)
            except ToolError:
                pass
        try:
            self.requests.close()
        except OSError:
            pass
        if self.mode == "background":
            try:
                self.process.wait(timeout)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        self.log.close()


# Code snippets that run inside Blender. `ARGS` is replaced with a JSON string literal.

SCENE_INFO_CODE = r'''
import bpy

scene = bpy.context.scene
cycles = scene.render.engine == "CYCLES"


def object_info(ob):
    info = {
        "name": ob.name,
        "type": ob.type,
        "location": [round(v, 4) for v in ob.location],
        "dimensions": [round(v, 4) for v in ob.dimensions],
        "materials": [slot.material.name for slot in ob.material_slots if slot.material],
        "hide_render": ob.hide_render,
    }
    if ob.type == "VOLUME" and ob.data:
        ob.data.grids.load()
        info["volume_file"] = ob.data.filepath
        info["grids"] = [grid.name for grid in ob.data.grids]
    if ob.type == "LIGHT":
        info["light"] = {"type": ob.data.type, "energy": ob.data.energy,
                         "color": list(ob.data.color)}
    return info


def material_info(mat):
    info = {"name": mat.name, "users": mat.users}
    if mat.node_tree:
        info["nodes"] = sorted({node.bl_idname for node in mat.node_tree.nodes})
        tables = [node.table.name for node in mat.node_tree.nodes
                  if node.bl_idname == "ShaderNodeVolumeTabulated" and node.table]
        if tables:
            info["medium_tables"] = tables
    return info


result = {
    "file": bpy.data.filepath,
    "scene": scene.name,
    "engine": scene.render.engine,
    "cycles": {"device": scene.cycles.device, "samples": scene.cycles.samples,
               "volume_bounces": scene.cycles.volume_bounces} if cycles else None,
    "resolution": [scene.render.resolution_x, scene.render.resolution_y,
                   scene.render.resolution_percentage],
    "frames": {"start": scene.frame_start, "end": scene.frame_end,
               "current": scene.frame_current},
    "camera": scene.camera.name if scene.camera else None,
    "unit_scale": scene.unit_settings.scale_length,
    "objects": [object_info(ob) for ob in scene.objects],
    "materials": [material_info(mat) for mat in bpy.data.materials],
    "texts": [text.name for text in bpy.data.texts],
    "has_tabulated_volume_node": hasattr(bpy.types, "ShaderNodeVolumeTabulated"),
}
'''

RENDER_CODE = r'''
import json
import os
import time

import bpy

args = json.loads(ARGS)
scene = bpy.context.scene
settings = scene.render.image_settings
cycles = scene.render.engine == "CYCLES"

if args.get("frame") is not None:
    scene.frame_set(int(args["frame"]))
if args.get("camera"):
    scene.camera = bpy.data.objects[args["camera"]]
for key in ("resolution_x", "resolution_y", "resolution_percentage"):
    if args.get(key):
        setattr(scene.render, key, int(args[key]))
if args.get("samples"):
    if not cycles:
        raise RuntimeError("samples only applies to Cycles")
    scene.cycles.samples = int(args["samples"])

device = args.get("device")
if device:
    if not cycles:
        raise RuntimeError("device only applies to Cycles")
    if device == "CPU":
        scene.cycles.device = "CPU"
    else:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        if bpy.app.background:
            # Headless sessions have their own preferences, configure the device there.
            prefs.compute_device_type = device
            prefs.refresh_devices()
            for dev in prefs.devices:
                dev.use = dev.type == device
        if prefs.compute_device_type != device or not any(
                dev.use and dev.type == device for dev in prefs.devices):
            raise RuntimeError(
                device + " is not enabled in Preferences > System > Cycles Render Devices")
        scene.cycles.device = "GPU"

settings.file_format = args.get("file_format", "PNG")
if settings.file_format == "OPEN_EXR":
    settings.color_depth = "32"
scene.render.filepath = args["output_path"]

start = time.time()
bpy.ops.render.render(write_still=True)
# A still is written to the file path itself, with the extension added when missing.
path = bpy.path.abspath(scene.render.filepath)
extension = {"PNG": ".png", "OPEN_EXR": ".exr"}.get(settings.file_format, "")
if scene.render.use_file_extension and extension and not path.lower().endswith(extension):
    path += extension
if not os.path.isfile(path):
    raise RuntimeError("Render finished but no file was written to " + path)

result = {
    "path": path,
    "seconds": round(time.time() - start, 2),
    "engine": scene.render.engine,
    "device": scene.cycles.device if cycles else None,
    "samples": scene.cycles.samples if cycles else None,
    "resolution": [scene.render.resolution_x * scene.render.resolution_percentage // 100,
                   scene.render.resolution_y * scene.render.resolution_percentage // 100],
}

preview = args.get("preview_path")
if preview:
    # Display-referred preview through the scene's view transform, at most 1024 pixels wide.
    saved = (settings.file_format, settings.color_depth, settings.color_mode)
    settings.file_format = "PNG"
    settings.color_depth = "8"
    settings.color_mode = "RGB"
    try:
        bpy.data.images["Render Result"].save_render(preview, scene=scene)
    finally:
        settings.file_format = saved[0]
        settings.color_depth = saved[1]
        settings.color_mode = saved[2]
    image = bpy.data.images.load(preview)
    width, height = image.size
    scale = min(1.0, 1024.0 / max(width, height, 1))
    if scale < 1.0:
        image.scale(max(1, int(width * scale)), max(1, int(height * scale)))
        image.save()
    bpy.data.images.remove(image)
    result["preview"] = preview
'''

TABULATED_VOLUME_CODE = r'''
import json
import os

import bpy

args = json.loads(ARGS)
if not hasattr(bpy.types, "ShaderNodeVolumeTabulated"):
    raise RuntimeError("This Blender build has no Tabulated Volume node")

ob = bpy.data.objects[args["object"]]
path = os.path.abspath(os.path.expanduser(args["table_path"]))
with open(path, encoding="utf-8") as fh:
    content = fh.read()
if not content.lstrip().startswith("CYCLES_MEDIUM_TABLE"):
    raise RuntimeError("Not a medium table: " + path)

# Reuse the text data-block of this file, with fresh content.
text = None
for candidate in bpy.data.texts:
    if candidate.filepath and os.path.abspath(bpy.path.abspath(candidate.filepath)) == path:
        text = candidate
        break
if text is None:
    text = bpy.data.texts.load(path)
else:
    text.clear()
    text.write(content)

name = args.get("material") or ("Tabulated " + os.path.splitext(os.path.basename(path))[0])
mat = bpy.data.materials.get(name)
created = mat is None
if created:
    mat = bpy.data.materials.new(name)
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
if created:
    # Volume only: no surface shader, so mesh domains are transparent at their boundary.
    for node in list(nodes):
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)

output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
if output is None:
    output = nodes.new("ShaderNodeOutputMaterial")
volume = next((n for n in nodes if n.bl_idname == "ShaderNodeVolumeTabulated"), None)
if volume is None:
    volume = nodes.new("ShaderNodeVolumeTabulated")
    volume.location = (output.location.x - 250, output.location.y - 150)
volume.table = text
links.new(volume.outputs["Volume"], output.inputs["Volume"])

density = float(args.get("density", 1.0))
density_input = volume.inputs["Density"]
for link in list(density_input.links):
    links.remove(link)
attribute_name = args.get("density_attribute")
if attribute_name:
    attribute = nodes.get("MCP Density Attribute") or nodes.new("ShaderNodeAttribute")
    attribute.name = "MCP Density Attribute"
    attribute.attribute_name = attribute_name
    attribute.location = (volume.location.x - 400, volume.location.y)
    scale = nodes.get("MCP Density Scale") or nodes.new("ShaderNodeMath")
    scale.name = "MCP Density Scale"
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = density
    scale.location = (volume.location.x - 200, volume.location.y)
    links.new(attribute.outputs["Fac"], scale.inputs[0])
    links.new(scale.outputs["Value"], density_input)
else:
    density_input.default_value = density

if mat.name not in [slot.material.name for slot in ob.material_slots if slot.material]:
    if ob.material_slots:
        ob.material_slots[0].material = mat
    else:
        ob.data.materials.append(mat)

result = {"object": ob.name, "material": mat.name, "text": text.name, "density": density,
          "density_attribute": attribute_name}
'''


def snippet(code, args):
    return code.replace("ARGS", repr(json.dumps(args)), 1)


def text_content(text):
    return {"type": "text", "text": text}


def json_content(data):
    return text_content(json.dumps(data, indent=2))


TOOLS = [
    {
        "name": "launch",
        "description": (
            "Start a Blender session that the other tools act on. Headless by default; "
            "mode='gui' opens the Blender window. Optionally opens a .blend file. If a session "
            "is already running, reports it instead."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["background", "gui"], "default": "background"},
                "blend_file": {"type": "string", "description": "Absolute path of a .blend file"},
                "wait_s": {"type": "number", "default": 120,
                           "description": "Seconds to wait for Blender to start"},
            },
        },
    },
    {
        "name": "status",
        "description": "Report whether a Blender session is running, its version and open file.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "stop",
        "description": (
            "Stop the session. A headless Blender exits; a Blender window stays open for the "
            "user, it is only disconnected."),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "exec_python",
        "description": (
            "Run Python code in the Blender session (bpy is imported). Variables persist between "
            "calls. Assign to `result` to return JSON-compatible data; printed output is returned "
            "too."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout_s": {"type": "number", "default": 600},
            },
            "required": ["code"],
        },
    },
    {
        "name": "scene_info",
        "description": (
            "Summarize the open scene: render settings, camera, objects (with volume grids), "
            "materials (with medium tables of Tabulated Volume nodes) and text data-blocks."),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "render",
        "description": (
            "Render the current scene to a file and return a preview image. The given settings "
            "are applied to the scene. GPU devices must be enabled in the preferences of a GUI "
            "session; headless sessions configure them automatically."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string",
                                "description": "Output file; default in ~/.cache/blender-mcp/renders"},
                "file_format": {"type": "string", "enum": ["PNG", "OPEN_EXR"], "default": "PNG"},
                "device": {"type": "string",
                           "enum": ["CPU", "METAL", "CUDA", "OPTIX", "HIP", "ONEAPI"]},
                "samples": {"type": "integer"},
                "resolution_x": {"type": "integer"},
                "resolution_y": {"type": "integer"},
                "resolution_percentage": {"type": "integer"},
                "frame": {"type": "integer"},
                "camera": {"type": "string", "description": "Camera object name"},
                "preview": {"type": "boolean", "default": True},
                "timeout_s": {"type": "number", "default": 3600},
            },
        },
    },
    {
        "name": "save_blend",
        "description": "Save the open file, to `path` if given (absolute path).",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
    {
        "name": "set_tabulated_volume",
        "description": (
            "Give an object a volume material with the Tabulated Volume node, using a medium "
            "table file. Density scales the table coefficients; with density_attribute, the "
            "volume grid of that name times density drives it."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "object": {"type": "string"},
                "table_path": {"type": "string", "description": "Absolute path of the medium table"},
                "density": {"type": "number", "default": 1.0},
                "density_attribute": {"type": "string",
                                      "description": "Volume grid name, e.g. 'density'"},
                "material": {"type": "string", "description": "Material name"},
            },
            "required": ["object", "table_path"],
        },
    },
    {
        "name": "run_headless",
        "description": (
            "Run Python code in a separate, short-lived headless Blender (factory settings, or "
            "the given .blend file) and return its exit code and output. Independent of the "
            "session; useful for batch renders and scripts."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "blend_file": {"type": "string"},
                "timeout_s": {"type": "number", "default": 1800},
            },
            "required": ["code"],
        },
    },
    {
        "name": "read_log",
        "description": "Return the last lines of the session's Blender output (errors, warnings).",
        "inputSchema": {
            "type": "object",
            "properties": {"lines": {"type": "integer", "default": 100}},
        },
    },
]


class Server:
    def __init__(self, blender):
        self.blender = blender
        self.session = None

    def require_session(self):
        if self.session is None or not self.session.alive():
            raise ToolError("No Blender session is running. Use `launch` first.")
        return self.session

    # Tools

    def tool_launch(self, args):
        if self.session is not None and self.session.alive():
            info = self.session.call({"cmd": "ping"}, 30)
            return [text_content("A Blender session is already running."), json_content(info)]
        mode = args.get("mode", "background")
        if mode not in ("background", "gui"):
            raise ToolError("mode must be 'background' or 'gui'")
        blend_file = args.get("blend_file")
        if blend_file:
            blend_file = os.path.abspath(os.path.expanduser(blend_file))
            if not os.path.isfile(blend_file):
                raise ToolError("No such file: " + blend_file)
        self.session = Session(self.blender, mode, blend_file)
        info = self.session.call({"cmd": "ping"}, float(args.get("wait_s", 120)))
        return [text_content("Blender session started ({}).".format(mode)), json_content(info)]

    def tool_status(self, args):
        if self.session is None or not self.session.alive():
            return [text_content("No Blender session is running. Blender: " + self.blender)]
        info = self.session.call({"cmd": "ping"}, 30)
        info["mode"] = self.session.mode
        return [json_content(info)]

    def tool_stop(self, args):
        if self.session is None:
            return [text_content("No Blender session is running.")]
        mode = self.session.mode
        self.session.stop()
        self.session = None
        if mode == "gui":
            return [text_content("Disconnected. The Blender window stays open.")]
        return [text_content("Blender session stopped.")]

    def tool_exec_python(self, args):
        response = self.require_session().exec(args["code"], float(args.get("timeout_s", 600)))
        content = []
        if response.get("output"):
            content.append(text_content(truncate(response["output"])))
        content.append(json_content({"result": response.get("result")}))
        return content

    def tool_scene_info(self, args):
        response = self.require_session().exec(SCENE_INFO_CODE, 120)
        return [json_content(response["result"])]

    def tool_render(self, args):
        session = self.require_session()
        os.makedirs(RENDER_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        render_args = dict(args)
        extension = ".exr" if args.get("file_format") == "OPEN_EXR" else ".png"
        render_args["output_path"] = os.path.abspath(os.path.expanduser(
            args.get("output_path") or os.path.join(RENDER_DIR, "render-" + stamp + extension)))
        if args.get("preview", True):
            render_args["preview_path"] = os.path.join(RENDER_DIR, "preview-" + stamp + ".png")
        response = session.exec(snippet(RENDER_CODE, render_args),
                                 float(args.get("timeout_s", 3600)))
        result = response["result"]
        content = [json_content(result)]
        preview = result.get("preview")
        if preview and os.path.isfile(preview):
            with open(preview, "rb") as fh:
                data = base64.b64encode(fh.read()).decode("ascii")
            content.append({"type": "image", "data": data, "mimeType": "image/png"})
        return content

    def tool_save_blend(self, args):
        path = args.get("path")
        if path:
            path = os.path.abspath(os.path.expanduser(path))
            code = "bpy.ops.wm.save_as_mainfile(filepath={!r})\nresult = bpy.data.filepath".format(path)
        else:
            code = ("if not bpy.data.filepath:\n"
                    "    raise RuntimeError('The file has never been saved, give a path')\n"
                    "bpy.ops.wm.save_mainfile()\n"
                    "result = bpy.data.filepath")
        response = self.require_session().exec(code, 300)
        return [text_content("Saved " + str(response["result"]))]

    def tool_set_tabulated_volume(self, args):
        response = self.require_session().exec(snippet(TABULATED_VOLUME_CODE, args), 120)
        return [json_content(response["result"])]

    def tool_run_headless(self, args):
        command = [self.blender, "-b"]
        if args.get("blend_file"):
            command.append(os.path.abspath(os.path.expanduser(args["blend_file"])))
        else:
            command.append("--factory-startup")
        command += ["--python-exit-code", "1", "--python-expr", args["code"]]
        try:
            process = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                     text=True, timeout=float(args.get("timeout_s", 1800)),
                                     env=isolated_env())
        except subprocess.TimeoutExpired:
            raise ToolError("Timed out")
        except OSError as ex:
            raise ToolError("Cannot start Blender ({}): {}".format(self.blender, ex))
        output = (process.stdout or "") + (process.stderr or "")
        return [text_content("Exit code {}\n\n{}".format(process.returncode, truncate(output)))]

    def tool_read_log(self, args):
        try:
            with open(LOG_FILE, encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            return [text_content("No log yet.")]
        count = max(1, int(args.get("lines", 100)))
        return [text_content(truncate("".join(lines[-count:])))]

    # Protocol

    def call_tool(self, name, args):
        handler = getattr(self, "tool_" + str(name), None)
        if handler is None or name not in {tool["name"] for tool in TOOLS}:
            return {"content": [text_content("Unknown tool: " + str(name))], "isError": True}
        try:
            return {"content": handler(args), "isError": False}
        except ToolError as ex:
            return {"content": [text_content(str(ex))], "isError": True}
        except Exception as ex:
            return {"content": [text_content("{}: {}".format(type(ex).__name__, ex))],
                    "isError": True}

    def handle(self, message):
        method = message.get("method")
        message_id = message.get("id")
        if message_id is None:
            return None  # Notification, e.g. notifications/initialized.
        params = message.get("params") or {}
        if method == "initialize":
            requested = params.get("protocolVersion")
            result = {
                "protocolVersion": requested if requested in PROTOCOL_VERSIONS
                else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": INSTRUCTIONS,
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            result = self.call_tool(params.get("name"), params.get("arguments") or {})
        else:
            return {"jsonrpc": "2.0", "id": message_id,
                    "error": {"code": -32601, "message": "Method not found: " + str(method)}}
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    def shutdown(self):
        if self.session is not None:
            self.session.stop()
            self.session = None


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="MCP server for a Blender build")
    parser.add_argument("--blender", default=os.environ.get("BLENDER_MCP_BLENDER", "blender"),
                        help="Path of the Blender executable")
    args = parser.parse_args()

    server = Server(args.blender)
    try:
        for line in iter(sys.stdin.readline, ""):
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                send({"jsonrpc": "2.0", "id": None,
                      "error": {"code": -32700, "message": "Parse error"}})
                continue
            if not isinstance(message, dict):
                send({"jsonrpc": "2.0", "id": None,
                      "error": {"code": -32600, "message": "Invalid request"}})
                continue
            response = server.handle(message)
            if response is not None:
                send(response)
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
