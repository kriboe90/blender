# Blender MCP server

An MCP server that lets Claude control a Blender build: run Python, inspect the scene, set up
Tabulated Volume materials, and render with preview images. It uses only the Python standard
library (3.9+) and has no dependencies.

## How it works

`blender_mcp_server.py` speaks MCP over stdio. It starts Blender itself with
`blender_bridge.py`, and sends requests through two private pipes. Nothing listens on the
network, so only this server can send requests to that Blender.

There are two kinds of sessions:

- **Headless** (default) runs Blender without a window. It uses its own user folders in
  `~/.cache/blender-mcp/user`, so it never changes your preferences. The render device (Metal,
  CUDA, OptiX…) is configured automatically.
- **GUI** (`launch` with `mode="gui"`) opens a Blender window that you and Claude both work in. It
  uses your normal preferences and startup file, so GPU devices must be enabled there. `stop`
  disconnects Claude and leaves the window open.

The server cannot attach to a Blender you started yourself. To share a window with Claude, have
it run `launch` with `mode="gui"`, optionally with a `.blend` file.

## Tools

| Tool | Purpose |
|---|---|
| `launch`, `status`, `stop` | Start, inspect and end the session |
| `exec_python` | Run bpy code in the session; assign `result` to return data |
| `scene_info` | Render settings, camera, objects (volume grids), materials (medium tables) |
| `render` | Render the current scene, and return the file path and a preview image |
| `save_blend` | Save the open file |
| `set_tabulated_volume` | Volume material with the Tabulated Volume node from a medium table file. Density, or a volume grid × density, scales the coefficients. |
| `run_headless` | Run code in a separate, short-lived headless Blender, e.g. for batch renders |
| `read_log` | The session's Blender output, e.g. medium table errors |

Renders go to `~/.cache/blender-mcp/renders` unless `output_path` is given. The session log is
in `~/.cache/blender-mcp/blender.log`.

## Registering with Claude Code

```bash
claude mcp add blender -s user -- /usr/bin/python3 /path/to/tools/blender_mcp/blender_mcp_server.py --blender /path/to/Blender
```

On macOS, the Blender executable is `…/Blender.app/Contents/MacOS/Blender`. New Claude Code
sessions then have the tools as `mcp__blender__…`. Remove the server with:

```bash
claude mcp remove blender -s user
```

## Security

`exec_python` and `run_headless` run arbitrary Python with your user rights, just like running a
script in Blender. Claude Code asks for permission before each tool call, unless you allow the
tools in its settings.
