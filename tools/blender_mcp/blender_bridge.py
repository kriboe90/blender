"""
Blender side of the Blender MCP server.

The MCP server starts Blender with this script and passes two private pipe file descriptors in the
environment: BLENDER_MCP_IN_FD for requests and BLENDER_MCP_OUT_FD for responses. Nothing listens
on the network; only the MCP server process that started Blender can send requests.

Protocol, one JSON object per line in each direction:

    {"id": 1, "cmd": "ping"}
    {"id": 2, "cmd": "exec", "code": "result = len(bpy.data.objects)"}
    {"id": 3, "cmd": "stop"}

Code runs on Blender's main thread, in a namespace that persists between requests. A variable
named `result` is returned, as JSON when possible. "stop", or the MCP server closing the pipe, ends
a headless session; with the user interface, the bridge stops and Blender stays open.
"""

import contextlib
import io
import json
import os
import queue
import threading
import traceback

import bpy

_jobs = queue.Queue()
_stop = threading.Event()
_namespace = {"__name__": "__blender_mcp__", "bpy": bpy}
_out = None


def _run(request):
    """Execute one request on Blender's main thread."""
    cmd = request.get("cmd")
    if cmd == "ping":
        return {
            "ok": True,
            "version": bpy.app.version_string,
            "build_branch": bpy.app.build_branch.decode(errors="replace"),
            "build_hash": bpy.app.build_hash.decode(errors="replace"),
            "background": bpy.app.background,
            "file": bpy.data.filepath,
            "pid": os.getpid(),
        }
    if cmd == "exec":
        output = io.StringIO()
        _namespace.pop("result", None)
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(request.get("code", ""), "<mcp>", "exec"), _namespace)
        except Exception:
            return {"ok": False, "output": output.getvalue(), "error": traceback.format_exc()}
        result = _namespace.get("result")
        try:
            json.dumps(result)
        except (TypeError, ValueError):
            result = repr(result)
        return {"ok": True, "output": output.getvalue(), "result": result}
    if cmd == "stop":
        _stop.set()
        return {"ok": True}
    return {"ok": False, "error": "unknown command: {!r}".format(cmd)}


def _process(request):
    try:
        response = _run(request)
    except Exception:
        response = {"ok": False, "error": traceback.format_exc()}
    response["id"] = request.get("id")
    _out.write(json.dumps(response) + "\n")
    _out.flush()


def _read_requests(stream):
    """Read requests from the MCP server; runs in a thread, the main thread executes them."""
    for line in stream:
        try:
            request = json.loads(line)
        except ValueError:
            request = None
        _jobs.put(request if isinstance(request, dict) else {"cmd": "invalid"})
    # The MCP server closed the pipe.
    _stop.set()


def _drain():
    while True:
        try:
            request = _jobs.get_nowait()
        except queue.Empty:
            return
        _process(request)


def _timer():
    """Run queued requests while the user interface is running."""
    _drain()
    if _stop.is_set():
        print("Blender MCP bridge stopped", flush=True)
        return None
    return 0.05


def main():
    global _out
    try:
        in_fd = int(os.environ["BLENDER_MCP_IN_FD"])
        out_fd = int(os.environ["BLENDER_MCP_OUT_FD"])
    except (KeyError, ValueError):
        print("Blender MCP bridge: must be started by the MCP server", flush=True)
        return

    _out = os.fdopen(out_fd, "w", encoding="utf-8")
    threading.Thread(target=_read_requests, args=(os.fdopen(in_fd, "r", encoding="utf-8"),),
                     daemon=True).start()
    print("Blender MCP bridge ready", flush=True)

    if bpy.app.background:
        # Headless: serve on the main thread until stopped, then let Blender exit.
        while not _stop.is_set() or not _jobs.empty():
            try:
                request = _jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            _process(request)
    else:
        bpy.app.timers.register(_timer, first_interval=0.1, persistent=True)


main()
