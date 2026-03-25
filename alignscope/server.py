"""
AlignScope — Dashboard Server

Serves the live dashboard frontend and handles data ingestion from:
1. Built-in demo simulator (--demo mode)
2. SDK clients via WebSocket (/ws/sdk)
3. REST API posts (/api/log) for Tier 4 integration

Refactored from the original backend/main.py to work as part of
the installable package.
"""

import asyncio
import json
import os
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from alignscope.simulator import MARLSimulator, SimulatorConfig
from alignscope.metrics import AlignmentMetrics
from alignscope.detector import DefectionDetector

# Locate frontend files
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_CANDIDATES = [
    os.path.join(_PACKAGE_DIR, "_frontend"),                    # Installed via pip (bundled)
    os.path.join(os.path.dirname(_PACKAGE_DIR), "frontend"),    # Dev mode (source tree)
]
FRONTEND_DIR = next((d for d in _FRONTEND_CANDIDATES if os.path.isdir(d)), None)


def create_app(demo: bool = False) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="AlignScope", version="0.1.0")

    # CORS — allow any origin for SDK clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared state for SDK data ingestion
    # Each connected frontend gets its own queue (broadcast pattern)
    app.state.frontend_queues: Set[asyncio.Queue] = set()
    app.state.sdk_config = None
    app.state.demo_mode = demo

    # Serve frontend static files
    if FRONTEND_DIR:
        css_dir = os.path.join(FRONTEND_DIR, "css")
        js_dir = os.path.join(FRONTEND_DIR, "js")
        if os.path.isdir(css_dir):
            app.mount("/css", StaticFiles(directory=css_dir), name="css")
        if os.path.isdir(js_dir):
            app.mount("/js", StaticFiles(directory=js_dir), name="js")

    @app.get("/")
    async def root():
        if FRONTEND_DIR:
            return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
        return JSONResponse({"error": "Frontend not found"}, status_code=404)

    # ============================================================
    # REST API — Tier 4 (Any language)
    # ============================================================

    @app.get("/api/config")
    async def get_config():
        """Return current environment schema for discovery."""
        if app.state.sdk_config:
            return app.state.sdk_config
        # Return demo config as default
        sim = MARLSimulator()
        return sim.get_config_payload()

    @app.post("/api/log")
    async def api_log(request: Request):
        """
        REST endpoint for Tier 4 integration.
        Accepts JSON payloads from any language/framework.

        curl -X POST http://localhost:8000/api/log \\
          -H "Content-Type: application/json" \\
          -d '{"step": 100, "agents": [...]}'
        """
        body = await request.json()
        payload = {"type": "tick", "data": body} if "type" not in body else body
        _broadcast(app, payload)
        return {"status": "ok", "step": body.get("step")}

    @app.post("/api/config")
    async def set_config(request: Request):
        """Set the environment config from an external source."""
        body = await request.json()
        app.state.sdk_config = body
        return {"status": "ok"}

    # ============================================================
    # WebSocket — SDK ingestion endpoint
    # ============================================================

    @app.websocket("/ws/sdk")
    async def ws_sdk_endpoint(websocket: WebSocket):
        """Receives data from the AlignScope SDK (Python client)."""
        await websocket.accept()
        print("[SERVER] SDK client connected to /ws/sdk")
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                if payload.get("type") == "config":
                    app.state.sdk_config = payload.get("data")
                    # Broadcast config to all frontends
                    _broadcast(app, {"type": "config", "data": app.state.sdk_config})
                    print("[SERVER] SDK config received and broadcast")
                elif payload.get("type") == "tick":
                    _broadcast(app, payload)
        except WebSocketDisconnect:
            print("[SERVER] SDK client disconnected from /ws/sdk")

    # ============================================================
    # WebSocket — Dashboard frontend connection
    # ============================================================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        print(f"[SERVER] Frontend connected to /ws (demo_mode={app.state.demo_mode})")

        if app.state.demo_mode:
            await _run_demo_simulation(websocket)
        else:
            await _run_sdk_relay(websocket, app)

    return app


def _broadcast(app: FastAPI, payload: dict):
    """Broadcast a payload to all connected frontend queues."""
    for q in app.state.frontend_queues:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # Drop if a frontend is too slow


async def _run_demo_simulation(websocket: WebSocket):
    """Run the built-in demo simulator and stream to frontend."""
    sim = MARLSimulator(SimulatorConfig(seed=42))
    metrics_engine = AlignmentMetrics()
    detector = DefectionDetector()

    try:
        # Send config
        await websocket.send_json({
            "type": "config",
            "data": sim.get_config_payload(),
        })

        while not sim.is_finished:
            # Check for client commands
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                client_data = json.loads(msg)
                if client_data.get("action") == "restart":
                    sim = MARLSimulator(SimulatorConfig(seed=None))
                    metrics_engine = AlignmentMetrics()
                    detector = DefectionDetector()
                    await websocket.send_json({
                        "type": "config",
                        "data": sim.get_config_payload(),
                    })
                    await websocket.send_json({"type": "restart"})
                    continue
            except asyncio.TimeoutError:
                pass

            # Advance simulation
            tick_data = sim.step()
            tick_metrics = metrics_engine.update(tick_data)
            events = detector.analyze(tick_data, tick_metrics)
            relationships = sim.get_agent_relationships()

            payload = {
                "type": "tick",
                "data": {
                    "tick": tick_data["tick"],
                    "agents": tick_data["agents"],
                    "objectives": tick_data["objectives"],
                    "team_scores": tick_data["team_scores"],
                    "metrics": {
                        "agent_metrics": tick_metrics["agent_metrics"],
                        "pair_metrics": tick_metrics["pair_metrics"],
                        "team_metrics": {
                            str(k): v for k, v in tick_metrics["team_metrics"].items()
                        },
                        "overall_alignment_score": tick_metrics["overall_alignment_score"],
                    },
                    "relationships": relationships,
                    "events": [
                        {
                            "tick": e["tick"],
                            "type": e["type"],
                            "agent_id": e.get("agent_id"),
                            "team": e.get("team"),
                            "severity": e.get("severity", 0.5),
                            "description": e["description"],
                        }
                        for e in events
                    ],
                },
            }

            await websocket.send_json(payload)
            await asyncio.sleep(0.05)

        await websocket.send_json({
            "type": "episode_complete",
            "data": {
                "total_ticks": sim.tick,
                "defection_summary": detector.get_summary(),
            },
        })

    except WebSocketDisconnect:
        pass


async def _run_sdk_relay(websocket: WebSocket, app: FastAPI):
    """Relay data from SDK clients to the dashboard frontend.

    Each frontend connection gets its own asyncio.Queue. The SDK
    ingestion endpoint broadcasts to ALL queues, so every browser
    tab/window receives every tick.
    """
    # Create a dedicated queue for this frontend connection
    my_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    app.state.frontend_queues.add(my_queue)
    print(f"[SERVER] Frontend relay started (total clients: {len(app.state.frontend_queues)})")

    try:
        # Send config if available
        if app.state.sdk_config:
            await websocket.send_json({
                "type": "config",
                "data": app.state.sdk_config,
            })

        while True:
            # Check for client commands (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                client_data = json.loads(msg)
                if client_data.get("action") == "restart":
                    await websocket.send_json({"type": "restart"})
                    continue
            except asyncio.TimeoutError:
                pass

            # Check for SDK data from this connection's queue
            try:
                payload = my_queue.get_nowait()
                await websocket.send_json(payload)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        pass
    finally:
        app.state.frontend_queues.discard(my_queue)
        print(f"[SERVER] Frontend relay stopped (total clients: {len(app.state.frontend_queues)})")


def run_server(host: str = "0.0.0.0", port: int = 8000, demo: bool = False):
    """Start the AlignScope dashboard server."""
    import uvicorn

    app = create_app(demo=demo)

    if not FRONTEND_DIR:
        print("⚠️  Frontend files not found. Dashboard UI will not be available.")
        print("   Run from the project root or install via pip.")

    mode = "demo" if demo else "SDK"
    print(f"\n🔬 AlignScope Dashboard")
    print(f"   Mode: {mode}")
    print(f"   Dashboard:  http://localhost:{port}")
    print(f"   REST API:   http://localhost:{port}/api/log")
    print(f"   WebSocket:  ws://localhost:{port}/ws/sdk")
    print()

    uvicorn.run(app, host=host, port=port, log_level="info")
