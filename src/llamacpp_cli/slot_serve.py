"""Slot-based serving with NUMA awareness."""

from __future__ import annotations

import asyncio
import signal
import sys

import click
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .slot_manager import SlotManager


def create_slot_app(manager: SlotManager) -> FastAPI:
    """Create FastAPI app for slot-based serving."""
    app = FastAPI(title="llamacpp-slot-server")

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return JSONResponse({"status": "ok"})

    @app.get("/slots")
    async def list_slots():
        """List all slots and their status."""
        return JSONResponse(manager.get_all_status())

    @app.post("/load")
    async def load_model(request: Request):
        """Load a model on best available slot.

        Body:
            {
                "model": "model-name or path",
                "ctx_size": 16384 (optional),
                "extra_args": ["--arg1", "value1"] (optional)
            }
        """
        try:
            body = await request.json()
            model = body.get("model")
            ctx_size = body.get("ctx_size")
            extra_args = body.get("extra_args")

            if not model:
                raise HTTPException(status_code=400, detail="Missing 'model' field")

            model_name, model_path = manager.resolve_model_path(model)

            slot = manager.select_slot(model_name)
            if not slot:
                raise HTTPException(status_code=503, detail="No available slots")

            slot.load(model_name, model_path, ctx_size=ctx_size, extra_args=extra_args)

            return JSONResponse(
                {
                    "status": "loaded",
                    "slot_id": slot.id,
                    "port": slot.port,
                    "model": model_name,
                }
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/unload/{slot_id}")
    async def unload_slot(slot_id: int):
        """Unload model from specific slot."""
        slot = manager.get_slot(slot_id)
        if not slot:
            raise HTTPException(status_code=404, detail=f"Slot {slot_id} not found")

        slot.unload()
        return JSONResponse({"status": "unloaded", "slot_id": slot_id})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Forward chat completion request to appropriate slot."""
        try:
            body = await request.json()
            model = body.get("model", "default")

            slot = manager.select_slot(model)
            if not slot:
                raise HTTPException(status_code=503, detail="No available slots")

            if not slot.is_loaded(model):
                model_name, model_path = manager.resolve_model_path(model)
                slot.load(model_name, model_path)

            slot.busy = True
            try:
                async with httpx.AsyncClient(timeout=600.0) as client:
                    response = await client.post(
                        f"http://127.0.0.1:{slot.port}/v1/chat/completions",
                        json=body,
                    )
                    return JSONResponse(
                        content=response.json(), status_code=response.status_code
                    )
            finally:
                slot.busy = False

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def run_slot_serve(
    host: str = "127.0.0.1",
    port: int = 7000,
    base_port: int = 8000,
    model: str | None = None,
    ctx_size: int | None = None,
) -> None:
    """Run slot-based server with NUMA awareness."""
    import uvicorn

    manager = SlotManager(base_port=base_port)

    print(f"Initialized {len(manager.slots)} slots:")
    for slot in manager.slots:
        print(f"  Slot {slot.id}: socket={slot.socket_id}, port={slot.port}")

    if model:
        print(f"\nPre-loading model '{model}'...")
        try:
            model_name, model_path = manager.resolve_model_path(model)
            slot = manager.select_slot(model_name)
            if slot:
                slot.load(model_name, model_path, ctx_size=ctx_size)
                print(f"  Loaded on slot {slot.id} (port {slot.port})")
            else:
                print("  Warning: No available slots")
        except Exception as e:
            print(f"  Failed to load model: {e}")

    def shutdown_handler(signum, frame):
        print("\nShutting down slots...")
        manager.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    app = create_slot_app(manager)

    print(f"\nSlot server listening on {host}:{port}")
    print(f"Management API:")
    print(f"  GET  {host}:{port}/slots - List slots")
    print(f"  POST {host}:{port}/load - Load model")
    print(f"  POST {host}:{port}/unload/<slot_id> - Unload slot")
    print(f"\nInference API:")
    print(f"  POST {host}:{port}/v1/chat/completions")

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        manager.shutdown()
