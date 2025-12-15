"""Per-device CasFly FastAPI service.

Each physical device (wearable, bedside monitor, Raspberry Pi, etc.) runs
its own instance of this service independently.  Raw patient data never
leaves the device — causal chain packets are forwarded peer-to-peer via
UDP between devices following the CasFly distributed protocol.

Launch one instance per device::

    DEVICE_ID=BPCuff \\
    LOOKUP_CSV=data/All_Device_Lookup_with_Probabilities.csv \\
    TPHG_DIR=data/tphg_cache/BPCuff \\
    FILTERED_PATIENTS_CSV=data/filtered_patients.csv \\
    DEVICE_IP=192.168.1.42 \\
    DEVICE_PORT=5020 \\
    uvicorn casfly_service.app:app --host 0.0.0.0 --port 8020

Typical multi-device deployment::

    Smartwatch  :8001   CasFlyHub   :8002   BPCuff     :8003
    ──────────────────────────────────────────────────────────
    CasFlyDevice        CasFlyDevice         CasFlyDevice
         │  <──── UDP chain packets ────>  │  <──── UDP ────> │

The HTTP layer is for management only (start, initiate, status-check).
The causal chain itself flows directly between devices over UDP — no
central server ever sees the aggregated raw data.

Endpoints
---------
GET  /health     Liveness check.
POST /start      Initialise and bind the UDP port for this device.
POST /initiate   Trigger causal chain tracing from this device (initiator only).
GET  /status     Return device identity and port.

For in-process simulation (all devices in one Python process) use
``CasFlyOrchestrator`` directly — see ``examples/smartwatch_bp_cuff_demo.py``.
"""
from __future__ import annotations

import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from casfly_sdk import CasFlyDevice

app = FastAPI(
    title="CasFly Device Service",
    version="1.0.0",
    description=(
        "Per-device HTTP management wrapper around a CasFlyDevice UDP node. "
        "Deploy one instance per physical device. "
        "Raw patient data stays on-device; only causal chain packets are exchanged."
    ),
)

# ---------------------------------------------------------------------------
# Device singleton — one per process, initialised via POST /start
# ---------------------------------------------------------------------------

_device: Optional[CasFlyDevice] = None
_device_lock = threading.Lock()


def _get_device() -> CasFlyDevice:
    global _device
    if _device is None:
        raise HTTPException(status_code=503, detail="Device not started. Call POST /start first.")
    return _device


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    device_id: str = Field(..., description="Unique device name, e.g. 'BPCuff' or 'Smartwatch'")
    lookup_csv: str = Field(..., description="Path to All_Device_Lookup_with_Probabilities.csv")
    tphg_dir: str = Field(..., description="Directory containing <patient_id>_tphg.pkl files for this device")
    filtered_patients_csv: str = Field(..., description="Path to filtered_patients.csv")
    ip: str = Field(default="127.0.0.1", description="IP address this device binds its UDP socket to")
    port: int = Field(default=0, ge=0, le=65535, description="UDP port (0 = auto-assign)")
    metrics_dir: str = Field(default="./logs", description="Directory for per-device metric logs")
    max_depth: int = Field(default=10, ge=1, le=128, description="Max Viterbi expansion depth")


class InitiateRequest(BaseModel):
    event: str = Field(..., min_length=1, description="Trigger event name, e.g. 'Heart Attack'")
    patient_id: str = Field(..., min_length=1, description="Patient identifier to trace")


class DeviceStatusResponse(BaseModel):
    device_id: str
    ip: str
    port: int
    running: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness check")
def health() -> dict[str, str]:
    """Returns ``{"status": "ok"}`` when the service is up."""
    return {"status": "ok"}


@app.post(
    "/start",
    response_model=DeviceStatusResponse,
    summary="Initialise and start the UDP device",
)
def start_device(req: StartRequest) -> DeviceStatusResponse:
    """Create the ``CasFlyDevice``, bind its UDP port, and start the listener thread.

    Idempotent — calling ``/start`` again while a device is already running
    returns the current status without restarting.

    This endpoint must be called before ``/initiate``.  Typically called once
    at container / process startup.
    """
    global _device
    with _device_lock:
        if _device is not None:
            return DeviceStatusResponse(
                device_id=_device.device_id,
                ip=_device.ip,
                port=_device.port,
                running=True,
            )
        device = CasFlyDevice(
            device_id=req.device_id,
            lookup_csv=req.lookup_csv,
            tphg_dir=req.tphg_dir,
            filtered_patients_csv=req.filtered_patients_csv,
            ip=req.ip,
            port=req.port,
            metrics_dir=req.metrics_dir,
            max_depth=req.max_depth,
        )
        device.start()
        _device = device

    return DeviceStatusResponse(
        device_id=device.device_id,
        ip=device.ip,
        port=device.port,
        running=True,
    )


@app.post("/initiate", summary="Initiate causal chain tracing from this device")
def initiate_chain(req: InitiateRequest) -> dict[str, str]:
    """Trigger causal chain tracing for *event* / *patient_id* from this device.

    Only call this on the designated initiator device (the CasFlyHub).
    Once initiated, the chain is expanded locally and forwarded via UDP to
    subsequent devices automatically — no further HTTP calls are needed.

    The initiator device must have been started via ``POST /start`` first.
    """
    device = _get_device()
    threading.Thread(
        target=device.initiate_chain,
        args=(req.event, req.patient_id),
        daemon=True,
    ).start()
    return {
        "status": "initiated",
        "device_id": device.device_id,
        "event": req.event,
        "patient_id": req.patient_id,
    }


@app.get(
    "/status",
    response_model=DeviceStatusResponse,
    summary="Get device status",
)
def device_status() -> DeviceStatusResponse:
    """Return the current device identity, bound IP, and UDP port."""
    device = _get_device()
    return DeviceStatusResponse(
        device_id=device.device_id,
        ip=device.ip,
        port=device.port,
        running=True,
    )
