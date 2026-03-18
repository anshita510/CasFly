from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class EventIn(BaseModel):
    name: str = Field(..., min_length=1)
    timestamp: datetime


class CPTCountIn(BaseModel):
    cause: str = Field(..., min_length=1)
    effect: str = Field(..., min_length=1)
    lag_bin: str = Field(..., min_length=1)
    count: int = Field(..., ge=0)


class PLTRouteIn(BaseModel):
    event_name: str = Field(..., min_length=1)
    device_id: str = Field(..., min_length=1)
    probability: float = Field(..., ge=0.0, le=1.0)


class NodeInput(BaseModel):
    device_id: str = Field(..., min_length=1)
    events: list[EventIn] = Field(default_factory=list)


class TraceRequest(BaseModel):
    cpt_counts: list[CPTCountIn]
    routing_table: list[PLTRouteIn]
    nodes: list[NodeInput]
    start_device: str = Field(..., min_length=1)
    start_event: str = Field(..., min_length=1)
    max_hops: int = Field(default=16, ge=1, le=128)
    max_depth: int = Field(default=10, ge=1, le=128)
    fallback_depth: int = Field(default=3, ge=0, le=32)


class HopOut(BaseModel):
    source_device: str
    target_device: str | None
    trigger_event: str
    best_path: list[str]
    path_probability: float
    cumulative_lag_days: float


class TraceResponse(BaseModel):
    visited_devices: list[str]
    visited_edges: list[tuple[str, str]]
    chain_confidence: float
    chain_raw_confidence: float
    chain_lag_weight_product: float
    theorem_lower_bound: float
    hops: list[HopOut]
