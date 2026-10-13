"""Column schemas for v0.2 CSV. not implemented in v0.1.

No writer, no unload candidate, and no bucket-count or duration formula.
Empty fields stay unknown.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

InferenceStatus = Literal["candidate", "inferred", "unknown"]
ReviewStatus = Literal["unreviewed", "confirmed", "corrected", "rejected"]


class EventRow(BaseModel):
    """§17.4 `events.csv` columns. not implemented in v0.1."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    run_id: str
    video_id: str
    view_id: str
    excavator_id: str
    truck_visit_id_nullable: str | None = None
    event_type: str | None = None
    start_s: float | None = None
    end_s: float | None = None
    anchor_s: float | None = None
    inference_status: InferenceStatus | None = None
    rule_score_nullable: float | None = None
    review_status: ReviewStatus | None = None
    evidence_observation_ids: list[str] | None = None
    left_censored: bool | None = None
    right_censored: bool | None = None
    note: str | None = None


class TruckVisitRow(BaseModel):
    """§17.5 `truck_visits.csv` columns. not implemented in v0.1."""

    model_config = ConfigDict(extra="forbid")

    truck_visit_id: str
    video_id: str
    excavator_id_nullable: str | None = None
    linked_track_ids: list[str] | None = None
    physical_vehicle_id_nullable: str | None = None
    first_seen_s: float | None = None
    in_position_s_nullable: float | None = None
    loading_start_s_nullable: float | None = None
    loading_end_s_nullable: float | None = None
    departure_s_nullable: float | None = None
    last_seen_s: float | None = None
    confirmed_bucket_count: int | None = None
    estimated_bucket_count: int | None = None
    left_censored: bool | None = None
    right_censored: bool | None = None
    identity_uncertain: bool | None = None
    review_status: ReviewStatus | None = None
