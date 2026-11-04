"""Pydantic schemas for the structured report the LLM must return."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

DefectKind = Literal["crack", "corrosion", "dent", "rivet_damage"]
Risk = Literal["LOW", "MEDIUM", "HIGH"]
Readiness = Literal["NO_GO", "CONDITIONAL", "MONITOR", "NO_ACTION"]


class Finding(BaseModel):
    defect_id: str = Field(pattern=r"^D\d+$", description="id from the record, e.g. D1")
    type: DefectKind
    observation: str = Field(min_length=5, description="what was detected, using only record values")
    recommended_action: str = Field(min_length=5)
    cited_chunks: List[str] = Field(default_factory=list, description="ids of retrieved guidance chunks")


class InspectionReport(BaseModel):
    summary: str = Field(min_length=10)
    overall_risk: Risk
    flight_recommendation: Readiness
    findings: List[Finding] = Field(default_factory=list)
    engine_note: Optional[str] = None
    progression_note: Optional[str] = None
    inspection_interval: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)

    @field_validator("overall_risk", mode="before")
    @classmethod
    def _upper_risk(cls, v):
        return v.upper() if isinstance(v, str) else v

    @field_validator("flight_recommendation", mode="before")
    @classmethod
    def _upper_ready(cls, v):
        return v.upper().replace(" ", "_") if isinstance(v, str) else v


class VisionOpinion(BaseModel):
    defect_present: bool
    defect_type: Optional[Literal["crack", "corrosion", "dent", "rivet_damage", "other", "none"]] = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


VisionScanDefectType = Literal["crack", "corrosion", "dent", "rivet_damage", "other"]
VisionSceneType = Literal["closeup_panel", "wide_aircraft", "other_object", "unclear"]
VisionConcern = Literal["none", "low", "medium", "high"]


class VisionScanDefect(BaseModel):
    type: VisionScanDefectType
    region: str
    confidence: float = Field(ge=0.0, le=1.0)
    note: str

    @field_validator("region")
    @classmethod
    def _valid_region(cls, value):
        regions = {
            "upper-left region", "upper-center region", "upper-right region",
            "center-left region", "center region", "center-right region",
            "lower-left region", "lower-center region", "lower-right region",
        }
        if value not in regions:
            raise ValueError("region must be a 3x3 image-grid label")
        return value


class VisionScanReport(BaseModel):
    scene_type: VisionSceneType
    summary: str
    defects: List[VisionScanDefect] = Field(default_factory=list)
    overall_concern: VisionConcern
    needs_human_review: bool
