"""Versioned bridge to the fraud pilot's ledger; no dependency on its repository."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    model_validator,
)
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from src.drift.detectors import DriftMonitorEngine

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


class PredictionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cursor: int = Field(ge=1)
    prediction_id: str = Field(min_length=1, max_length=128)
    transaction_id: str = Field(min_length=1, max_length=128)
    created_at: AwareDatetime
    model_version: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_version: str
    threshold: FiniteFloat = Field(ge=0, le=2)
    score: FiniteFloat = Field(ge=0, le=1)
    features: list[FiniteFloat] = Field(min_length=29, max_length=29)
    label: int | None = Field(default=None, ge=0, le=1, strict=True)
    label_available_at: AwareDatetime | None = None
    label_source: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_label(self):
        if self.label is None:
            if self.label_available_at is not None or self.label_source is not None:
                raise ValueError("Unlabeled events cannot claim label provenance.")
        elif self.label_available_at is None or not self.label_source:
            raise ValueError("Labels require availability timestamp and source.")
        elif self.label_available_at < self.created_at:
            raise ValueError("Label availability must not precede prediction receipt.")
        return self


class FraudWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = Field(pattern=r"^fraud-events-v1$")
    events: list[PredictionEvent] = Field(min_length=25, max_length=5000)
    as_of: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    minimum_label_age_seconds: int = Field(default=0, ge=0, le=31_536_000)
    next_cursor: int | None = Field(default=None, ge=0)
    label_refresh: str | None = None


class FraudReference:
    def __init__(self, path: Path):
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Reference file exceeds 10 MiB.")
        raw = path.read_bytes()
        if len(raw) > 10 * 1024 * 1024:
            raise ValueError("Reference file exceeds 10 MiB.")
        snapshot = json.loads(raw)
        if (
            snapshot["schema_version"] != "fraud-reference-v1"
            or snapshot["feature_version"] != "ulb-pca-v1"
        ):
            raise ValueError("Unsupported reference schema.")
        if snapshot["feature_names"] != FEATURE_NAMES:
            raise ValueError(
                "Reference feature order does not match ULB serving contract."
            )
        rows, scores = snapshot["records"], snapshot["scores"]
        if not 25 <= len(rows) <= 5000 or len(scores) != len(rows):
            raise ValueError(
                "Reference must contain 25–5000 aligned feature/score rows."
            )
        # Reuse the strict finite feature/score validators for every baseline row.
        for row, score in zip(rows, scores):
            if len(row) != 29 or not all(
                isinstance(v, (int, float)) and abs(v) < float("inf") for v in row
            ):
                raise ValueError("Invalid reference features.")
            if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                raise ValueError("Invalid reference score.")
        self.version = hashlib.sha256(raw).hexdigest()
        self.model_version = snapshot["model_version"]
        self.threshold = float(snapshot["threshold"])
        if (
            not isinstance(self.model_version, str)
            or len(self.model_version) != 64
            or any(c not in "0123456789abcdef" for c in self.model_version)
        ):
            raise ValueError("Invalid reference model identity.")
        if not 0 <= self.threshold <= 2:
            raise ValueError("Invalid reference decision threshold.")
        self.source = snapshot["source"]
        reference_df = pd.DataFrame(rows, columns=FEATURE_NAMES)
        reference_df["model_score"] = scores
        self.engine = DriftMonitorEngine()
        self.engine.set_reference(reference_df)

    def evaluate(self, window: FraudWindow):
        events = window.events
        if window.as_of > datetime.now(UTC):
            raise ValueError("as_of must not be in the future.")
        if len({event.prediction_id for event in events}) != len(events) or len(
            {event.transaction_id for event in events}
        ) != len(events):
            raise ValueError(
                "A window must contain unique prediction and transaction IDs."
            )
        for event in events:
            if event.created_at > window.as_of:
                raise ValueError("Prediction receipt is after the evaluation cutoff.")
            if (
                event.model_version != self.model_version
                or event.feature_version != "ulb-pca-v1"
            ):
                raise ValueError(
                    "Window model/feature versions do not match the reference."
                )
            if event.threshold != self.threshold:
                raise ValueError(
                    "Mixed or changed decision policies require a separate reference."
                )
        frame = pd.DataFrame(
            [event.features for event in events], columns=FEATURE_NAMES
        )
        frame["model_score"] = [event.score for event in events]
        drift = self.engine.evaluate_batch(frame)
        mature = [
            event
            for event in events
            if event.label is not None
            and event.label_available_at <= window.as_of
            and (window.as_of - event.created_at).total_seconds()
            >= window.minimum_label_age_seconds
        ]
        performance = {
            "status": "insufficient_mature_labels",
            "labeled_rows": len(mature),
            "coverage": len(mature) / len(events),
            "minimum_rows": 25,
        }
        if len(mature) >= 25:
            labels = [event.label for event in mature]
            scores = [event.score for event in mature]
            predictions = [event.score >= event.threshold for event in mature]
            tn, fp, fn, tp = (
                int(value)
                for value in confusion_matrix(
                    labels, predictions, labels=[0, 1]
                ).ravel()
            )
            both = len(set(labels)) == 2
            performance.update(
                status="evaluated",
                positive_labels=sum(labels),
                negative_labels=len(labels) - sum(labels),
                average_precision=float(average_precision_score(labels, scores))
                if both
                else None,
                roc_auc=float(roc_auc_score(labels, scores)) if both else None,
                uncalibrated_score_brier=float(brier_score_loss(labels, scores)),
                confusion_matrix={
                    "true_negative": tn,
                    "false_positive": fp,
                    "false_negative": fn,
                    "true_positive": tp,
                },
                illustrative_cost_units=20 * fn + fp,
                ranking_metrics_status="available"
                if both
                else "undefined_single_class",
            )
        return {
            "reference_version": self.version,
            "model_version": self.model_version,
            "reference_source": self.source,
            "as_of": window.as_of.isoformat(),
            "minimum_label_age_seconds": window.minimum_label_age_seconds,
            "rows": len(events),
            "distribution_diagnostics": drift,
            "label_performance": performance,
            "limitations": [
                "Drift flags are exploratory and do not establish model failure or lead time.",
                "Labeled-cohort metrics may be biased by missing/selective labels; coverage is not representativeness.",
                "The 20:1 cost ratio is illustrative; no real payment or INR outcome is inferred.",
                "created_at is service receipt time, not the historical dataset event timestamp.",
            ],
        }
