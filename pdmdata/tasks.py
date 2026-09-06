"""Canonical predictive-maintenance task taxonomy."""

from __future__ import annotations

from typing import Dict, List


TASKS: List[Dict[str, str]] = [
    {
        "id": "anomaly_detection",
        "short_name": "Anomaly",
        "name": "Anomaly detection",
        "description": "Detect observations or sequences that depart from normal operation.",
    },
    {
        "id": "fault_classification",
        "short_name": "Fault class.",
        "name": "Fault or health-state classification",
        "description": "Assign a discrete fault type, component state, or healthy/faulty label.",
    },
    {
        "id": "operating_state_classification",
        "short_name": "State class.",
        "name": "Operating-state classification",
        "description": "Identify operating modes or machine states that are not themselves faults.",
    },
    {
        "id": "condition_estimation",
        "short_name": "Condition",
        "name": "Condition estimation",
        "description": "Estimate a continuous health indicator, degradation level, or component condition.",
    },
    {
        "id": "rul_prediction",
        "short_name": "RUL",
        "name": "Remaining useful life prediction",
        "description": "Predict remaining cycles or time before a defined failure endpoint.",
    },
    {
        "id": "time_to_event",
        "short_name": "TTE",
        "name": "Time-to-event prediction",
        "description": "Predict when a failure, alarm, or maintenance-relevant event will occur.",
    },
    {
        "id": "survival_analysis",
        "short_name": "Survival",
        "name": "Survival analysis",
        "description": "Model event-time distributions while explicitly accounting for censoring.",
    },
    {
        "id": "event_forecasting",
        "short_name": "Event forecast",
        "name": "Event or sequence forecasting",
        "description": "Predict the type or sequence of future alarms, errors, or machine events.",
    },
    {
        "id": "maintenance_policy",
        "short_name": "Maintenance",
        "name": "Maintenance-policy evaluation",
        "description": "Compare or learn maintenance decisions using intervention and outcome histories.",
    },
]

TASK_BY_ID = {task["id"]: task for task in TASKS}
SUPPORT_LEVELS = {"direct", "derived"}

