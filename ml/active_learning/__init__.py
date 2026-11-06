"""
LinguaLabel Active Learning & Intelligent Annotation Engine Package.

Provides:
- Uncertainty Sampling (lowest confidence, prediction margin, normalized Shannon entropy)
- Model Disagreement Sampling (baseline vs transformer conflict detection)
- Intelligent Priority Scoring & Queue Orchestration
- Isolated Human Feedback Collection & Analytics
- Controlled Dataset Preparation & Partitioning
"""

from ml.active_learning.uncertainty import (
    TaskUncertainty,
    RecordUncertainty,
    compute_task_uncertainty,
    compute_record_uncertainty,
)
from ml.active_learning.disagreement import (
    TaskDisagreement,
    RecordDisagreement,
    evaluate_task_disagreement,
    compute_record_disagreement,
)
from ml.active_learning.priority import (
    PriorityWeights,
    PriorityRecord,
    compute_priority_record,
)
from ml.active_learning.queue import (
    ActiveLearningQueue,
    QueueSummary,
)
from ml.active_learning.feedback import (
    FeedbackRecord,
    FeedbackManager,
    compute_feedback_analytics,
)
from ml.active_learning.preparation import (
    DatasetSplit,
    compute_training_data_fingerprint,
    format_feedback_for_training,
    check_data_leakage,
    prepare_controlled_split,
)

__all__ = [
    "TaskUncertainty",
    "RecordUncertainty",
    "compute_task_uncertainty",
    "compute_record_uncertainty",
    "TaskDisagreement",
    "RecordDisagreement",
    "evaluate_task_disagreement",
    "compute_record_disagreement",
    "PriorityWeights",
    "PriorityRecord",
    "compute_priority_record",
    "ActiveLearningQueue",
    "QueueSummary",
    "FeedbackRecord",
    "FeedbackManager",
    "compute_feedback_analytics",
    "DatasetSplit",
    "compute_training_data_fingerprint",
    "format_feedback_for_training",
    "check_data_leakage",
    "prepare_controlled_split",
]

