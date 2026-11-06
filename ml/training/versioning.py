"""
LinguaLabel — Model Versioning, Registry & Lifecycle Management (Part 11.3)

Maintains complete version tracking and lifecycle state for all trained models:
- Lifecycle states: 'candidate', 'production', 'archived', 'rejected', 'rolled_back'
- Tracks training datasets, parameters, timestamps, evaluation metrics, and provenance.
- Atomic production promotion and rollback mechanisms.
- Maintains an append-only audit log for all promotion/rollback operations.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional


@dataclass
class ModelVersion:
    version_id: str
    task: str  # 'sentiment', 'text_type', 'language', 'abuse_category'
    model_name: str
    architecture: str
    status: str  # 'candidate', 'production', 'archived', 'rejected', 'rolled_back'
    dataset_name: str
    dataset_fingerprint: str
    created_at: str  # ISO 8601 UTC
    promoted_at: Optional[str] = None
    training_params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts_dir: str = ""
    notes: str = ""

    @property
    def lifecycle_state(self) -> str:
        return self.status

    @property
    def artifacts_path(self) -> str:
        return self.artifacts_dir

    @property
    def training_config(self) -> Dict[str, Any]:
        return self.training_params

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


TASK_COMPATIBILITY_GROUPS = {
    "language": {"language"},
    "sentiment": {"sentiment"},
    "text_type": {"text_type"},
    "complaint": {"complaint", "complaining"},
    "complaining": {"complaint", "complaining"},
    "abuse_category": {"abuse_category", "abuse"},
    "abuse": {"abuse_category", "abuse"},
    "abuse_severity": {"abuse_severity", "severity"},
    "severity": {"abuse_severity", "severity"},
}


class ModelVersionRegistry:
    """
    Manages cataloging, promotion, rollback, and auditing of trained model versions.
    """

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent.parent
        self.models_dir = self.root_dir / "trained_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / "model_registry.json"
        self.active_file = self.models_dir / "active_production.json"
        self.audit_log_file = self.models_dir / "lifecycle_audit.jsonl"
        self._ensure_files()

    def _ensure_files(self):
        if not self.registry_file.exists():
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)
        if not self.active_file.exists():
            with open(self.active_file, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_registry(self, data: Dict[str, Dict[str, Any]]):
        temp_file = self.models_dir / "model_registry.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.registry_file)

    def _load_active(self) -> Dict[str, str]:
        try:
            with open(self.active_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_active(self, data: Dict[str, str]):
        temp_file = self.models_dir / "active_production.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.active_file)

    def _log_audit_event(self, action: str, version_id: str, task: str, details: Dict[str, Any]):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "version_id": version_id,
            "task": task,
            "details": details,
        }
        with open(self.audit_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def register_candidate(
        self,
        task: str,
        model_name: str,
        architecture: str,
        dataset_name: str,
        dataset_fingerprint: str,
        artifacts_dir: Path,
        training_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
        version_id: Optional[str] = None,
    ) -> ModelVersion:
        """Registers a newly trained candidate model."""
        ts = datetime.now(timezone.utc)
        safe_task = task.replace(" ", "_")
        effective_vid = version_id or f"{safe_task}_{ts.strftime('%Y%m%d_%H%M%S')}"

        model_ver = ModelVersion(
            version_id=effective_vid,
            task=task,
            model_name=model_name,
            architecture=architecture,
            status="candidate",
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            created_at=ts.isoformat(),
            training_params=training_params or {},
            metrics=metrics or {},
            artifacts_dir=str(artifacts_dir),
            notes=notes,
        )

        reg = self._load_registry()
        reg[effective_vid] = model_ver.to_dict()
        self._save_registry(reg)

        self._log_audit_event(
            action="register_candidate",
            version_id=effective_vid,
            task=task,
            details={"architecture": architecture, "dataset": dataset_name},
        )
        return model_ver

    def get_model(self, version_id: str) -> Optional[ModelVersion]:
        reg = self._load_registry()
        data = reg.get(version_id)
        if not data:
            return None
        return ModelVersion(**data)

    def list_models(self, task: Optional[str] = None) -> List[ModelVersion]:
        reg = self._load_registry()
        models = [ModelVersion(**d) for d in reg.values()]
        if task:
            models = [m for m in models if m.task == task]
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models

    def get_active_production_model(self, task: str) -> Optional[ModelVersion]:
        active_map = self._load_active()
        active_vid = active_map.get(task)
        if not active_vid:
            return None
        return self.get_model(active_vid)

    def promote_to_production(
        self,
        version_id: str,
        target_task: Optional[str] = None,
        approved_by: str = "owner",
        actor: Optional[str] = None,
        notes: str = "",
    ) -> bool:
        """
        Promotes a candidate model to active production status with strict task compatibility checks.
        Archives any existing production model for this task and records audit log.
        """
        effective_actor = actor or approved_by
        model = self.get_model(version_id)
        if not model:
            return False

        if not Path(model.artifacts_dir).exists():
            return False

        # Task compatibility verification
        slot_task = target_task or model.task
        allowed_tasks = TASK_COMPATIBILITY_GROUPS.get(slot_task, {slot_task})
        if model.task not in allowed_tasks:
            # Reject task mismatches (e.g. promoting severity model to category slot or vice versa)
            return False

        # Class compatibility check if meta.json exists
        meta_file = Path(model.artifacts_dir) / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                classes = set(meta.get("classes", []))
                if classes:
                    if slot_task in ("abuse_severity", "severity"):
                        # Must NOT contain category labels like 'Threat', 'Sexualized Abuse'
                        if any(c in {"Threat", "Sexualized Abuse", "Severe Personal Insult", "Harassment", "Profanity / Vulgarity"} for c in classes):
                            return False
                    elif slot_task in ("abuse_category", "abuse"):
                        # Must NOT contain severity labels alone like 'Mild', 'Moderate', 'Severe'
                        if classes.issubset({"None", "Mild", "Moderate", "Severe"}):
                            return False
            except Exception:
                pass

        reg = self._load_registry()
        active_map = self._load_active()

        # Archive prior production model if present
        prior_vid = active_map.get(slot_task)
        if prior_vid and prior_vid in reg and prior_vid != version_id:
            reg[prior_vid]["status"] = "archived"

        # Update candidate to production
        reg[version_id]["status"] = "production"
        reg[version_id]["promoted_at"] = datetime.now(timezone.utc).isoformat()
        active_map[slot_task] = version_id
        # Also sync alias slot if applicable
        if slot_task in ("abuse_category", "abuse"):
            active_map["abuse_category"] = version_id
            active_map["abuse"] = version_id
        elif slot_task in ("abuse_severity", "severity"):
            active_map["abuse_severity"] = version_id
            active_map["severity"] = version_id
        elif slot_task in ("complaint", "complaining"):
            active_map["complaint"] = version_id
            active_map["complaining"] = version_id

        self._save_registry(reg)
        self._save_active(active_map)

        self._log_audit_event(
            action="promote",
            version_id=version_id,
            task=slot_task,
            details={"approved_by": effective_actor, "actor": effective_actor, "notes": notes, "prior_production": prior_vid},
        )
        return True

    def rollback_production(
        self,
        task: str,
        target_version_id: Optional[str] = None,
        reason: str = "manual_rollback",
        actor: Optional[str] = None,
        notes: str = "",
    ) -> bool:
        """
        Rolls back the active production model to a target version or default baseline.
        """
        effective_actor = actor or "owner"
        active_map = self._load_active()
        current_vid = active_map.get(task)
        reg = self._load_registry()

        if target_version_id:
            target_model = self.get_model(target_version_id)
            if not target_model:
                return False
            allowed = TASK_COMPATIBILITY_GROUPS.get(task, {task})
            if target_model.task not in allowed:
                return False

            if current_vid and current_vid in reg:
                reg[current_vid]["status"] = "rolled_back"

            reg[target_version_id]["status"] = "production"
            reg[target_version_id]["promoted_at"] = datetime.now(timezone.utc).isoformat()
            active_map[task] = target_version_id
            for t_alias in TASK_COMPATIBILITY_GROUPS.get(task, {task}):
                active_map[t_alias] = target_version_id

            self._save_registry(reg)
            self._save_active(active_map)

            self._log_audit_event(
                action="rollback",
                version_id=target_version_id,
                task=task,
                details={"reverted_from": current_vid, "reason": reason, "actor": effective_actor, "notes": notes},
            )
            return True
        else:
            # Revert to default baseline models
            if current_vid and current_vid in reg:
                reg[current_vid]["status"] = "rolled_back"
                self._save_registry(reg)

            for t_alias in TASK_COMPATIBILITY_GROUPS.get(task, {task}):
                active_map.pop(t_alias, None)
            self._save_active(active_map)

            self._log_audit_event(
                action="rollback",
                version_id="baseline",
                task=task,
                details={"reverted_from": current_vid, "reason": reason, "actor": effective_actor, "notes": notes},
            )
            return True

            self._log_audit_event(
                action="rollback",
                version_id="builtin_baseline",
                task=task,
                details={"reverted_from": current_vid, "reason": reason, "actor": effective_actor, "notes": notes},
            )
            return True

    get_version = get_model
    list_versions = list_models

    def get_active_production_models(self) -> Dict[str, str]:
        return self._load_active()

    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent audit events."""
        if not self.audit_log_file.exists():
            return []
        events = []
        with open(self.audit_log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
        events.reverse()
        return events[:limit]

    get_audit_trail = get_audit_log
