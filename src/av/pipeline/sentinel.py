"""Sentinel — Video intelligence agent for surveillance events.

Detects: FALL, LONG_QUEUE, CROWD_GATHERING, WHEELCHAIR_COMPLIANCE.
Runs on any VLM provider: Gemini (cloud, quick-start) or ollama (local, free).

Architecture: Perceive (VLM) → Remember (state) → Reason (rules) → Alert

Based on 107 autoresearch experiments across 21 vision models.
Key finding: structural extraction + temporal rules > generic anomaly prompts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Prompts — validated across 107 experiments
# ---------------------------------------------------------------------------

# V04: Multi-alert prompt for Gemma, Qwen, GPT models
# Produces structured JSON with person positions, queue, crowd, wheelchair
V04_PROMPT = """\
Analyze {n_frames} sequential surveillance frames.

OBSERVE and describe:

1. PEOPLE: For each person visible:
   - Body position per frame (standing/walking/sitting/lying/crouching/running)
   - Key transition (e.g. standing→lying)
   - Mobility aid (wheelchair, crutch, walker) — yes/no
   - Carrying items (bag, luggage, etc.)

2. FORMATIONS:
   - Are people in a LINE/QUEUE? (sequential, facing same direction, stationary)
   - Queue length estimate (number of people)
   - Are people in a CLUSTER/CROWD? (grouped, not flowing)
   - Crowd density (sparse/moderate/dense)

3. SCENE:
   - Total person count
   - Notable objects (wheelchair, luggage, barriers, signs)
   - Path clearance (is walkway blocked or clear?)

Output JSON:
{{"scene": "", "person_count": 0,
 "persons": [
   {{"id": "P1", "positions": ["standing"], "transition": "none",
    "mobility_aid": null, "carrying": null, "action": ""}}
 ],
 "queue": {{"detected": false, "length": 0, "direction": "", "wait_area": ""}},
 "crowd": {{"detected": false, "density": "sparse", "stationary_pct": 0}},
 "wheelchair": {{"detected": false, "person_id": null, "attended": false,
                 "path_clear": true}},
 "events": [{{"start": 0, "end": {n_frames}, "text": ""}}],
 "alerts": []}}

Rules:
- FALL alert ONLY if someone transitions from upright to lying/ground
- Bending or crouching is NOT a fall
- Empty alerts is the EXPECTED default
- Describe what you SEE, don't interpret"""

# V03+: Focused fall detection for Mistral (experiment 105: V04 hurts Mistral)
V03_PROMPT = """\
Analyze {n_frames} sequential surveillance frames.

For each person visible:
- Body position in EACH frame (standing/walking/sitting/lying/crouching/running)
- Key transition: most significant position change (e.g. standing to lying)
- FALL alert ONLY if someone transitions from upright to lying/ground
- Bending or crouching is NOT a fall

Output JSON:
{{"scene": "", "person_count": 0,
 "persons": [{{"id": "P1", "clothing": "", "positions": ["standing"], "transition": "none", "action_summary": ""}}],
 "formation": "scattered",
 "events": [{{"start": 0, "end": {n_frames}, "text": ""}}],
 "alerts": []}}

Empty alerts is the EXPECTED default for normal activity."""


def get_prompt_for_model(model: str, n_frames: int) -> str:
    """Select best prompt for this model (validated in experiment 105).

    Mistral 3.2 works best with V03+ (focused, F1=0.944).
    All other models work best with V04 (multi-alert, F1=0.857-0.867).
    """
    if "mistral" in model.lower():
        return V03_PROMPT.format(n_frames=n_frames)
    return V04_PROMPT.format(n_frames=n_frames)


# ---------------------------------------------------------------------------
# Alert data type
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    type: str          # FALL, LONG_QUEUE, CROWD_GATHERING, WHEELCHAIR_COMPLIANCE
    confidence: float
    severity: str      # CRITICAL, HIGH, MEDIUM, LOW
    evidence: str
    camera_id: str = ""
    person_id: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "type": self.type, "confidence": self.confidence,
            "severity": self.severity, "evidence": self.evidence,
            "camera_id": self.camera_id, "person_id": self.person_id,
            "timestamp": self.timestamp,
        }.items() if v}


# ---------------------------------------------------------------------------
# Camera State (temporal memory across chunks)
# ---------------------------------------------------------------------------

@dataclass
class CameraState:
    """Rolling state for one camera across chunks."""
    camera_id: str
    chunk_count: int = 0
    person_counts: list[int] = field(default_factory=list)
    queue_detected_streak: int = 0
    queue_lengths: list[int] = field(default_factory=list)
    queue_first_seen: float | None = None
    crowd_detected_streak: int = 0
    crowd_first_seen: float | None = None
    wheelchair_detected_at: float | None = None
    wheelchair_attended: bool = False
    wheelchair_wait_sec: float = 0
    last_alert: dict[str, float] = field(default_factory=dict)

    def update(self, obs: dict, timestamp: float):
        self.chunk_count += 1
        pc = obs.get("person_count", 0)
        if isinstance(pc, str):
            m = re.search(r"\d+", pc)
            pc = int(m.group()) if m else 0
        self.person_counts.append(pc)
        if len(self.person_counts) > 20:
            self.person_counts.pop(0)

        q = obs.get("queue", {})
        if q.get("detected"):
            self.queue_detected_streak += 1
            self.queue_lengths.append(q.get("length", 0))
            if len(self.queue_lengths) > 20:
                self.queue_lengths.pop(0)
            if self.queue_first_seen is None:
                self.queue_first_seen = timestamp
        else:
            self.queue_detected_streak = 0
            self.queue_lengths.clear()
            self.queue_first_seen = None

        c = obs.get("crowd", {})
        if c.get("detected"):
            self.crowd_detected_streak += 1
            if self.crowd_first_seen is None:
                self.crowd_first_seen = timestamp
        else:
            self.crowd_detected_streak = 0
            self.crowd_first_seen = None

        w = obs.get("wheelchair", {})
        if w.get("detected"):
            if self.wheelchair_detected_at is None:
                self.wheelchair_detected_at = timestamp
            self.wheelchair_attended = w.get("attended", False)
            self.wheelchair_wait_sec = timestamp - self.wheelchair_detected_at
        else:
            self.wheelchair_detected_at = None
            self.wheelchair_wait_sec = 0

    def cooldown_ok(self, alert_type: str, cooldown_sec: float = 60) -> bool:
        return (time.time() - self.last_alert.get(alert_type, 0)) >= cooldown_sec


# ---------------------------------------------------------------------------
# Detection rules (one per alert type)
# ---------------------------------------------------------------------------

def check_fall(obs: dict, state: CameraState) -> list[Alert]:
    """Position transition detection. F1=0.944 on 130 Le2i videos."""
    upright = {"standing", "walking", "running"}
    down = {"lying", "fallen", "collapsed", "on the ground", "lying down"}
    alerts = []
    for person in obs.get("persons", []):
        positions = [p.lower().strip() for p in person.get("positions", [])]
        for i in range(1, len(positions)):
            if positions[i - 1] in upright and positions[i] in down:
                alerts.append(Alert(
                    type="FALL", person_id=person.get("id", "?"),
                    evidence=f"{positions[i-1]}→{positions[i]}",
                    confidence=0.9, severity="CRITICAL",
                    camera_id=state.camera_id,
                ))
                break
        if not any(a.person_id == person.get("id") for a in alerts):
            action = person.get("action", "") or ""
            transition = person.get("transition", "") or ""
            if isinstance(action, list): action = " ".join(str(a) for a in action)
            if isinstance(transition, list): transition = " ".join(str(t) for t in transition)
            for text in [action.lower(), transition.lower()]:
                if any(w in text for w in ["fall", "fell", "collapse", "lying on"]):
                    alerts.append(Alert(
                        type="FALL", person_id=person.get("id", "?"),
                        evidence=f"text: {text[:60]}",
                        confidence=0.8, severity="CRITICAL",
                        camera_id=state.camera_id,
                    ))
                    break
    for a in obs.get("alerts", []):
        if not isinstance(a, dict):
            if isinstance(a, str) and "fall" in a.lower() and not alerts:
                alerts.append(Alert(type="FALL", evidence="vlm-alert",
                                   confidence=0.7, severity="CRITICAL",
                                   camera_id=state.camera_id))
            continue
        if a.get("type", "").upper() == "FALL" and not alerts:
            alerts.append(Alert(type="FALL", evidence="vlm-detected",
                               confidence=float(a.get("confidence", 0.8)),
                               severity="CRITICAL", camera_id=state.camera_id))
    return alerts


def check_long_queue(obs: dict, state: CameraState,
                     min_streak: int = 3, min_length: int = 5) -> list[Alert]:
    """Queue must persist across 3+ consecutive chunks (90s at 30s chunks)."""
    if state.queue_detected_streak < min_streak:
        return []
    recent = state.queue_lengths[-min_streak:]
    if not recent:
        return []
    avg_length = sum(recent) / len(recent)
    if avg_length < min_length:
        return []
    wait = time.time() - state.queue_first_seen if state.queue_first_seen else 0
    return [Alert(
        type="LONG_QUEUE",
        evidence=f"Queue ~{avg_length:.0f} people for {wait:.0f}s ({state.queue_detected_streak} chunks)",
        confidence=min(0.95, 0.6 + state.queue_detected_streak * 0.1),
        severity="HIGH" if avg_length >= 10 else "MEDIUM",
        camera_id=state.camera_id,
    )]


def check_crowd_gathering(obs: dict, state: CameraState,
                          count_threshold: int = 15) -> list[Alert]:
    """Sustained dense crowd (3+ chunks) OR rapid person count growth."""
    pc = obs.get("person_count", 0)
    if isinstance(pc, str):
        m = re.search(r"\d+", pc)
        pc = int(m.group()) if m else 0
    if state.crowd_detected_streak >= 3:
        density = obs.get("crowd", {}).get("density", "sparse")
        if density in ("moderate", "dense") and pc >= count_threshold:
            return [Alert(
                type="CROWD_GATHERING",
                evidence=f"{pc} people, {density} density, {state.crowd_detected_streak} chunks",
                confidence=0.8 if density == "dense" else 0.7,
                severity="HIGH" if density == "dense" else "MEDIUM",
                camera_id=state.camera_id,
            )]
    if len(state.person_counts) >= 4:
        prev = sum(state.person_counts[-4:-1]) / 3
        if prev > 3 and pc / prev >= 1.5 and pc >= count_threshold:
            return [Alert(
                type="CROWD_GATHERING",
                evidence=f"Rapid growth: {prev:.0f}→{pc} ({pc/prev:.1f}x)",
                confidence=0.7, severity="HIGH",
                camera_id=state.camera_id,
            )]
    return []


def check_wheelchair_compliance(obs: dict, state: CameraState,
                                max_wait_sec: int = 120) -> list[Alert]:
    """Alert if wheelchair user unattended too long or path blocked."""
    w = obs.get("wheelchair", {})
    if not w.get("detected"):
        return []
    alerts = []
    if not state.wheelchair_attended and state.wheelchair_wait_sec > max_wait_sec:
        alerts.append(Alert(
            type="WHEELCHAIR_COMPLIANCE",
            evidence=f"Wheelchair user unattended {state.wheelchair_wait_sec:.0f}s (limit: {max_wait_sec}s)",
            confidence=0.8, severity="CRITICAL",
            camera_id=state.camera_id,
        ))
    if not w.get("path_clear", True):
        alerts.append(Alert(
            type="WHEELCHAIR_COMPLIANCE",
            evidence="Wheelchair path blocked",
            confidence=0.7, severity="HIGH",
            camera_id=state.camera_id,
        ))
    return alerts


ALERT_RULES = {
    "FALL": check_fall,
    "LONG_QUEUE": check_long_queue,
    "CROWD_GATHERING": check_crowd_gathering,
    "WHEELCHAIR_COMPLIANCE": check_wheelchair_compliance,
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SentinelAgent:
    """Video intelligence agent with temporal state.

    Usage:
        agent = SentinelAgent()
        for chunk_obs in process_video(video):
            alerts = agent.process("cam1", chunk_obs)
            if alerts:
                handle_alerts(alerts)
    """

    def __init__(
        self,
        alert_types: list[str] | None = None,
        cooldowns: dict[str, float] | None = None,
    ):
        self.alert_types = alert_types or list(ALERT_RULES.keys())
        self.cooldowns = cooldowns or {
            "FALL": 30, "LONG_QUEUE": 120,
            "CROWD_GATHERING": 120, "WHEELCHAIR_COMPLIANCE": 60,
        }
        self._states: dict[str, CameraState] = {}

    def get_state(self, camera_id: str) -> CameraState:
        if camera_id not in self._states:
            self._states[camera_id] = CameraState(camera_id=camera_id)
        return self._states[camera_id]

    def process(self, camera_id: str, observation: dict,
                timestamp: float | None = None) -> list[Alert]:
        ts = timestamp or time.time()
        state = self.get_state(camera_id)
        state.update(observation, ts)

        all_alerts = []
        for alert_type in self.alert_types:
            rule = ALERT_RULES.get(alert_type)
            if not rule:
                continue
            if not state.cooldown_ok(alert_type, self.cooldowns.get(alert_type, 60)):
                continue
            alerts = rule(observation, state)
            for alert in alerts:
                alert.timestamp = ts
                all_alerts.append(alert)
                state.last_alert[alert_type] = time.time()
        return all_alerts

    def reset(self, camera_id: str | None = None):
        if camera_id:
            self._states.pop(camera_id, None)
        else:
            self._states.clear()


def parse_vlm_response(raw: str) -> dict:
    """Parse VLM JSON response, handling markdown fences and trailing commas."""
    if not raw:
        return {}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", m.group(0)))
        try:
            return json.loads(fixed)
        except Exception:
            return {}
