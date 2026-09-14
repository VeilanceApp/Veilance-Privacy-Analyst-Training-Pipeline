import gzip
import json
from pathlib import Path

SNAPSHOT_SCHEMA = "veilance.telemetry-snapshot.v2"
BATCH_SCHEMA = "veilance.telemetry-snapshot-batch.v1"
EXPECTED_SNAPSHOT_KEYS = {
    "schemaVersion", "eventId", "extensionVersion", "site", "observation",
    "thirdPartyHosts", "trackers", "signals", "page", "security", "interest",
    "redactedDocument"
}


def validate_snapshot_shape(snapshot):
    if not isinstance(snapshot, dict):
        raise ValueError("telemetry snapshot must be an object")
    if snapshot.get("schemaVersion") != SNAPSHOT_SCHEMA:
        raise ValueError(f"expected {SNAPSHOT_SCHEMA}, got {snapshot.get('schemaVersion')!r}")
    missing = EXPECTED_SNAPSHOT_KEYS - set(snapshot)
    extra = set(snapshot) - EXPECTED_SNAPSHOT_KEYS
    if missing:
        raise ValueError(f"snapshot missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"snapshot has unexpected keys: {sorted(extra)}")
    if not isinstance(snapshot.get("signals"), list):
        raise ValueError("signals must be a list")
    if not isinstance(snapshot.get("thirdPartyHosts"), list):
        raise ValueError("thirdPartyHosts must be a list")
    if not isinstance(snapshot.get("trackers"), list):
        raise ValueError("trackers must be a list")
    return snapshot


def load_telemetry_file(path):
    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    obj = json.loads(raw.decode("utf-8"))
    if obj.get("schemaVersion") == BATCH_SCHEMA:
        observations = obj.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError("batch has no observations")
        return [validate_snapshot_shape(x) for x in observations]
    return [validate_snapshot_shape(obj)]
