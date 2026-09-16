"""Generate coherent synthetic Veilance telemetry/policy/report triples.

Every row is constructed from one scenario. Telemetry is generated first, policy
language is generated for the selected disclosure relationship, and the expected
finding is then grounded in those exact two artifacts.
"""

import argparse
import copy
import datetime as dt
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from input_normalization import validate_policy_document
from schema import recompute_counts, validate_report
from telemetry import SNAPSHOT_SCHEMA, validate_snapshot_shape


DEFAULT_CONFIG = {
    "variants_per_source": 24,
    "no_policy_probability": 0.08,
    "augment_probability": 0.30,
    "max_findings": 8,
    "comparison_weights": {
        "matched": 0.27,
        "partially_matched": 0.30,
        "policy_only": 0.10,
        "observed_only": 0.27,
        "possible_contradiction": 0.06,
    },
}

COMPARISON_POLICY_STATUS = {
    "matched": "explicitly_disclosed",
    "policy_only": "explicitly_disclosed",
    "observed_only": "not_clearly_disclosed",
    "possible_contradiction": "contradicted",
    "indeterminate": "unknown",
}

CATEGORY_ALIASES = {
    "browser_characteristics": "browser_and_device_characteristics",
    "device_characteristics": "browser_and_device_characteristics",
    "site_analytics": "analytics",
    "tracking": "tracking",
    "advertising_tracker": "advertising",
    "social_media": "social_media_tracking",
}

SIGNAL_TEMPLATES = {
    "cookies": [
        ("cookie-access", "Cookie", "read"),
        ("cookie-access", "Cookie", "write"),
    ],
    "browser_storage": [
        ("browser-storage", "Storage", "write"),
        ("browser-storage", "IndexedDB", "open"),
    ],
    "browser_and_device_characteristics": [
        ("navigator-characteristics", "Navigator", "read-user-agent"),
        ("navigator-characteristics", "Navigator", "read-plugins"),
        ("navigator-characteristics", "Navigator", "read-mime-types"),
        ("navigator-characteristics", "Navigator", "read-languages"),
        ("navigator-characteristics", "Navigator", "read-hardware-concurrency"),
        ("navigator-characteristics", "Navigator", "read-device-memory"),
    ],
    "network_characteristics": [
        ("network-information", "NetworkInformation", "read-effective-type"),
        ("network-information", "NetworkInformation", "read-rtt"),
        ("network-information", "NetworkInformation", "read-downlink"),
    ],
    "fingerprinting": [
        ("canvas-readback", "Canvas", "readback"),
        ("webgl-characteristics", "WebGL", "read-renderer"),
    ],
    "device_sensors": [("device-sensors", "Sensors", "listen")],
    "webrtc": [
        ("webrtc", "WebRTC", "create-data-channel"),
        ("webrtc", "WebRTC", "create-offer"),
    ],
    "service_workers": [("browser-storage", "ServiceWorker", "register")],
    "permissions": [("permissions", "Permissions", "query")],
    "geolocation": [("geolocation", "Geolocation", "get-current-position")],
    "telemetry": [("beacon", "Beacon", "send")],
}

TRACKER_CATEGORIES = {
    "analytics": "analytics",
    "advertising": "advertising",
    "social_media_tracking": "social_media",
    "tracking": "unknown",
}

DEFAULT_BEHAVIOR = {
    "cookies": "cookie_access",
    "browser_storage": "browser_storage_activity",
    "browser_and_device_characteristics": "browser_and_device_characteristics",
    "network_characteristics": "network_characteristics",
    "fingerprinting": "fingerprinting_related_behavior",
    "analytics": "analytics_tracking",
    "advertising": "advertising_tracking",
    "tracking": "tracker_activity",
    "social_media_tracking": "social_media_tracking",
    "third_party_requests": "third_party_communications",
    "device_sensors": "device_sensor_listener",
    "webrtc": "webrtc_initialization",
    "service_workers": "service_worker_registration",
    "permissions": "permission_query",
    "geolocation": "geolocation_access",
    "telemetry": "telemetry_reporting",
}

POLICY_LANGUAGE = {
    "cookies": {
        "heading": "Cookies and Similar Technologies",
        "explicit": "We read and write browser cookies to maintain sessions, remember settings, and measure use of the service.",
        "broad": "We use cookies and similar technologies to operate and understand our service.",
        "implicit": "We remember session state and user preferences between page loads.",
        "denial": "We do not use cookies or similar browser technologies on this service.",
    },
    "browser_storage": {
        "heading": "Browser Storage",
        "explicit": "We store service state in local storage, session storage, IndexedDB, and browser caches.",
        "broad": "We use cookies and similar local technologies to store service settings.",
        "implicit": "The service remembers application state and preferences in the browser.",
        "denial": "We do not store information in your browser.",
    },
    "browser_and_device_characteristics": {
        "heading": "Device and Browser Information",
        "explicit": "We collect browser type, user agent, language, plugins, MIME types, device memory, and hardware concurrency.",
        "broad": "We collect device, browser, and other technical information when you use the service.",
        "implicit": "We adapt the service to the technical configuration of the device used to access it.",
        "denial": "We do not collect browser or device characteristics.",
    },
    "network_characteristics": {
        "heading": "Connection Information",
        "explicit": "We access connection type, estimated round-trip time, and estimated downlink speed to adapt service performance.",
        "broad": "We collect technical and connection information about use of the service.",
        "implicit": "We adapt content delivery to current connection conditions.",
        "denial": "We do not access characteristics of your network connection.",
    },
    "fingerprinting": {
        "heading": "Device Identification",
        "explicit": "We use Canvas and WebGL characteristics with other device signals to distinguish browsers for security purposes.",
        "broad": "We collect device and browser information for security and fraud prevention.",
        "implicit": "We use technical signals to recognize unusual devices and protect accounts.",
        "denial": "We do not fingerprint browsers or use Canvas or WebGL for device identification.",
    },
    "analytics": {
        "heading": "Analytics",
        "explicit": "We use an analytics tracker to measure page use and service performance.",
        "broad": "We collect usage and technical information to understand and improve the service.",
        "implicit": "We evaluate how the service performs and which features are useful.",
        "denial": "We do not use analytics trackers.",
    },
    "advertising": {
        "heading": "Advertising",
        "explicit": "We load an advertising tracker to measure and personalize advertising.",
        "broad": "We and our partners may use online activity information for marketing.",
        "implicit": "We work with partners to show and measure promotional content.",
        "denial": "We do not use tracking technologies for advertising.",
    },
    "tracking": {
        "heading": "Tracking Technologies",
        "explicit": "We use a classified tracking service to measure activity across service pages.",
        "broad": "We use cookies and similar technologies to understand online activity.",
        "implicit": "We measure interactions across the service to improve user experience.",
        "denial": "We do not use tracking technologies.",
    },
    "social_media_tracking": {
        "heading": "Social Media Features",
        "explicit": "A social media integration may load tracking resources when a page is visited.",
        "broad": "Our pages may include features supplied by social media services.",
        "implicit": "Embedded social features are provided by external platforms.",
        "denial": "We do not load social media tracking resources.",
    },
    "third_party_requests": {
        "heading": "Service Providers",
        "explicit": "Your browser connects to an external service provider to load scripts required by this page.",
        "broad": "We use service providers to host, secure, and operate the service.",
        "implicit": "Some service functions are delivered with help from external providers.",
        "denial": "This page does not connect to third-party services.",
    },
    "device_sensors": {
        "heading": "Device Sensors",
        "explicit": "The service registers listeners for device sensor events when supported by the browser.",
        "broad": "We may collect information about device capabilities and interactions.",
        "implicit": "Interactive features respond to supported device movement and orientation capabilities.",
        "denial": "We do not access device sensor functionality.",
    },
    "webrtc": {
        "heading": "Real-Time Features",
        "explicit": "We initialize WebRTC data-channel and offer functionality for real-time features.",
        "broad": "Real-time communication features process technical connection information.",
        "implicit": "Peer-enabled features establish real-time browser connections when used.",
        "denial": "We do not initialize peer-to-peer or WebRTC functionality.",
    },
    "service_workers": {
        "heading": "Offline and Background Features",
        "explicit": "We register a service worker to support caching, offline use, and application updates.",
        "broad": "We use browser technologies for caching and background application functions.",
        "implicit": "The web application can retain resources for faster or offline use.",
        "denial": "We do not register service workers.",
    },
    "permissions": {
        "heading": "Browser Permissions",
        "explicit": "The service queries browser permission state before enabling permission-dependent features.",
        "broad": "Some features may request access to browser or device capabilities.",
        "implicit": "Feature availability depends on permissions selected in the browser.",
        "denial": "We do not query browser permission state.",
    },
    "geolocation": {
        "heading": "Location Information",
        "explicit": "With browser permission, the service attempts to access precise geolocation for location features.",
        "broad": "We may collect location information when location-based features are used.",
        "implicit": "Location-based features use the location selected or permitted by the user.",
        "denial": "We do not access browser geolocation.",
    },
    "telemetry": {
        "heading": "Diagnostics and Telemetry",
        "explicit": "We use browser telemetry and Beacon API reporting to send service diagnostics and usage measurements.",
        "broad": "We collect diagnostic, usage, and performance information to operate and improve the service.",
        "implicit": "The service reports operational measurements used to diagnose performance and reliability.",
        "denial": "We do not send browser telemetry or diagnostic reports.",
    },
}

BASE_SEVERITY = {
    "cookies": "low",
    "browser_storage": "medium",
    "browser_and_device_characteristics": "medium",
    "network_characteristics": "medium",
    "fingerprinting": "high",
    "analytics": "medium",
    "advertising": "high",
    "tracking": "medium",
    "social_media_tracking": "medium",
    "third_party_requests": "low",
    "device_sensors": "high",
    "webrtc": "medium",
    "service_workers": "low",
    "permissions": "medium",
    "geolocation": "high",
    "telemetry": "medium",
}


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class RelaxedMongoParser:
    """Parse data-only Mongo shell output without evaluating JavaScript.

    Supported input includes objects/arrays, quoted or bare object keys,
    single- or double-quoted strings, trailing commas, comments, numbers,
    booleans/null, and common scalar wrappers such as ObjectId(...), ISODate(...),
    NumberLong(...), Decimal128(...), and UUID(...). Unknown function calls are
    retained as plain serializable metadata and are never executed.
    """

    _NUMBER_RE = re.compile(
        r"[+-]?(?:0[xX][0-9a-fA-F]+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    )
    _IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")

    def __init__(self, text: str):
        self.text = text.lstrip("\ufeff")
        self.index = 0

    def parse(self) -> Any:
        self._skip_ignored()
        value = self._parse_value()
        self._skip_ignored()
        if self._peek() == ";":
            self.index += 1
            self._skip_ignored()
        if self.index != len(self.text):
            self._error("unexpected content after the top-level value")
        return value

    def parse_many(self) -> list[Any]:
        values = []
        while True:
            self._skip_ignored()
            while self._peek() == ";":
                self.index += 1
                self._skip_ignored()
            if self.index >= len(self.text):
                return values
            values.append(self._parse_value())

    def _position(self) -> str:
        line = self.text.count("\n", 0, self.index) + 1
        line_start = self.text.rfind("\n", 0, self.index) + 1
        column = self.index - line_start + 1
        return f"line {line}, column {column}"

    def _error(self, message: str):
        raise ValueError(f"{message} at {self._position()}")

    def _peek(self, offset: int = 0) -> str:
        position = self.index + offset
        return self.text[position] if position < len(self.text) else ""

    def _skip_ignored(self) -> None:
        while True:
            while self._peek() and self._peek().isspace():
                self.index += 1
            if self.text.startswith("//", self.index):
                newline = self.text.find("\n", self.index + 2)
                self.index = len(self.text) if newline < 0 else newline + 1
                continue
            if self.text.startswith("/*", self.index):
                end = self.text.find("*/", self.index + 2)
                if end < 0:
                    self._error("unterminated block comment")
                self.index = end + 2
                continue
            break

    def _consume(self, expected: str) -> None:
        self._skip_ignored()
        if not self.text.startswith(expected, self.index):
            self._error(f"expected {expected!r}")
        self.index += len(expected)

    def _parse_value(self) -> Any:
        self._skip_ignored()
        current = self._peek()
        if not current:
            self._error("expected a value")
        if current == "{":
            return self._parse_object()
        if current == "[":
            return self._parse_array()
        if current in {"'", '"'}:
            return self._parse_string()
        if current.isdigit() or current in {"+", "-", "."}:
            match = self._NUMBER_RE.match(self.text, self.index)
            if match:
                return self._parse_number(match.group(0))
        if self._IDENTIFIER_RE.match(self.text, self.index):
            return self._parse_identifier_value()
        self._error(f"unsupported token {current!r}")

    def _parse_object(self) -> dict:
        self._consume("{")
        result = {}
        self._skip_ignored()
        if self._peek() == "}":
            self.index += 1
            return result
        while True:
            self._skip_ignored()
            if self._peek() in {"'", '"'}:
                key = self._parse_string()
            else:
                key = self._parse_identifier()
            self._consume(":")
            result[str(key)] = self._parse_value()
            self._skip_ignored()
            if self._peek() == "}":
                self.index += 1
                return result
            self._consume(",")
            self._skip_ignored()
            if self._peek() == "}":
                self.index += 1
                return result

    def _parse_array(self) -> list:
        self._consume("[")
        result = []
        self._skip_ignored()
        if self._peek() == "]":
            self.index += 1
            return result
        while True:
            result.append(self._parse_value())
            self._skip_ignored()
            if self._peek() == "]":
                self.index += 1
                return result
            self._consume(",")
            self._skip_ignored()
            if self._peek() == "]":
                self.index += 1
                return result

    def _parse_string(self) -> str:
        quote = self._peek()
        self.index += 1
        output = []
        escapes = {
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "v": "\v",
            "0": "\0",
        }
        while self.index < len(self.text):
            current = self._peek()
            self.index += 1
            if current == quote:
                return "".join(output)
            if current != "\\":
                output.append(current)
                continue
            if self.index >= len(self.text):
                self._error("unterminated escape sequence")
            escaped = self._peek()
            self.index += 1
            if escaped in {"'", '"', "\\", "/"}:
                output.append(escaped)
            elif escaped in escapes:
                output.append(escapes[escaped])
            elif escaped == "x":
                digits = self.text[self.index : self.index + 2]
                if not re.fullmatch(r"[0-9a-fA-F]{2}", digits):
                    self._error("invalid hexadecimal escape")
                output.append(chr(int(digits, 16)))
                self.index += 2
            elif escaped == "u":
                if self._peek() == "{":
                    end = self.text.find("}", self.index + 1)
                    if end < 0:
                        self._error("unterminated Unicode escape")
                    digits = self.text[self.index + 1 : end]
                    if not re.fullmatch(r"[0-9a-fA-F]{1,6}", digits):
                        self._error("invalid Unicode escape")
                    self.index = end + 1
                else:
                    digits = self.text[self.index : self.index + 4]
                    if not re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                        self._error("invalid Unicode escape")
                    self.index += 4
                output.append(chr(int(digits, 16)))
            elif escaped in {"\n", "\r"}:
                if escaped == "\r" and self._peek() == "\n":
                    self.index += 1
            else:
                output.append(escaped)
        self._error("unterminated string")

    def _parse_identifier(self) -> str:
        self._skip_ignored()
        match = self._IDENTIFIER_RE.match(self.text, self.index)
        if not match:
            self._error("expected an identifier")
        self.index = match.end()
        return match.group(0)

    def _parse_identifier_value(self) -> Any:
        name = self._parse_identifier()
        lowered = name.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered in {"null", "none", "undefined", "nan"}:
            return None
        if lowered == "infinity":
            return None
        if name == "new":
            name = self._parse_identifier()
        self._skip_ignored()
        if self._peek() == "(":
            return self._parse_wrapper(name)
        return name

    def _parse_wrapper(self, name: str) -> Any:
        self._consume("(")
        arguments = []
        self._skip_ignored()
        if self._peek() != ")":
            while True:
                arguments.append(self._parse_value())
                self._skip_ignored()
                if self._peek() == ")":
                    break
                self._consume(",")
        self._consume(")")
        first = arguments[0] if arguments else None
        if name in {"ObjectId", "ISODate", "Date", "UUID", "Long"}:
            return first
        if name in {"NumberLong", "NumberInt", "Int32"}:
            try:
                return int(first)
            except (TypeError, ValueError):
                return first
        if name in {"Decimal128", "NumberDecimal"}:
            return str(first) if first is not None else None
        return {"$mongo_function": name, "arguments": arguments}

    def _parse_number(self, token: str) -> int | float:
        self.index += len(token)
        if token.lower().startswith(("0x", "+0x", "-0x")):
            sign = -1 if token.startswith("-") else 1
            return sign * int(token.lstrip("+-")[2:], 16)
        if any(character in token for character in ".eE"):
            return float(token)
        return int(token)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_database_text(text: str) -> Any:
    """Parse strict JSON first, then safe Mongo-shell-style data literals."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == 'Type "it" for more' or re.fullmatch(
                r"[A-Za-z0-9_.-]+>\s*it", stripped
            ):
                continue
            cleaned_lines.append(line)
        values = RelaxedMongoParser("\n".join(cleaned_lines)).parse_many()
        if not values:
            raise ValueError("input did not contain a data value")
        if len(values) == 1:
            return values[0]
        combined = []
        for value in values:
            if isinstance(value, list):
                combined.extend(value)
            else:
                combined.append(value)
        return combined


def _records_from_value(value: Any, label: str) -> list[dict]:
    if isinstance(value, dict):
        for collection_key in ("documents", "records", "results", "items", "data"):
            collection = value.get(collection_key)
            if isinstance(collection, list) and not any(
                key in value for key in ("policy_raw_results", "expected", "findings")
            ):
                value = collection
                break
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError(f"{label}: top level must be an object or an array of objects")


def load_input_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        return _records_from_value(parse_database_text(text), str(path))
    except ValueError as whole_file_error:
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.extend(
                    _records_from_value(
                        parse_database_text(line),
                        f"{path}:{line_number}",
                    )
                )
            except ValueError as line_error:
                raise ValueError(
                    f"{path}: could not parse as a document or line-delimited records; "
                    f"document error: {whole_file_error}; line error: {line_error}"
                ) from line_error
        return records


def iter_input(path: Path) -> Iterable[dict]:
    files = (
        [path]
        if path.is_file()
        else sorted(
            item
            for item in path.rglob("*")
            if item.is_file()
            and item.suffix.lower() in {".json", ".jsonl", ".txt", ".js"}
        )
    )
    if not files:
        raise ValueError(f"no .json, .jsonl, .txt, or .js data files found at {path}")
    for file in files:
        yield from load_input_records(file)


def source_report(record: dict) -> dict:
    if isinstance(record.get("expected"), dict):
        return record["expected"]
    policy_raw_results = record.get("policy_raw_results")
    if isinstance(policy_raw_results, str):
        policy_raw_results = parse_database_text(policy_raw_results)
    if isinstance(policy_raw_results, dict):
        return policy_raw_results
    if isinstance(record.get("findings"), list):
        return record
    raise ValueError(
        "source must be a Veilance report, a database row with policy_raw_results, or a training row with expected"
    )


def normalize_category(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "browser_and_device_characteristics"
    value = value.strip().lower().replace(" ", "_")
    return CATEGORY_ALIASES.get(value, value)


def scenario_templates(report: dict, config: dict, rng: random.Random) -> list[dict]:
    findings = report.get("findings")
    if not isinstance(findings, list) or not findings:
        raise ValueError("source report must contain at least one finding")
    templates = []
    seen = set()
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        category = normalize_category(finding.get("category"))
        if category not in POLICY_LANGUAGE or category in seen:
            continue
        seen.add(category)
        behavior = finding.get("behavior")
        if not isinstance(behavior, str) or not behavior:
            behavior = DEFAULT_BEHAVIOR[category]
        templates.append({"category": category, "behavior": behavior})

    if not templates:
        templates.append(
            {
                "category": "browser_and_device_characteristics",
                "behavior": "browser_and_device_characteristics",
            }
        )

    if rng.random() < float(config.get("augment_probability", 0.30)):
        candidates = [category for category in POLICY_LANGUAGE if category not in seen]
        if candidates:
            category = rng.choice(candidates)
            templates.append(
                {"category": category, "behavior": DEFAULT_BEHAVIOR[category]}
            )
    return templates[: int(config.get("max_findings", 8))]


def choose_comparison(category: str, config: dict, rng: random.Random) -> str:
    weights = copy.deepcopy(config.get("comparison_weights", {}))
    if category not in {
        "cookies",
        "fingerprinting",
        "analytics",
        "advertising",
        "tracking",
        "social_media_tracking",
        "device_sensors",
        "webrtc",
        "geolocation",
        "telemetry",
    }:
        weights["possible_contradiction"] = 0.0
    values = [
        "matched",
        "partially_matched",
        "policy_only",
        "observed_only",
        "possible_contradiction",
    ]
    numeric_weights = [max(0.0, float(weights.get(value, 0.0))) for value in values]
    if not any(numeric_weights):
        return "partially_matched"
    return rng.choices(values, weights=numeric_weights, k=1)[0]


def synthetic_domain(source_domain: str, source_id: str, variant_index: int) -> str:
    label = stable_hash(f"{source_domain}:{source_id}:{variant_index}")
    return f"https://site-{label}.example"


def source_family_id(source_domain: str) -> str:
    candidate = source_domain if "://" in source_domain else "https://" + source_domain
    parsed = urlparse(candidate)
    family = (parsed.hostname or source_domain).lower().strip().rstrip(".")
    return stable_hash("source-domain:" + family)


def source_provenance(source: dict) -> dict:
    provenance = {
        "source_kind": (
            "processed_database_result"
            if "policy_raw_results" in source
            else "training_row"
            if "expected" in source
            else "report"
        )
    }
    for source_key, metadata_key in (
        ("_id", "database_record_id"),
        ("policy_id", "policy_id"),
        ("inserted_at", "inserted_at"),
        ("associated_domain", "associated_domain"),
        ("associated_policy_link", "associated_policy_link"),
    ):
        value = source.get(source_key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if value is not None:
                provenance[metadata_key] = value
    return provenance


def make_signal_observation(category: str, rng: random.Random) -> tuple[list[dict], list[str], int]:
    signals = []
    evidence = []
    total = 0
    for indicator_id, api, action in SIGNAL_TEMPLATES.get(category, []):
        count = rng.randint(1, 6)
        total += count
        signals.append(
            {
                "indicatorId": indicator_id,
                "api": api,
                "action": action,
                "count": count,
            }
        )
        evidence.append(
            f"{api}.{action} observed {count} {'time' if count == 1 else 'times'}"
        )
    return signals, evidence, total


def make_network_observation(
    category: str,
    token: str,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[str], int, dict]:
    needs_host = category in TRACKER_CATEGORIES or category == "third_party_requests"
    if not needs_host:
        return [], [], [], 0, {}
    host = f"{category.replace('_', '-')}-{token[:8]}.example"
    requests = rng.randint(1, 9)
    host_record = {
        "host": host,
        "requests": requests,
        "resourceTypes": {"script": requests},
    }
    evidence = [
        f"Communication with {host} observed for {requests} {'request' if requests == 1 else 'requests'}"
    ]
    trackers = []
    observation_count = requests
    context = {"host": host, "requests": requests}
    if category in TRACKER_CATEGORIES:
        tracker_category = TRACKER_CATEGORIES[category]
        tracker = {
            "id": f"tracker.synthetic-{category}-{token}",
            "category": tracker_category,
            "requests": requests,
        }
        trackers.append(tracker)
        evidence.append(
            f"Veilance {tracker_category} tracker classification observed for {requests} {'request' if requests == 1 else 'requests'}"
        )
        # Host and tracker records describe the same requests. Count them once.
        observation_count = requests
    return [host_record], trackers, evidence, observation_count, context


def make_observed_evidence(
    category: str,
    token: str,
    rng: random.Random,
) -> dict:
    signals, signal_evidence, signal_count = make_signal_observation(category, rng)
    hosts, trackers, network_evidence, network_count, context = make_network_observation(
        category, token, rng
    )
    evidence = signal_evidence + network_evidence
    count = signal_count if signals else network_count
    if count < 1:
        # A category without a low-level API template is represented by a
        # conservative Permissions query rather than invented network activity.
        signals = [
            {
                "indicatorId": f"synthetic-{category}",
                "api": "Permissions",
                "action": "query",
                "count": 1,
            }
        ]
        evidence = ["Permissions.query observed 1 time"]
        count = 1
    return {
        "signals": signals,
        "hosts": hosts,
        "trackers": trackers,
        "evidence": evidence,
        "observation_count": count,
        "context": context,
    }


def policy_statement(
    category: str,
    comparison: str,
    rng: random.Random,
    context: dict,
) -> tuple[str, str, str]:
    language = POLICY_LANGUAGE[category]
    if comparison in {"observed_only", "indeterminate"}:
        return COMPARISON_POLICY_STATUS[comparison], "", ""
    if comparison == "matched" or comparison == "policy_only":
        status = "explicitly_disclosed"
        key = "explicit"
    elif comparison == "partially_matched":
        status = rng.choice(["broadly_disclosed", "implicitly_disclosed"])
        key = "broad" if status == "broadly_disclosed" else "implicit"
    else:
        status = "contradicted"
        key = "denial"
    statement = language[key]
    if context.get("host") and key == "explicit":
        statement = statement.rstrip(".") + f" The observed provider is {context['host']}."
    return status, statement, language["heading"]


def severity_for(category: str, comparison: str, rng: random.Random) -> str:
    if comparison in {"matched", "policy_only"}:
        return "informational"
    if comparison == "partially_matched":
        return "low" if BASE_SEVERITY.get(category) == "low" else "medium"
    if comparison == "possible_contradiction":
        if category in {"fingerprinting", "geolocation", "device_sensors"} and rng.random() < 0.08:
            return "critical"
        return "high"
    return BASE_SEVERITY.get(category, "medium")


def confidence_for(comparison: str, duration: int, rng: random.Random) -> float:
    base = {
        "matched": 0.92,
        "partially_matched": 0.80,
        "policy_only": 0.66,
        "observed_only": 0.84,
        "possible_contradiction": 0.90,
        "indeterminate": 0.46,
    }[comparison]
    if comparison == "policy_only" and duration <= 5:
        base -= 0.18
    return round(max(0.2, min(0.98, base + rng.uniform(-0.04, 0.04))), 2)


def description_for(category: str, comparison: str) -> str:
    behavior = DEFAULT_BEHAVIOR[category].replace("_", " ")
    if comparison == "policy_only":
        return f"The policy describes {behavior}, which was not directly observed in this visit."
    return f"Veilance observed {behavior} during this browser visit."


def explanation_for(comparison: str) -> str:
    return {
        "matched": "The observed behavior is specifically described by the applicable policy.",
        "partially_matched": "The policy covers a broader or indirect category but does not describe the observed behavior with the same specificity.",
        "policy_only": "The policy describes this behavior, but this visit did not provide direct evidence that it occurred.",
        "observed_only": "The behavior was observed during this visit but was not clearly disclosed in the applicable policy, creating a potential disclosure gap.",
        "possible_contradiction": "Strong observed evidence appears inconsistent with an explicit policy statement, but this is not a legal conclusion.",
        "indeterminate": "The available policy evidence is insufficient for a reliable comparison.",
    }[comparison]


def merge_records(records: list[dict], key_fields: tuple[str, ...], count_field: str) -> list[dict]:
    merged = {}
    for record in records:
        key = tuple(record[field] for field in key_fields)
        if key not in merged:
            merged[key] = copy.deepcopy(record)
        else:
            merged[key][count_field] += record[count_field]
            if "resourceTypes" in record:
                for resource_type, count in record["resourceTypes"].items():
                    merged[key]["resourceTypes"][resource_type] = (
                        merged[key]["resourceTypes"].get(resource_type, 0) + count
                    )
    return list(merged.values())


def redacted_document(hosts: list[dict], categories: set[str]) -> dict:
    scripts = "".join(
        f'<script data-veilance-src-origin="https://{host["host"]}" type="application/veilance-redacted"></script>'
        for host in hosts
    )
    html = f"<!doctype html><html><head>{scripts}</head><body>[REDACTED TEXT]</body></html>"
    return {
        "format": "veilance.redacted-html.v1",
        "html": html,
        "truncated": False,
        "originalElementCount": 3 + len(hosts),
        "serializedChars": len(html),
        "redaction": {
            "textNodesRedacted": 1,
            "attributesRemoved": 0,
            "urlsReduced": len(hosts),
            "privateUrlsRemoved": 0,
            "inlineScriptsRedacted": 0,
            "styleBlocksRedacted": 0,
            "formControlsRedacted": 0,
            "commentsRemoved": 0,
            "opaqueNodesRedacted": 0,
            "nodesOmitted": 0,
        },
        "evidence": {
            "resourceHosts": [
                {
                    "host": host["host"],
                    "thirdParty": True,
                    "count": host["requests"],
                    "tags": {"script": host["resourceTypes"].get("script", 0)},
                }
                for host in hosts
            ],
            "inlineScriptHints": {
                "canvas": int("fingerprinting" in categories),
                "webgl": int("fingerprinting" in categories),
                "webgpu": 0,
                "audio": 0,
                "fonts": 0,
                "navigator": int("browser_and_device_characteristics" in categories),
                "screen": 0,
                "webrtc": int("webrtc" in categories),
                "advertising": int("advertising" in categories),
                "antiBlocking": 0,
            },
            "domMarkers": {
                "advertising": 0,
                "consent": 0,
                "antiBlocking": 0,
                "tracking": 0,
            },
        },
    }


def make_snapshot(
    hostname: str,
    event_id: str,
    duration: int,
    observations: list[dict],
    categories: set[str],
    rng: random.Random,
) -> dict:
    signals = merge_records(
        [signal for item in observations for signal in item["signals"]],
        ("api", "action"),
        "count",
    )
    hosts = merge_records(
        [host for item in observations for host in item["hosts"]],
        ("host",),
        "requests",
    )
    trackers = merge_records(
        [tracker for item in observations for tracker in item["trackers"]],
        ("id",),
        "requests",
    )
    third_party_requests = sum(host["requests"] for host in hosts)
    first_party_requests = rng.randint(2, 12)
    accessible_cookies = rng.randint(1, 12) if "cookies" in categories else 0
    has_storage = "browser_storage" in categories
    service_worker = "service_workers" in categories

    reasons = []
    reason_specs = [
        ("cookies", "cookie-access", "low", 2),
        ("browser_storage", "persistent-storage", "low", 2),
        ("browser_and_device_characteristics", "navigator-characteristics", "low", 2),
        ("device_sensors", "device-sensor-access", "medium", 10),
        ("webrtc", "webrtc", "medium", 10),
        ("fingerprinting", "broad-fingerprint-surface", "high", 20),
        ("service_workers", "service-worker-registration", "low", 2),
    ]
    for category, reason_id, severity, points in reason_specs:
        if category in categories:
            reasons.append({"id": reason_id, "severity": severity, "points": points})
    for tracker in trackers:
        reasons.append(
            {
                "id": tracker["id"],
                "severity": "low" if tracker["category"] != "advertising" else "medium",
                "points": 2 if tracker["category"] != "advertising" else 10,
            }
        )
    score = min(100, sum(reason["points"] for reason in reasons))

    snapshot = {
        "schemaVersion": SNAPSHOT_SCHEMA,
        "eventId": event_id,
        "extensionVersion": "0.8-synthetic",
        "site": {"hostname": hostname, "https": True},
        "observation": {
            "durationSeconds": duration,
            "totalRequests": first_party_requests + third_party_requests,
            "firstPartyRequests": first_party_requests,
            "thirdPartyRequests": third_party_requests,
        },
        "thirdPartyHosts": hosts,
        "trackers": trackers,
        "signals": signals,
        "page": {
            "scriptCount": rng.randint(2, 10) + third_party_requests,
            "thirdPartyScriptCount": third_party_requests,
            "iframeCount": 0,
            "thirdPartyIframeCount": 0,
            "accessibleCookieCount": accessible_cookies,
            "localStorageKeyCount": rng.randint(1, 10) if has_storage else 0,
            "sessionStorageKeyCount": rng.randint(0, 5) if has_storage else 0,
            "indexedDbCount": rng.randint(1, 3) if has_storage else 0,
            "cacheCount": rng.randint(1, 4) if has_storage or service_worker else 0,
            "serviceWorkerControlled": service_worker,
        },
        "security": {
            "contentSecurityPolicy": rng.choice([True, False]),
            "strictTransportSecurity": rng.choice([True, False]),
            "permissionsPolicy": rng.choice([True, False]),
            "referrerPolicy": rng.choice([True, False]),
            "xFrameOptions": rng.choice([True, False]),
            "crossOriginOpenerPolicy": rng.choice([True, False]),
            "crossOriginResourcePolicy": rng.choice([True, False]),
        },
        "interest": {
            "score": score,
            "level": "interesting" if score >= 25 else "routine",
            "minimumScore": 25,
            "eligible": score >= 25,
            "reasons": reasons,
        },
        "redactedDocument": redacted_document(hosts, categories),
    }
    return validate_snapshot_shape(snapshot)


def important_limitations(duration: int, snapshot: dict, categories: set[str]) -> list[str]:
    limitations = [
        f"This telemetry represents a {duration}-second browser observation. Absence of a behavior in this sample does not demonstrate that the behavior never occurs."
    ]
    if snapshot["thirdPartyHosts"]:
        limitations.append(
            "Veilance observed network destinations and browser activity but did not inspect request or response contents. Third-party communication does not by itself establish what personal information was transmitted."
        )
    if snapshot["signals"]:
        limitations.append(
            "Browser API access does not necessarily establish the purpose for which the accessed information was used."
        )
    if any(
        snapshot["page"][key] > 0
        for key in (
            "accessibleCookieCount",
            "localStorageKeyCount",
            "sessionStorageKeyCount",
            "indexedDbCount",
            "cacheCount",
        )
    ) or snapshot["page"]["serviceWorkerControlled"]:
        limitations.append(
            "Existing cookies, browser storage entries, IndexedDB databases, cache entries, or service-worker state may have existed before the current observation."
        )
    if "device_sensors" in categories:
        limitations.append(
            "Sensor listener registration does not by itself establish which sensor was accessed or whether sensor values were read."
        )
    if "webrtc" in categories:
        limitations.append(
            "WebRTC initialization does not by itself establish that a peer connection succeeded or that local or public IP information was exposed."
        )
    return limitations


def summary_for(findings: list[dict], applicable: bool) -> str:
    observed = sum(
        finding["telemetry"]["status"] == "observed" for finding in findings
    )
    counts = recompute_counts(findings)
    if not applicable:
        return (
            f"Veilance identified {observed} grouped observed behaviors during this visit, but no applicable privacy policy was available, so the comparisons are indeterminate."
        )
    gaps = counts["observed_only"]
    contradictions = counts["possible_contradictions"]
    return (
        f"During this visit, Veilance identified {observed} grouped observed behaviors. "
        f"The comparison produced {counts['matched']} matched, {counts['partially_matched']} partially matched, "
        f"{gaps} observed-only, and {contradictions} possible-contradiction findings."
    )


def generate_variant(
    source: dict,
    source_index: int,
    variant_index: int,
    seed: str,
    config: dict,
) -> dict:
    report_source = source_report(source)
    source_domain = str(
        report_source.get("domain")
        or source.get("associated_domain")
        or f"source-{source_index}.example"
    )
    source_id = stable_hash(stable_json(source))
    rng = random.Random(f"{seed}:{source_id}:{variant_index}")
    domain = synthetic_domain(source_domain, source_id, variant_index)
    hostname = urlparse(domain).hostname
    duration = rng.choice([1, 2, 3, 5, 15, 30, 45, 60, 90])
    templates = scenario_templates(report_source, config, rng)
    no_policy = rng.random() < float(config.get("no_policy_probability", 0.08))

    scenario_items = []
    observed_records = []
    observed_categories = set()
    for item_index, template in enumerate(templates):
        category = template["category"]
        comparison = (
            "indeterminate" if no_policy else choose_comparison(category, config, rng)
        )
        observed = comparison != "policy_only"
        observation = (
            make_observed_evidence(
                category,
                stable_hash(f"{source_id}:{variant_index}:{item_index}"),
                rng,
            )
            if observed
            else {
                "signals": [],
                "hosts": [],
                "trackers": [],
                "evidence": [],
                "observation_count": None,
                "context": {},
            }
        )
        if observed:
            observed_records.append(observation)
            observed_categories.add(category)
        scenario_items.append(
            {
                **template,
                "comparison": comparison,
                "observation": observation,
            }
        )

    snapshot = make_snapshot(
        hostname=hostname,
        event_id=f"synthetic-{stable_hash(f'{source_id}:{variant_index}:{seed}')}",
        duration=duration,
        observations=observed_records,
        categories=observed_categories,
        rng=rng,
    )

    sections_by_heading: dict[str, list[str]] = {}
    findings = []
    for item in scenario_items:
        category = item["category"]
        comparison = item["comparison"]
        observation = item["observation"]
        policy_status, policy_evidence, policy_section = policy_statement(
            category,
            comparison,
            rng,
            observation["context"],
        )
        if policy_evidence:
            sections_by_heading.setdefault(policy_section, []).append(policy_evidence)

        if comparison == "policy_only":
            if category in {"data_retention", "data_sale", "government_disclosure"}:
                telemetry_status = "unsupported"
                observation_count = None
            elif duration <= 5:
                telemetry_status = "insufficient_sample"
                observation_count = None
            else:
                telemetry_status = "not_observed"
                observation_count = 0
            telemetry_evidence = []
        else:
            telemetry_status = "observed"
            observation_count = observation["observation_count"]
            telemetry_evidence = observation["evidence"]

        findings.append(
            {
                "behavior": item["behavior"],
                "category": category,
                "description": description_for(category, comparison),
                "policy": {
                    "status": policy_status,
                    "evidence": policy_evidence,
                    "section": policy_section,
                },
                "telemetry": {
                    "status": telemetry_status,
                    "evidence": telemetry_evidence,
                    "observation_count": observation_count,
                },
                "comparison": comparison,
                "severity": severity_for(category, comparison, rng),
                "confidence": confidence_for(comparison, duration, rng),
                "explanation": explanation_for(comparison),
            }
        )

    if no_policy:
        sections = []
        policy_url = ""
    else:
        if not sections_by_heading:
            sections_by_heading["Privacy Questions"] = [
                "This notice explains how users can contact us with privacy questions."
            ]
        sections = [
            {"heading": heading, "text": " ".join(statements)}
            for heading, statements in sections_by_heading.items()
        ]
        policy_url = domain + "/privacy"

    observed_at = (
        dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        + dt.timedelta(days=variant_index, seconds=source_index)
    ).isoformat().replace("+00:00", "Z")
    policy_document = {
        "url": policy_url,
        "found": not no_policy,
        "applicable": not no_policy,
        "complete": True,
        "retrieval_method": "synthetic_fixture",
        "sections": sections,
        "limitations": (
            []
            if not no_policy
            else [
                "No applicable privacy policy could be retrieved or verified for the supplied domain."
            ]
        ),
    }
    expected = {
        "domain": domain,
        "privacy_policy": {
            "url": policy_url,
            "found": not no_policy,
            "applicable": not no_policy,
        },
        "visit": {
            "observed_at": observed_at,
            "duration_seconds": duration,
        },
        "analysis": {
            "summary": summary_for(findings, not no_policy),
            "overall_confidence": round(
                sum(finding["confidence"] for finding in findings) / len(findings),
                2,
            ),
            "counts": recompute_counts(findings),
        },
        "findings": findings,
        "important_limitations": important_limitations(
            duration, snapshot, observed_categories
        )
        + policy_document["limitations"],
    }
    row = {
        "telemetry": snapshot,
        "policy_document": policy_document,
        "expected": expected,
        "_synthetic_metadata": {
            "synthetic": True,
            "source_id": source_id,
            "family_id": source_family_id(source_domain),
            "source_index": source_index,
            "variant_index": variant_index,
            "seed": seed,
            "source_domain": source_domain,
            "synthetic_domain": domain,
            "provenance": source_provenance(source),
        },
    }
    validate_policy_document(policy_document)
    validate_report(expected)
    return row


def load_config(path: str | None) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        supplied = load_json(Path(path))
        if not isinstance(supplied, dict):
            raise ValueError("generator config must be a JSON object")
        for key, value in supplied.items():
            if key == "comparison_weights" and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate coherent raw training JSONL from Veilance reports, JSON/JSONL "
            "database exports, or pasted Mongo shell output. Output rows contain "
            "exact v2 telemetry, policy_document, and expected."
        )
    )
    parser.add_argument(
        "input",
        help=(
            "Input .json, .jsonl, .txt, .js, or directory. Mongo shell syntax "
            "such as unquoted keys and ObjectId(...) is accepted as data only."
        ),
    )
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--seed", default="1337")
    parser.add_argument("--count", type=int, help="Variants per source")
    parser.add_argument(
        "--strip-metadata",
        action="store_true",
        help=(
            "Remove database provenance and generation details while retaining "
            "the minimum source-domain family metadata needed for leakage-safe splits."
        ),
    )
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    count = int(config["variants_per_source"]) if args.count is None else args.count
    if count < 1:
        raise ValueError("--count must be at least 1")
    records = list(iter_input(Path(args.input)))
    if not records:
        raise ValueError("no source records found")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    invalid = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for source_index, record in enumerate(records):
            for variant_index in range(count):
                try:
                    row = generate_variant(
                        record,
                        source_index,
                        variant_index,
                        args.seed,
                        config,
                    )
                    if args.strip_metadata:
                        metadata = row.get("_synthetic_metadata", {})
                        row["_synthetic_metadata"] = {
                            key: metadata[key]
                            for key in ("synthetic", "family_id", "source_domain")
                            if key in metadata
                        }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                except Exception as exc:
                    invalid += 1
                    print(
                        f"[invalid] source={source_index} variant={variant_index}: {exc}",
                        file=sys.stderr,
                    )
                    if args.fail_on_invalid:
                        raise

    print(
        json.dumps(
            {
                "source_records": len(records),
                "processed_database_records": sum(
                    "policy_raw_results" in record for record in records
                ),
                "source_domains": len(
                    {
                        str(source_report(record).get("domain") or "")
                        for record in records
                    }
                ),
                "variants_per_source": count,
                "written": written,
                "invalid": invalid,
                "output": str(output_path),
            },
            indent=2,
        )
    )
    if written == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
