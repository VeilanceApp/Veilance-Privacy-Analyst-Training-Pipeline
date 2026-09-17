"""Safe, host-orchestrated privacy-policy retrieval with Playwright.

The model never launches a browser. The inference host retrieves and verifies the
policy first, preserves a canonical extracted document, and supplies a bounded set
of relevant policy sections to Qwen.

Search is used only to discover candidate URLs. Search snippets are never treated
as policy evidence.

Backward compatibility:
    result["sections"] remains the model-ready section list.

Additional canonical evidence is available at:
    result["document"]["full_text"]
    result["document"]["sections"]

The model context limit is separate from the extraction limit. A long policy can be
stored completely while only the most relevant chunks are supplied to Qwen.
"""

import datetime as dt
import hashlib
import ipaddress
import re
import socket

from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse


POLICY_LINK_RE = re.compile(
    r"(?:privacy(?:\s+(?:policy|notice|statement))?|"
    r"data\s+(?:policy|privacy|protection)|privacy\s+center)",
    re.IGNORECASE,
)

POLICY_TEXT_RE = re.compile(
    r"(?:personal (?:data|information)|information we collect|data we collect|"
    r"information we receive|how we (?:use|share|collect|process)|your privacy|"
    r"privacy rights|data protection|your information)",
    re.IGNORECASE,
)

GENERIC_HOST_LABELS = {
    "www",
    "app",
    "web",
    "www2",
    "com",
    "org",
    "net",
    "io",
    "ai",
    "co",
    "uk",
    "us",
    "ca",
    "de",
    "fr",
    "jp",
    "au",
}

COMMON_POLICY_PATHS = (
    "/privacy",
    "/privacy-policy",
    "/privacy_policy",
    "/legal/privacy",
    "/legal/privacy-policy",
    "/legal/privacy-notice",
    "/policies/privacy",
    "/policies/privacy-policy",
    "/privacy/statement",
    "/privacy/notice",
)

CATEGORY_TERMS = {
    "cookies": (
        "cookie",
        "cookies",
        "browser storage",
        "local storage",
        "session storage",
        "storage technologies",
    ),
    "tracking": (
        "tracking",
        "analytics",
        "pixel",
        "pixels",
        "beacon",
        "advertising",
        "measurement",
        "third party",
        "third-party",
    ),
    "fingerprinting": (
        "fingerprint",
        "fingerprinting",
        "device information",
        "browser information",
        "device identifier",
        "unique identifier",
        "device characteristics",
        "browser characteristics",
        "canvas",
        "webgl",
    ),
    "network": (
        "ip address",
        "internet protocol",
        "network",
        "server logs",
        "connection information",
    ),
    "sharing": (
        "share",
        "sharing",
        "shared",
        "third parties",
        "third-party",
        "service providers",
        "partners",
        "vendors",
        "disclose",
        "disclosure",
    ),
    "collection": (
        "collect",
        "collection",
        "information we collect",
        "data we collect",
        "information collected",
        "receive information",
    ),
    "usage": (
        "use your information",
        "use personal information",
        "use personal data",
        "purposes",
        "processing",
        "process your",
    ),
    "device": (
        "device",
        "browser",
        "operating system",
        "screen",
        "user agent",
        "device type",
    ),
    "location": (
        "location",
        "geolocation",
        "approximate location",
        "precise location",
    ),
    "permissions": (
        "permission",
        "permissions",
        "camera",
        "microphone",
        "notifications",
        "sensor",
        "sensors",
    ),
    "security": (
        "security",
        "protect",
        "safeguard",
        "encryption",
    ),
    "retention": (
        "retain",
        "retention",
        "delete",
        "deletion",
        "storage period",
    ),
    "rights": (
        "your rights",
        "privacy rights",
        "access",
        "correction",
        "deletion",
        "opt out",
        "opt-out",
    ),
}


def _plain_space(value):
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _origin(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _host(url):
    parsed = urlparse(url)
    if not parsed.hostname:
        return ""
    return parsed.hostname.lower().rstrip(".")


def _same_site(left, right):
    left_host = _host(left)
    right_host = _host(right)

    if not left_host or not right_host:
        return False

    return (
        left_host == right_host
        or left_host.endswith("." + right_host)
        or right_host.endswith("." + left_host)
    )


def _service_tokens(domain_url):
    labels = []

    for label in _host(domain_url).split("."):
        if label in GENERIC_HOST_LABELS:
            continue
        if len(label) < 3:
            continue
        labels.append(label)

    labels.sort(key=len, reverse=True)
    return labels[:3]


def _unwrap_search_url(href):
    if href.startswith("//"):
        href = "https:" + href

    parsed = urlparse(href)

    if "duckduckgo.com" in (parsed.hostname or ""):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)

    return href


def _candidate_key(url):
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"


def _document_hash(text):
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _active_value(value):
    if value is None or value is False:
        return False

    if value is True:
        return True

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        return value.strip() != ""

    if isinstance(value, dict):
        return any(_active_value(item) for item in value.values())

    if isinstance(value, (list, tuple, set)):
        return any(_active_value(item) for item in value)

    return True


def _active_telemetry_terms(value, output=None):
    if output is None:
        output = set()

    if isinstance(value, dict):
        for key, item in value.items():
            if _active_value(item):
                output.add(key.lower().replace("_", " ").replace("-", " "))
                _active_telemetry_terms(item, output)
        return output

    if isinstance(value, (list, tuple, set)):
        for item in value:
            if _active_value(item):
                _active_telemetry_terms(item, output)
        return output

    if isinstance(value, str) and value.strip():
        output.add(value.lower())

    return output


class PolicyRetriever:
    def __init__(
        self,
        timeout_ms=20_000,
        max_document_chars=250_000,
        section_chunk_chars=4_500,
        max_sections=256,
        max_model_policy_chars=14_000,
        max_model_sections=8,
        max_candidates=14,
        search_enabled=True,
        browser_name="chromium",
        headless=True,
        allow_private_network=False,
    ):
        self.timeout_ms = timeout_ms
        self.max_document_chars = max_document_chars
        self.section_chunk_chars = section_chunk_chars
        self.max_sections = max_sections
        self.max_model_policy_chars = max_model_policy_chars
        self.max_model_sections = max_model_sections
        self.max_candidates = max_candidates
        self.search_enabled = search_enabled
        self.browser_name = browser_name
        self.headless = headless
        self.allow_private_network = allow_private_network
        self._dns_cache = {}

    def _assert_safe_url(self, url):
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL is empty")

        url = url.strip()
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("only absolute HTTP(S) URLs are allowed")

        if parsed.username or parsed.password:
            raise ValueError("URLs containing credentials are not allowed")

        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise ValueError("URL contains an invalid port") from exc

        if port not in {80, 443}:
            raise ValueError("only ports 80 and 443 are allowed")

        hostname = parsed.hostname.lower().rstrip(".")

        if self.allow_private_network:
            return url

        if hostname in {"localhost", "localhost.localdomain"}:
            raise ValueError("local hosts are not allowed")

        if hostname.endswith((".local", ".internal", ".localhost")):
            raise ValueError("local or internal hosts are not allowed")

        cache_key = (hostname, port)
        addresses = self._dns_cache.get(cache_key)

        if addresses is None:
            try:
                info = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
            except socket.gaierror as exc:
                raise ValueError(f"host did not resolve: {hostname}") from exc

            addresses = sorted({row[4][0] for row in info})

            if not addresses:
                raise ValueError(f"host did not resolve: {hostname}")

            self._dns_cache[cache_key] = addresses

        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise ValueError(
                    f"host resolves to a non-public address: {hostname}"
                )

        return url

    def _route(self, route):
        request = route.request
        parsed = urlparse(request.url)

        if parsed.scheme not in {"http", "https"}:
            if parsed.scheme in {"data", "blob", "about"}:
                route.continue_()
            else:
                route.abort()
            return

        try:
            self._assert_safe_url(request.url)
        except ValueError:
            route.abort()
            return

        if request.resource_type in {"image", "media", "font"}:
            route.abort()
            return

        route.continue_()

    def _looks_like_policy(self, url, title, body_text):
        title = _plain_space(title)
        body_text = _plain_space(body_text)

        if len(body_text) < 250:
            return False

        score = 0

        if POLICY_LINK_RE.search(url):
            score += 2
        if POLICY_LINK_RE.search(title):
            score += 4
        if POLICY_LINK_RE.search(body_text[:6_000]):
            score += 2
        if POLICY_TEXT_RE.search(body_text[:20_000]):
            score += 3
        if re.search(
            r"last updated|effective date|last modified",
            body_text[:6_000],
            re.IGNORECASE,
        ):
            score += 1

        title_lower = title.lower()
        cookie_only = (
            "cookie" in title_lower
            and "privacy" not in title_lower
            and "data" not in title_lower
        )

        return score >= 5 and not cookie_only

    def _is_applicable(self, domain_url, policy_url, title, body_text):
        if _same_site(domain_url, policy_url):
            return True

        haystack = re.sub(
            r"[^a-z0-9]+",
            "",
            (title + " " + body_text[:30_000]).lower(),
        )

        for token in _service_tokens(domain_url):
            cleaned = re.sub(r"[^a-z0-9]+", "", token.lower())
            if cleaned and cleaned in haystack:
                return True

        return False

    def _wait_for_document_stability(self, page):
        try:
            page.wait_for_load_state(
                "networkidle",
                timeout=min(self.timeout_ms, 4_000),
            )
        except Exception:
            pass

        previous_length = -1
        stable_reads = 0

        for _ in range(16):
            try:
                current_length = page.evaluate(
                    """
                    () => {
                        if (!document.body) return 0;
                        return (document.body.innerText || document.body.textContent || '').length;
                    }
                    """
                )
            except Exception:
                current_length = 0

            if current_length >= 500 and previous_length >= 0:
                difference = abs(current_length - previous_length)
                tolerance = max(25, current_length // 100)

                if difference <= tolerance:
                    stable_reads += 1
                else:
                    stable_reads = 0

                if stable_reads >= 3:
                    return

            previous_length = current_length
            page.wait_for_timeout(350)

    def _prime_document(self, page):
        try:
            page.evaluate(
                """
                () => {
                    window.scrollTo(0, document.body ? document.body.scrollHeight : 0);
                }
                """
            )
            page.wait_for_timeout(250)
            page.evaluate("() => window.scrollTo(0, 0)")
            page.wait_for_timeout(150)
        except Exception:
            pass

    def _extract_from_frame(self, frame):
        return frame.evaluate(
            r"""
            () => {
                const normalize = value => (value || '')
                    .replace(/\u00a0/g, ' ')
                    .replace(/\r/g, '\n')
                    .replace(/[ \t]+/g, ' ')
                    .replace(/\n[ \t]+/g, '\n')
                    .replace(/\n{3,}/g, '\n\n')
                    .trim();

                const policyHint = /privacy|legal|policy|data-protection|data_privacy/i;
                const policyText = /personal (data|information)|information we collect|data we collect|privacy rights|data protection|how we (use|share|collect|process)/i;

                const selector = [
                    'main',
                    'article',
                    '[role="main"]',
                    '.privacy-policy',
                    '.privacy',
                    '.legal',
                    '.legal-content',
                    '.policy',
                    '.policy-content',
                    '[class*="privacy"]',
                    '[class*="legal"]',
                    '[class*="policy"]',
                    '[id*="privacy"]',
                    '[id*="legal"]',
                    '[id*="policy"]'
                ].join(',');

                const candidates = [...document.querySelectorAll(selector)];
                let source = null;
                let bestScore = -1;

                for (const candidate of candidates) {
                    const text = normalize(candidate.textContent || candidate.innerText || '');
                    if (text.length < 250) continue;

                    const hint = [
                        candidate.id || '',
                        candidate.className || '',
                        candidate.getAttribute('role') || '',
                        candidate.tagName || ''
                    ].join(' ');

                    let score = Math.min(text.length, 60000);

                    if (policyHint.test(hint)) score += 50000;
                    if (policyText.test(text.slice(0, 20000))) score += 30000;
                    if (/^(MAIN|ARTICLE)$/.test(candidate.tagName)) score += 5000;
                    if (candidate.getAttribute('role') === 'main') score += 5000;

                    if (score > bestScore) {
                        bestScore = score;
                        source = candidate;
                    }
                }

                if (!source) source = document.body;

                if (!source) {
                    return {
                        title: document.title || '',
                        fullText: '',
                        items: [],
                        extractionMode: 'empty'
                    };
                }

                const root = source.cloneNode(true);

                root.querySelectorAll([
                    'script',
                    'style',
                    'noscript',
                    'svg',
                    'canvas',
                    'form',
                    'button',
                    'input',
                    'select',
                    'textarea',
                    'video',
                    'audio',
                    'nav',
                    '[role="navigation"]',
                    '[aria-hidden="true"]',
                    '[hidden]'
                ].join(',')).forEach(node => node.remove());

                const fullText = normalize(root.textContent || root.innerText || '');
                const blockSelector = 'h1,h2,h3,h4,h5,h6,p,li,dt,dd,blockquote,pre';
                const blocks = [...root.querySelectorAll(blockSelector)];
                const items = [];
                let heading = normalize(document.title) || 'Privacy Policy';

                for (const node of blocks) {
                    const text = normalize(node.textContent || node.innerText || '');
                    if (!text) continue;

                    if (/^H[1-6]$/.test(node.tagName)) {
                        heading = text.slice(0, 500);
                        continue;
                    }

                    if (node.querySelector(blockSelector)) continue;
                    if (text.length < 8) continue;

                    items.push({heading, text});
                }

                const semanticLength = items.reduce(
                    (total, item) => total + item.text.length,
                    0
                );

                if (
                    !items.length ||
                    semanticLength < Math.min(fullText.length * 0.55, 5000)
                ) {
                    return {
                        title: document.title || '',
                        fullText,
                        items: [{
                            heading: normalize(document.title) || 'Privacy Policy',
                            text: fullText
                        }],
                        semanticLength: fullText.length,
                        extractionMode: 'full_text_fallback'
                    };
                }

                return {
                    title: document.title || '',
                    fullText,
                    items,
                    semanticLength,
                    extractionMode: 'semantic'
                };
            }
            """
        )

    def _extract_raw_policy(self, page):
        extractions = []

        for frame in page.frames:
            try:
                if frame.url and frame.url != "about:blank":
                    parsed = urlparse(frame.url)
                    if parsed.scheme in {"http", "https"}:
                        self._assert_safe_url(frame.url)

                extracted = self._extract_from_frame(frame)
            except Exception:
                continue

            if not extracted:
                continue

            text = extracted.get("fullText") or ""
            if len(text) < 250:
                continue

            score = min(len(text), 100_000)
            title = extracted.get("title") or ""

            if POLICY_LINK_RE.search(title):
                score += 40_000
            if POLICY_TEXT_RE.search(text[:20_000]):
                score += 40_000
            if frame == page.main_frame:
                score += 5_000

            extracted["frame_url"] = frame.url
            extracted["selection_score"] = score
            extractions.append(extracted)

        if not extractions:
            return {
                "title": "",
                "fullText": "",
                "items": [],
                "semanticLength": 0,
                "extractionMode": "empty",
                "frame_url": "",
            }

        extractions.sort(key=lambda item: item["selection_score"], reverse=True)
        return extractions[0]

    def _split_text(self, text):
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.section_chunk_chars, len(text))

            if end < len(text):
                sentence = text.rfind(". ", start, end)
                question = text.rfind("? ", start, end)
                exclamation = text.rfind("! ", start, end)
                semicolon = text.rfind("; ", start, end)
                boundary = max(sentence, question, exclamation, semicolon)
                minimum_boundary = start + self.section_chunk_chars // 2

                if boundary > minimum_boundary:
                    end = boundary + 1

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end <= start:
                break

            start = end

        return chunks

    def _group_items(self, items):
        groups = []
        current_heading = None
        current_parts = []

        def flush():
            nonlocal current_heading, current_parts

            if not current_parts:
                return

            text = _plain_space(" ".join(current_parts))

            if text:
                groups.append({
                    "heading": current_heading or "Privacy Policy",
                    "text": text,
                })

            current_parts = []

        for item in items:
            if not isinstance(item, dict):
                continue

            heading = _plain_space(item.get("heading") or "Privacy Policy")[:500]
            text = _plain_space(item.get("text") or "")

            if not text:
                continue

            if current_heading is None:
                current_heading = heading

            if heading != current_heading:
                flush()
                current_heading = heading

            current_parts.append(text)

        flush()
        return groups

    def _build_sections(self, groups):
        sections = []
        section_limit_hit = False

        for group in groups:
            chunks = self._split_text(group["text"])
            total_parts = len(chunks)

            for part_number, chunk in enumerate(chunks, 1):
                if len(sections) >= self.max_sections:
                    section_limit_hit = True
                    break

                sections.append({
                    "heading": group["heading"],
                    "part": part_number,
                    "parts": total_parts,
                    "text": chunk,
                })

            if section_limit_hit:
                break

        return sections, section_limit_hit

    def _limit_document_sections(self, sections):
        if not self.max_document_chars:
            return sections, False

        output = []
        used_chars = 0
        truncated = False

        for section in sections:
            remaining = self.max_document_chars - used_chars

            if remaining <= 0:
                truncated = True
                break

            text = section["text"]

            if len(text) <= remaining:
                output.append(section.copy())
                used_chars += len(text)
                continue

            if remaining >= 500:
                shortened = section.copy()
                shortened["text"] = self._cut_text(text, remaining)
                shortened["document_truncated"] = True
                output.append(shortened)
                used_chars += len(shortened["text"])

            truncated = True
            break

        return output, truncated

    def _cut_text(self, text, limit):
        if len(text) <= limit:
            return text

        minimum_boundary = max(0, limit // 2)
        boundaries = (
            text.rfind(". ", minimum_boundary, limit),
            text.rfind("? ", minimum_boundary, limit),
            text.rfind("! ", minimum_boundary, limit),
            text.rfind("; ", minimum_boundary, limit),
        )
        boundary = max(boundaries)

        if boundary > 0:
            return text[:boundary + 1].strip()

        return text[:limit].strip()

    def _extract_sections(self, page):
        extracted = self._extract_raw_policy(page)
        raw_full_text = extracted.get("fullText") or ""
        original_chars = len(raw_full_text)
        items = extracted.get("items") or []

        if not raw_full_text or not items:
            return {
                "full_text": "",
                "sections": [],
                "original_chars": original_chars,
                "extracted_chars": 0,
                "uncapped_extracted_chars": 0,
                "coverage": 0,
                "complete": False,
                "document_truncated": False,
                "section_limit_hit": False,
                "extraction_mode": extracted.get("extractionMode") or "empty",
                "frame_url": extracted.get("frame_url") or "",
            }

        groups = self._group_items(items)
        uncapped_sections, section_limit_hit = self._build_sections(groups)
        uncapped_extracted_chars = sum(
            len(section["text"]) for section in uncapped_sections
        )

        sections, document_truncated = self._limit_document_sections(
            uncapped_sections
        )

        for index, section in enumerate(sections, 1):
            section["section_id"] = f"policy-{index:04d}"

        extracted_chars = sum(len(section["text"]) for section in sections)

        if extracted.get("extractionMode") == "full_text_fallback":
            coverage = 1.0 if original_chars else 0
        elif original_chars:
            coverage = min(uncapped_extracted_chars / original_chars, 1.0)
        else:
            coverage = 0

        complete = (
            sections
            and not document_truncated
            and not section_limit_hit
            and coverage >= 0.60
        )

        canonical_full_text = "\n\n".join(
            section["text"] for section in sections
        )

        return {
            "full_text": canonical_full_text,
            "sections": sections,
            "original_chars": original_chars,
            "extracted_chars": extracted_chars,
            "uncapped_extracted_chars": uncapped_extracted_chars,
            "coverage": round(coverage, 4),
            "complete": complete,
            "document_truncated": document_truncated,
            "section_limit_hit": section_limit_hit,
            "extraction_mode": extracted.get("extractionMode") or "unknown",
            "frame_url": extracted.get("frame_url") or "",
        }

    def _detect_categories(self, telemetry):
        categories = {"collection", "sharing"}

        if not telemetry:
            categories.update({"tracking", "device"})
            return sorted(categories)

        terms = " ".join(sorted(_active_telemetry_terms(telemetry)))

        if any(term in terms for term in (
            "cookie",
            "localstorage",
            "local storage",
            "sessionstorage",
            "session storage",
            "indexeddb",
            "cache storage",
            "storage",
        )):
            categories.add("cookies")

        if any(term in terms for term in (
            "tracker",
            "thirdparty",
            "third party",
            "analytics",
            "pixel",
            "beacon",
        )):
            categories.add("tracking")

        if any(term in terms for term in (
            "canvas",
            "webgl",
            "audio",
            "fingerprint",
            "useragent",
            "user agent",
        )):
            categories.add("fingerprinting")

        if any(term in terms for term in (
            "request",
            "host",
            "network",
            "ip",
            "third party host",
        )):
            categories.add("network")

        if any(term in terms for term in (
            "permission",
            "camera",
            "microphone",
            "sensor",
            "notification",
        )):
            categories.add("permissions")

        if any(term in terms for term in (
            "device",
            "screen",
            "browser",
            "useragent",
            "user agent",
            "webgl",
        )):
            categories.add("device")

        if any(term in terms for term in ("location", "geolocation")):
            categories.add("location")

        return sorted(categories)

    def _score_section(self, section, categories, position):
        heading = (section.get("heading") or "").lower()
        text = (section.get("text") or "").lower()
        score = 0

        if position == 0:
            score += 6
        elif position < 3:
            score += 2

        for category in categories:
            for term in CATEGORY_TERMS.get(category, ()):
                if term in heading:
                    score += 8
                if term in text:
                    score += 3

        if re.search(r"\bdefinitions?\b|\bscope\b|\babout this policy\b", heading):
            score += 4

        if "collect" in text:
            score += 1
        if "share" in text:
            score += 1
        if "third party" in text or "third-party" in text:
            score += 1
        if "cookie" in text:
            score += 1
        if "device" in text:
            score += 1

        return score

    def _select_model_sections(self, sections, telemetry=None):
        if not sections:
            return []

        categories = self._detect_categories(telemetry)
        scored = []

        for position, section in enumerate(sections):
            scored.append({
                "score": self._score_section(section, categories, position),
                "position": position,
                "section": section,
            })

        scored.sort(key=lambda item: (-item["score"], item["position"]))

        selected = []
        selected_ids = set()
        used_chars = 0

        def add_section(section):
            nonlocal used_chars

            section_id = section["section_id"]

            if section_id in selected_ids:
                return False

            if len(selected) >= self.max_model_sections:
                return False

            text_length = len(section["text"])

            if used_chars + text_length > self.max_model_policy_chars:
                return False

            selected.append(section.copy())
            selected_ids.add(section_id)
            used_chars += text_length
            return True

        # Always try to keep the opening scope/definitions context.
        add_section(sections[0])

        for item in scored:
            add_section(item["section"])

            if len(selected) >= self.max_model_sections:
                break

        # If the first section consumed too much of the budget, retry without
        # forcing it so relevant telemetry sections still reach the model.
        if len(selected) == 1 and len(sections) > 1:
            only = selected[0]
            if len(only["text"]) > self.max_model_policy_chars // 2:
                selected = []
                selected_ids = set()
                used_chars = 0

                for item in scored:
                    add_section(item["section"])
                    if len(selected) >= self.max_model_sections:
                        break

        if not selected:
            for section in sections:
                if add_section(section):
                    break

        position_map = {
            section["section_id"]: position
            for position, section in enumerate(sections)
        }

        selected.sort(
            key=lambda section: position_map.get(section["section_id"], 999999)
        )

        return selected

    def validate_policy_section_ids(self, findings, document):
        valid_ids = {
            section["section_id"]
            for section in document.get("sections", [])
            if section.get("section_id")
        }

        errors = []

        for finding_index, finding in enumerate(findings):
            for section_id in finding.get("policy_section_ids") or []:
                if section_id not in valid_ids:
                    errors.append({
                        "finding_index": finding_index,
                        "section_id": section_id,
                        "error": "unknown policy section ID",
                    })

        return {
            "valid": not errors,
            "errors": errors,
        }

    def _fetch_candidate(self, context, candidate, domain_url, telemetry=None):
        page = context.new_page()

        try:
            self._assert_safe_url(candidate["url"])

            response = page.goto(
                candidate["url"],
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )

            if response is None:
                return None, "navigation returned no response"

            if response.status >= 400:
                return None, f"HTTP {response.status}"

            final_url = self._assert_safe_url(page.url)
            content_type = (response.headers.get("content-type") or "").lower()

            if "application/pdf" in content_type:
                return None, "PDF policy extraction is not supported by this retriever"

            self._wait_for_document_stability(page)
            self._prime_document(page)
            self._wait_for_document_stability(page)

            title = _plain_space(page.title())

            try:
                body_text = _plain_space(
                    page.locator("body").inner_text(timeout=5_000)
                )
            except Exception:
                body_text = ""

            if not self._looks_like_policy(final_url, title, body_text):
                return None, "page did not verify as a privacy policy"

            if not self._is_applicable(
                domain_url,
                final_url,
                title,
                body_text,
            ):
                return None, "policy did not verify as applicable to the supplied domain"

            extracted = self._extract_sections(page)
            canonical_sections = extracted["sections"]

            if not canonical_sections:
                return None, "policy text could not be extracted"

            model_sections = self._select_model_sections(
                canonical_sections,
                telemetry,
            )

            if not model_sections:
                return None, "policy was extracted but no model context could be selected"

            limitations = []

            if extracted["document_truncated"]:
                limitations.append(
                    "The policy exceeded the host-side canonical document limit."
                )

            if extracted["section_limit_hit"]:
                limitations.append(
                    "The policy exceeded the maximum canonical section count."
                )

            if extracted["coverage"] < 0.60:
                limitations.append(
                    "The extracted structured text covered less than 60% of the selected policy container."
                )

            if len(model_sections) < len(canonical_sections):
                limitations.append(
                    "Only the policy sections most relevant to the observed visit were supplied to the model because of the Qwen context budget."
                )

            model_chars = sum(len(section["text"]) for section in model_sections)

            return {
                "url": final_url,
                "found": True,
                "applicable": True,
                "complete": extracted["complete"],
                "title": title,
                "retrieved_at": dt.datetime.now(dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "retrieval_method": candidate["method"],

                # Backward-compatible model-ready sections.
                "sections": model_sections,

                "document_hash": _document_hash(extracted["full_text"]),
                "document": {
                    "full_text": extracted["full_text"],
                    "original_chars": extracted["original_chars"],
                    "extracted_chars": extracted["extracted_chars"],
                    "uncapped_extracted_chars": extracted[
                        "uncapped_extracted_chars"
                    ],
                    "coverage": extracted["coverage"],
                    "extraction_mode": extracted["extraction_mode"],
                    "frame_url": extracted["frame_url"],
                    "sections": canonical_sections,
                },
                "model_context": {
                    "max_policy_chars": self.max_model_policy_chars,
                    "max_sections": self.max_model_sections,
                    "selected_chars": model_chars,
                    "selected_sections": len(model_sections),
                    "categories": self._detect_categories(telemetry),
                    "sections": model_sections,
                },
                "limitations": limitations,
            }, None

        except Exception as exc:
            return None, f"{exc.__class__.__name__}: {exc}"

        finally:
            page.close()

    def _discover_domain_links(self, context, domain_url):
        page = context.new_page()

        try:
            target = self._assert_safe_url(_origin(domain_url) + "/")
            response = page.goto(
                target,
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )

            if response is None or response.status >= 400:
                return []

            self._wait_for_document_stability(page)
            self._assert_safe_url(page.url)

            links = page.eval_on_selector_all(
                "a[href]",
                """
                els => els.map(a => ({
                    href: a.href,
                    text: (a.innerText || a.textContent || '').trim()
                }))
                """,
            )

            ranked = []

            for link in links:
                if not isinstance(link, dict):
                    continue

                href = link.get("href")
                text = link.get("text") or ""

                if not href or not POLICY_LINK_RE.search(f"{text} {href}"):
                    continue

                try:
                    href = self._assert_safe_url(href)
                except ValueError:
                    continue

                score = 0

                if _same_site(domain_url, href):
                    score += 3
                if POLICY_LINK_RE.search(text):
                    score += 3
                if "privacy" in href.lower():
                    score += 2
                if "/legal/" in href.lower():
                    score += 1

                ranked.append((score, href))

            ranked.sort(key=lambda item: (-item[0], item[1]))

            return [
                {"url": url, "method": "domain_link"}
                for _, url in ranked[:8]
            ]

        except Exception:
            return []

        finally:
            page.close()

    def _search_candidates(self, context, domain_url):
        if not self.search_enabled:
            return []

        page = context.new_page()

        try:
            hostname = _host(domain_url)
            tokens = _service_tokens(domain_url)
            service_name = tokens[0] if tokens else hostname
            candidates = []

            queries = (
                f'site:{hostname} "privacy policy" OR "privacy notice"',
                f'"{service_name}" "privacy policy" OR "privacy notice"',
            )

            for raw_query in queries:
                search_url = self._assert_safe_url(
                    "https://html.duckduckgo.com/html/?q="
                    + quote_plus(raw_query)
                )

                response = page.goto(
                    search_url,
                    wait_until="domcontentloaded",
                    timeout=self.timeout_ms,
                )

                if response is None or response.status >= 400:
                    continue

                results = page.eval_on_selector_all(
                    "a.result__a, a[data-testid='result-title-a']",
                    """
                    els => els.map(a => ({
                        href: a.href,
                        text: (a.innerText || a.textContent || '').trim()
                    }))
                    """,
                )

                for result in results:
                    if not isinstance(result, dict):
                        continue

                    href = _unwrap_search_url(result.get("href") or "")
                    text = result.get("text") or ""

                    if not POLICY_LINK_RE.search(f"{text} {href}"):
                        continue

                    try:
                        href = self._assert_safe_url(href)
                    except ValueError:
                        continue

                    candidates.append({
                        "url": href,
                        "method": "web_search",
                    })

                    if len(candidates) >= 8:
                        break

                if len(candidates) >= 8:
                    break

            return candidates

        except Exception:
            return []

        finally:
            page.close()

    def _dedupe(self, candidates):
        output = []
        seen = set()

        for candidate in candidates:
            key = _candidate_key(candidate["url"])

            if key in seen:
                continue

            seen.add(key)
            output.append(candidate)

        return output

    def retrieve(self, domain_url, privacy_policy_url=None, telemetry=None):
        """Retrieve, verify, extract, and prepare the applicable policy.

        The top-level `sections` field remains model-ready for compatibility.
        The complete canonical extraction is stored under `document`.
        """

        domain_url = self._assert_safe_url(domain_url)
        supplied = None
        supplied_failed = privacy_policy_url not in {None, ""}

        if privacy_policy_url:
            try:
                supplied = self._assert_safe_url(privacy_policy_url)
            except (TypeError, ValueError):
                supplied = None

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Run `pip install playwright` and "
                "`playwright install chromium`."
            ) from exc

        attempts = []

        with sync_playwright() as playwright:
            browser_type = getattr(playwright, self.browser_name, None)

            if browser_type is None:
                raise ValueError(
                    f"unsupported browser {self.browser_name!r}; "
                    "use chromium, firefox, or webkit"
                )

            browser = browser_type.launch(headless=self.headless)
            context = browser.new_context(
                service_workers="block",
                java_script_enabled=True,
                locale="en-US",
            )

            context.set_default_navigation_timeout(self.timeout_ms)
            context.set_default_timeout(min(self.timeout_ms, 5_000))
            context.route("**/*", self._route)

            try:
                if supplied:
                    candidate = {
                        "url": supplied,
                        "method": "supplied_url",
                    }

                    document, error = self._fetch_candidate(
                        context,
                        candidate,
                        domain_url,
                        telemetry,
                    )

                    if document:
                        return document

                    attempts.append({
                        "url": candidate["url"],
                        "method": candidate["method"],
                        "error": error or "unknown retrieval error",
                    })

                candidates = self._discover_domain_links(context, domain_url)
                origin = _origin(domain_url)

                for path in COMMON_POLICY_PATHS:
                    candidates.append({
                        "url": urljoin(origin + "/", path),
                        "method": "common_path",
                    })

                candidates.extend(
                    self._search_candidates(context, domain_url)
                )

                candidates = self._dedupe(candidates)[:self.max_candidates]

                for candidate in candidates:
                    if supplied and _candidate_key(candidate["url"]) == _candidate_key(supplied):
                        continue

                    document, error = self._fetch_candidate(
                        context,
                        candidate,
                        domain_url,
                        telemetry,
                    )

                    if document:
                        if supplied_failed:
                            document["limitations"].append(
                                "The supplied privacy-policy URL could not be verified; policy discovery located the applicable policy used for this comparison."
                            )

                        return document

                    attempts.append({
                        "url": candidate["url"],
                        "method": candidate["method"],
                        "error": error or "unknown retrieval error",
                    })

            finally:
                context.close()
                browser.close()

        limitation = (
            "No applicable privacy policy could be retrieved or verified for the supplied domain."
        )

        if supplied_failed:
            limitation += " The supplied privacy-policy URL could not be used."

        return {
            "url": "",
            "found": False,
            "applicable": False,
            "complete": False,
            "title": "",
            "retrieved_at": dt.datetime.now(dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "retrieval_method": "none",
            "sections": [],
            "document_hash": "",
            "document": {
                "full_text": "",
                "original_chars": 0,
                "extracted_chars": 0,
                "uncapped_extracted_chars": 0,
                "coverage": 0,
                "extraction_mode": "none",
                "frame_url": "",
                "sections": [],
            },
            "model_context": {
                "max_policy_chars": self.max_model_policy_chars,
                "max_sections": self.max_model_sections,
                "selected_chars": 0,
                "selected_sections": 0,
                "categories": self._detect_categories(telemetry),
                "sections": [],
            },
            "limitations": [limitation],
            "attempted_candidates": len(attempts),
            "attempts": attempts,
        }
