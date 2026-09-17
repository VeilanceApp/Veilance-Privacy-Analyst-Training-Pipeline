"""Safe, host-orchestrated privacy-policy retrieval with Playwright.

The model never launches a browser.

The host:

1. Retrieves and verifies the privacy policy.
2. Extracts a canonical copy of the policy.
3. Splits the policy into stable host-generated sections.
4. Selects only the most relevant sections for Qwen.
5. Supplies those bounded sections to the model.

Search is used only to discover candidate URLs.
Search snippets are never treated as policy evidence.

Important architecture:

    Website
        |
        v
    Playwright
        |
        v
    Full canonical policy
        |
        +--------------------------+
        |                          |
        v                          v
    Stored full text          Stable sections
                                  |
                                  v
                           Relevance scoring
                                  |
                                  v
                         Qwen context subset

`max_document_chars` controls how much policy text the host will preserve.

`max_model_policy_chars` controls how much policy text is supplied to Qwen.

These are deliberately separate.
"""

import datetime as dt
import hashlib
import ipaddress
import re
import socket

from urllib.parse import (
    parse_qs,
    quote_plus,
    unquote,
    urljoin,
    urlparse,
)


# ---------------------------------------------------------------------------
# Policy identification
# ---------------------------------------------------------------------------

POLICY_LINK_RE = re.compile(
    r"(?:"
    r"privacy(?:\s+(?:policy|notice|statement))?"
    r"|data\s+(?:policy|privacy|protection)"
    r"|privacy\s+center"
    r")",
    re.IGNORECASE,
)

POLICY_TEXT_RE = re.compile(
    r"(?:"
    r"personal (?:data|information)"
    r"|information we collect"
    r"|data we collect"
    r"|information we receive"
    r"|how we (?:use|share|collect|process)"
    r"|your privacy"
    r"|privacy rights"
    r"|data protection"
    r"|your information"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Hostname helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Common policy locations
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Terms used for model-context relevance selection
# ---------------------------------------------------------------------------

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
        "track",
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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _plain_space(value):
    if not value:
        return ""

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def _origin(url):
    parsed = urlparse(url)

    return (
        parsed.scheme
        + "://"
        + parsed.netloc
    )


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

    if left_host == right_host:
        return True

    if left_host.endswith("." + right_host):
        return True

    if right_host.endswith("." + left_host):
        return True

    return False


def _service_tokens(domain_url):
    labels = []

    for label in _host(domain_url).split("."):
        if label in GENERIC_HOST_LABELS:
            continue

        if len(label) < 3:
            continue

        labels.append(label)

    labels.sort(
        key=len,
        reverse=True,
    )

    return labels[:3]


def _unwrap_search_url(href):
    if href.startswith("//"):
        href = "https:" + href

    parsed = urlparse(href)

    if "duckduckgo.com" in (parsed.hostname or ""):
        target = parse_qs(
            parsed.query
        ).get(
            "uddg",
            [""],
        )[0]

        if target:
            return unquote(target)

    return href


def _candidate_key(url):
    parsed = urlparse(url)

    path = parsed.path.rstrip("/")

    if not path:
        path = "/"

    return (
        parsed.scheme.lower()
        + "://"
        + parsed.netloc.lower()
        + path
    )


def _normalize_search_text(value):
    value = value.lower()

    value = re.sub(
        r"[^a-z0-9\s-]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def _document_hash(text):
    return hashlib.sha256(
        text.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------

class Candidate:
    def __init__(self, url, method):
        self.url = url
        self.method = method


# ---------------------------------------------------------------------------
# Policy retriever
# ---------------------------------------------------------------------------

class PolicyRetriever:
    def __init__(
        self,
        timeout_ms=20_000,

        # Maximum amount preserved from the actual policy.
        max_document_chars=250_000,

        # Host-side section chunk size.
        section_chunk_chars=8_000,

        # Safety ceiling for absurdly fragmented policies.
        max_sections=256,

        # Qwen-specific context limits.
        max_model_policy_chars=16_000,
        max_model_sections=12,

        # Discovery.
        max_candidates=14,
        search_enabled=True,

        # Browser.
        browser_name="chromium",
        headless=True,

        # SSRF/network protection.
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

    # -----------------------------------------------------------------------
    # URL safety
    # -----------------------------------------------------------------------

    def _assert_safe_url(self, url):
        if not isinstance(url, str):
            raise ValueError(
                "URL must be a string"
            )

        if not url.strip():
            raise ValueError(
                "URL is empty"
            )

        url = url.strip()

        parsed = urlparse(url)

        if parsed.scheme not in {
            "http",
            "https",
        }:
            raise ValueError(
                "only HTTP(S) URLs are allowed"
            )

        if not parsed.hostname:
            raise ValueError(
                "URL does not contain a hostname"
            )

        if parsed.username or parsed.password:
            raise ValueError(
                "URLs containing credentials are not allowed"
            )

        try:
            if parsed.port:
                port = parsed.port
            elif parsed.scheme == "https":
                port = 443
            else:
                port = 80

        except ValueError as exc:
            raise ValueError(
                "URL contains an invalid port"
            ) from exc

        if port not in {
            80,
            443,
        }:
            raise ValueError(
                "only ports 80 and 443 are allowed"
            )

        hostname = parsed.hostname.lower().rstrip(".")

        if self.allow_private_network:
            return url

        if hostname in {
            "localhost",
            "localhost.localdomain",
        }:
            raise ValueError(
                "local hosts are not allowed"
            )

        if hostname.endswith(
            (
                ".local",
                ".internal",
                ".localhost",
            )
        ):
            raise ValueError(
                "local or internal hosts are not allowed"
            )

        cache_key = (
            hostname,
            port,
        )

        addresses = self._dns_cache.get(
            cache_key
        )

        if addresses is None:
            try:
                info = socket.getaddrinfo(
                    hostname,
                    port,
                    type=socket.SOCK_STREAM,
                )

            except socket.gaierror as exc:
                raise ValueError(
                    "host did not resolve: "
                    + hostname
                ) from exc

            addresses = set()

            for row in info:
                addresses.add(
                    row[4][0]
                )

            addresses = sorted(
                addresses
            )

            if not addresses:
                raise ValueError(
                    "host did not resolve: "
                    + hostname
                )

            self._dns_cache[
                cache_key
            ] = addresses

        for address in addresses:
            ip = ipaddress.ip_address(
                address
            )

            if not ip.is_global:
                raise ValueError(
                    "host resolves to a non-public "
                    "address: "
                    + hostname
                )

        return url

    # -----------------------------------------------------------------------
    # Request routing
    # -----------------------------------------------------------------------

    def _route(self, route):
        request = route.request

        parsed = urlparse(
            request.url
        )

        if parsed.scheme not in {
            "http",
            "https",
        }:
            if parsed.scheme in {
                "data",
                "blob",
                "about",
            }:
                route.continue_()
            else:
                route.abort()

            return

        try:
            self._assert_safe_url(
                request.url
            )

        except ValueError:
            route.abort()
            return

        if request.resource_type in {
            "image",
            "media",
            "font",
        }:
            route.abort()
            return

        route.continue_()

    # -----------------------------------------------------------------------
    # Policy verification
    # -----------------------------------------------------------------------

    def _looks_like_policy(
        self,
        url,
        title,
        body_text,
    ):
        title = _plain_space(
            title
        )

        body_text = _plain_space(
            body_text
        )

        if len(body_text) < 250:
            return False

        score = 0

        if POLICY_LINK_RE.search(
            url
        ):
            score += 2

        if POLICY_LINK_RE.search(
            title
        ):
            score += 4

        if POLICY_LINK_RE.search(
            body_text[:6_000]
        ):
            score += 2

        if POLICY_TEXT_RE.search(
            body_text[:20_000]
        ):
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

        if cookie_only:
            return False

        return score >= 5

    def _is_applicable(
        self,
        domain_url,
        policy_url,
        method,
        title,
        body_text,
    ):
        if _same_site(
            domain_url,
            policy_url,
        ):
            return True

        haystack = (
            title
            + " "
            + body_text[:30_000]
        ).lower()

        haystack = re.sub(
            r"[^a-z0-9]+",
            "",
            haystack,
        )

        tokens = _service_tokens(
            domain_url
        )

        if not tokens:
            return False

        for token in tokens:
            cleaned = re.sub(
                r"[^a-z0-9]+",
                "",
                token.lower(),
            )

            if cleaned and cleaned in haystack:
                return True

        return False

    # -----------------------------------------------------------------------
    # Dynamic content stabilization
    # -----------------------------------------------------------------------

    def _wait_for_document_stability(
        self,
        page,
    ):
        previous_length = -1
        stable_reads = 0

        for _ in range(14):
            try:
                current_length = page.evaluate(
                    """
                    () => {
                        if (!document.body) {
                            return 0;
                        }

                        return (
                            document.body.innerText ||
                            document.body.textContent ||
                            ''
                        ).length;
                    }
                    """
                )

            except Exception:
                current_length = 0

            if (
                current_length >= 500
                and previous_length >= 0
            ):
                difference = abs(
                    current_length
                    - previous_length
                )

                tolerance = max(
                    25,
                    current_length // 100,
                )

                if difference <= tolerance:
                    stable_reads += 1
                else:
                    stable_reads = 0

                if stable_reads >= 2:
                    return

            previous_length = (
                current_length
            )

            page.wait_for_timeout(
                400
            )

    # -----------------------------------------------------------------------
    # Raw policy extraction
    # -----------------------------------------------------------------------

    def _extract_raw_policy(
        self,
        page,
    ):
        return page.evaluate(
            r"""
            () => {
                const normalize = value => {
                    return (value || '')
                        .replace(/\u00a0/g, ' ')
                        .replace(/\r/g, '\n')
                        .replace(/[ \t]+/g, ' ')
                        .replace(/\n[ \t]+/g, '\n')
                        .replace(/\n{3,}/g, '\n\n')
                        .trim();
                };

                const candidateSelector = [
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

                const candidates = [
                    ...document.querySelectorAll(
                        candidateSelector
                    )
                ];

                let source = null;
                let sourceLength = 0;

                /*
                 * Do not blindly use the first <main> or <article>.
                 * Choose the largest plausible legal-content region.
                 */
                for (const candidate of candidates) {
                    const text = normalize(
                        candidate.innerText ||
                        candidate.textContent ||
                        ''
                    );

                    if (text.length > sourceLength) {
                        source = candidate;
                        sourceLength = text.length;
                    }
                }

                if (!source) {
                    source = document.body;
                }

                if (!source) {
                    return {
                        fullText: '',
                        items: [],
                        originalLength: 0,
                        semanticLength: 0,
                        extractionMode: 'empty'
                    };
                }

                const root = source.cloneNode(true);

                /*
                 * Remove non-policy UI/content.
                 *
                 * We intentionally do not blanket-remove header/footer
                 * because some sites place legal text inside unusual
                 * structures.
                 */
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
                    '[aria-hidden="true"]',
                    '[hidden]'
                ].join(',')).forEach(node => node.remove());

                root.querySelectorAll(
                    'nav,[role="navigation"]'
                ).forEach(node => node.remove());

                const fullText = normalize(
                    root.innerText ||
                    root.textContent ||
                    ''
                );

                const blockSelector = [
                    'h1',
                    'h2',
                    'h3',
                    'h4',
                    'h5',
                    'h6',
                    'p',
                    'li',
                    'dt',
                    'dd',
                    'blockquote',
                    'pre'
                ].join(',');

                const blocks = [
                    ...root.querySelectorAll(
                        blockSelector
                    )
                ];

                const items = [];

                let heading =
                    normalize(document.title) ||
                    'Privacy Policy';

                for (const node of blocks) {
                    const text = normalize(
                        node.innerText ||
                        node.textContent ||
                        ''
                    );

                    if (!text) {
                        continue;
                    }

                    if (/^H[1-6]$/.test(node.tagName)) {
                        heading = text.slice(
                            0,
                            500
                        );

                        continue;
                    }

                    if (text.length < 8) {
                        continue;
                    }

                    items.push({
                        heading,
                        text
                    });
                }

                let semanticLength = 0;

                for (const item of items) {
                    semanticLength += item.text.length;
                }

                /*
                 * Modern sites often use nested divs instead of semantic
                 * <p>/<li> markup. If semantic extraction captures too
                 * little of the visible document, fall back to the entire
                 * visible legal container.
                 */
                if (
                    !items.length ||
                    semanticLength <
                        Math.min(
                            fullText.length * 0.55,
                            5000
                        )
                ) {
                    return {
                        fullText,
                        items: [{
                            heading:
                                normalize(document.title) ||
                                'Privacy Policy',
                            text: fullText
                        }],
                        originalLength: fullText.length,
                        semanticLength,
                        extractionMode: 'full_text_fallback'
                    };
                }

                return {
                    fullText,
                    items,
                    originalLength: fullText.length,
                    semanticLength,
                    extractionMode: 'semantic'
                };
            }
            """
        )

    # -----------------------------------------------------------------------
    # Section chunking
    # -----------------------------------------------------------------------

    def _split_text(
        self,
        text,
    ):
        chunks = []

        start = 0

        while start < len(text):
            end = min(
                start + self.section_chunk_chars,
                len(text),
            )

            if end < len(text):
                sentence = text.rfind(
                    ". ",
                    start,
                    end,
                )

                paragraph = text.rfind(
                    "\n",
                    start,
                    end,
                )

                semicolon = text.rfind(
                    "; ",
                    start,
                    end,
                )

                boundary = max(
                    sentence,
                    paragraph,
                    semicolon,
                )

                minimum_boundary = (
                    start
                    + self.section_chunk_chars // 2
                )

                if boundary > minimum_boundary:
                    end = boundary + 1

            chunk = text[
                start:end
            ].strip()

            if chunk:
                chunks.append(
                    chunk
                )

            start = end

        return chunks

    def _extract_sections(
        self,
        page,
    ):
        extracted = self._extract_raw_policy(
            page
        )

        if not extracted:
            return {
                "full_text": "",
                "sections": [],
                "original_chars": 0,
                "extracted_chars": 0,
                "coverage": 0,
                "complete": False,
                "extraction_mode": "empty",
            }

        full_text = (
            extracted.get("fullText")
            or ""
        )

        original_chars = len(
            full_text
        )

        document_truncated = False

        if (
            self.max_document_chars
            and len(full_text)
            > self.max_document_chars
        ):
            full_text = full_text[
                :self.max_document_chars
            ]

            document_truncated = True

        items = (
            extracted.get("items")
            or []
        )

        grouped = []

        current_heading = None
        current_parts = []

        def flush():
            nonlocal current_heading
            nonlocal current_parts

            if not current_parts:
                return

            text = _plain_space(
                " ".join(
                    current_parts
                )
            )

            if text:
                grouped.append({
                    "heading":
                        current_heading
                        or "Privacy Policy",
                    "text": text,
                })

            current_parts = []

        for item in items:
            if not isinstance(
                item,
                dict,
            ):
                continue

            heading = _plain_space(
                item.get("heading")
                or "Privacy Policy"
            )

            heading = heading[:500]

            text = _plain_space(
                item.get("text")
                or ""
            )

            if not text:
                continue

            if current_heading is None:
                current_heading = heading

            if heading != current_heading:
                flush()
                current_heading = heading

            current_parts.append(
                text
            )

        flush()

        sections = []

        for group in grouped:
            chunks = self._split_text(
                group["text"]
            )

            total_parts = len(
                chunks
            )

            for part_number, chunk in enumerate(
                chunks,
                1,
            ):
                sections.append({
                    "heading":
                        group["heading"],
                    "part":
                        part_number,
                    "parts":
                        total_parts,
                    "text":
                        chunk,
                })

                if (
                    len(sections)
                    >= self.max_sections
                ):
                    break

            if (
                len(sections)
                >= self.max_sections
            ):
                break

        /*
        Python does not allow JS-style comments here.
        This line is intentionally kept as normal Python below.
        */

        if not sections and full_text:
            chunks = self._split_text(
                full_text
            )

            for part_number, chunk in enumerate(
                chunks,
                1,
            ):
                sections.append({
                    "heading":
                        "Privacy Policy",
                    "part":
                        part_number,
                    "parts":
                        len(chunks),
                    "text":
                        chunk,
                })

                if (
                    len(sections)
                    >= self.max_sections
                ):
                    break

        for index, section in enumerate(
            sections,
            1,
        ):
            section["section_id"] = (
                "policy-"
                + str(index).zfill(4)
            )

        extracted_chars = 0

        for section in sections:
            extracted_chars += len(
                section["text"]
            )

        if original_chars:
            coverage = (
                extracted_chars
                / original_chars
            )
        else:
            coverage = 0

        complete = (
            not document_truncated
            and coverage >= 0.80
        )

        return {
            "full_text":
                full_text,
            "sections":
                sections,
            "original_chars":
                original_chars,
            "extracted_chars":
                extracted_chars,
            "coverage":
                round(
                    coverage,
                    4,
                ),
            "complete":
                complete,
            "document_truncated":
                document_truncated,
            "extraction_mode":
                extracted.get(
                    "extractionMode"
                )
                or "unknown",
        }

    # -----------------------------------------------------------------------
    # Telemetry category detection
    # -----------------------------------------------------------------------

    def _detect_categories(
        self,
        telemetry,
    ):
        if not telemetry:
            return [
                "collection",
                "sharing",
                "tracking",
                "device",
            ]

        text = repr(
            telemetry
        ).lower()

        categories = set()

        if (
            "cookie" in text
            or "localstorage" in text
            or "local storage" in text
            or "sessionstorage" in text
            or "indexeddb" in text
            or "cache" in text
        ):
            categories.add(
                "cookies"
            )

        if (
            "tracker" in text
            or "thirdparty" in text
            or "third_party" in text
            or "third-party" in text
            or "analytics" in text
        ):
            categories.add(
                "tracking"
            )

        if (
            "canvas" in text
            or "webgl" in text
            or "audio" in text
            or "fingerprint" in text
            or "useragent" in text
            or "user agent" in text
        ):
            categories.add(
                "fingerprinting"
            )

        if (
            "request" in text
            or "host" in text
            or "network" in text
            or "ip" in text
        ):
            categories.add(
                "network"
            )

        if (
            "permission" in text
            or "camera" in text
            or "microphone" in text
            or "sensor" in text
        ):
            categories.add(
                "permissions"
            )

        if (
            "device" in text
            or "screen" in text
            or "browser" in text
            or "useragent" in text
            or "user agent" in text
        ):
            categories.add(
                "device"
            )

        if (
            "location" in text
            or "geolocation" in text
        ):
            categories.add(
                "location"
            )

        categories.add(
            "collection"
        )

        categories.add(
            "sharing"
        )

        return list(
            categories
        )

    # -----------------------------------------------------------------------
    # Qwen context selection
    # -----------------------------------------------------------------------

    def _score_section(
        self,
        section,
        categories,
        position,
    ):
        heading = (
            section.get("heading")
            or ""
        ).lower()

        text = (
            section.get("text")
            or ""
        ).lower()

        haystack = (
            heading
            + " "
            + text
        )

        score = 0

        # Early sections often contain definitions/scope.
        if position < 3:
            score += 2

        for category in categories:
            terms = CATEGORY_TERMS.get(
                category,
                (),
            )

            for term in terms:
                if term in heading:
                    score += 5

                if term in text:
                    score += 2

        # Generic privacy relevance.
        if "collect" in haystack:
            score += 1

        if "share" in haystack:
            score += 1

        if "third party" in haystack:
            score += 1

        if "third-party" in haystack:
            score += 1

        if "cookie" in haystack:
            score += 1

        if "device" in haystack:
            score += 1

        if "information" in haystack:
            score += 1

        return score

    def _select_model_sections(
        self,
        sections,
        telemetry=None,
    ):
        if not sections:
            return []

        categories = self._detect_categories(
            telemetry
        )

        scored = []

        for position, section in enumerate(
            sections
        ):
            score = self._score_section(
                section,
                categories,
                position,
            )

            scored.append({
                "score":
                    score,
                "position":
                    position,
                "section":
                    section,
            })

        scored.sort(
            key=lambda item: (
                -item["score"],
                item["position"],
            )
        )

        selected = []
        selected_ids = set()

        used_chars = 0

        for item in scored:
            if (
                len(selected)
                >= self.max_model_sections
            ):
                break

            section = item[
                "section"
            ]

            section_id = section[
                "section_id"
            ]

            if section_id in selected_ids:
                continue

            remaining = (
                self.max_model_policy_chars
                - used_chars
            )

            if remaining <= 0:
                break

            text = section[
                "text"
            ]

            if len(text) > remaining:
                if remaining < 500:
                    continue

                selected.append({
                    "section_id":
                        section_id,
                    "heading":
                        section["heading"],
                    "part":
                        section.get("part"),
                    "parts":
                        section.get("parts"),
                    "text":
                        text[:remaining],
                    "context_truncated":
                        True,
                })

                selected_ids.add(
                    section_id
                )

                used_chars += remaining

                break

            selected.append({
                "section_id":
                    section_id,
                "heading":
                    section["heading"],
                "part":
                    section.get("part"),
                "parts":
                    section.get("parts"),
                "text":
                    text,
                "context_truncated":
                    False,
            })

            selected_ids.add(
                section_id
            )

            used_chars += len(
                text
            )

        # If relevance scoring somehow selects nothing,
        # send the beginning of the policy.
        if not selected:
            used_chars = 0

            for section in sections:
                if (
                    len(selected)
                    >= self.max_model_sections
                ):
                    break

                remaining = (
                    self.max_model_policy_chars
                    - used_chars
                )

                if remaining <= 0:
                    break

                text = section[
                    "text"
                ]

                truncated = False

                if len(text) > remaining:
                    text = text[
                        :remaining
                    ]

                    truncated = True

                selected.append({
                    "section_id":
                        section["section_id"],
                    "heading":
                        section["heading"],
                    "part":
                        section.get("part"),
                    "parts":
                        section.get("parts"),
                    "text":
                        text,
                    "context_truncated":
                        truncated,
                })

                used_chars += len(
                    text
                )

        # Return sections in original policy order.
        position_map = {}

        for position, section in enumerate(
            sections
        ):
            position_map[
                section["section_id"]
            ] = position

        selected.sort(
            key=lambda section:
                position_map.get(
                    section["section_id"],
                    999999,
                )
        )

        return selected

    # -----------------------------------------------------------------------
    # Citation validation
    # -----------------------------------------------------------------------

    def validate_policy_section_ids(
        self,
        findings,
        document,
    ):
        sections = (
            document.get("sections")
            or []
        )

        valid_ids = set()

        for section in sections:
            section_id = section.get(
                "section_id"
            )

            if section_id:
                valid_ids.add(
                    section_id
                )

        errors = []

        for finding_index, finding in enumerate(
            findings
        ):
            cited_ids = (
                finding.get(
                    "policy_section_ids"
                )
                or []
            )

            for section_id in cited_ids:
                if section_id not in valid_ids:
                    errors.append({
                        "finding_index":
                            finding_index,
                        "section_id":
                            section_id,
                        "error":
                            "unknown policy section ID",
                    })

        return {
            "valid":
                not errors,
            "errors":
                errors,
        }

    # -----------------------------------------------------------------------
    # Candidate retrieval
    # -----------------------------------------------------------------------

    def _fetch_candidate(
        self,
        context,
        candidate,
        domain_url,
        telemetry=None,
    ):
        page = context.new_page()

        try:
            self._assert_safe_url(
                candidate.url
            )

            response = page.goto(
                candidate.url,
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )

            if response is None:
                return (
                    None,
                    "navigation returned no response",
                )

            if response.status >= 400:
                return (
                    None,
                    "HTTP "
                    + str(response.status),
                )

            final_url = self._assert_safe_url(
                page.url
            )

            content_type = (
                response.headers.get(
                    "content-type"
                )
                or ""
            ).lower()

            if "application/pdf" in content_type:
                return (
                    None,
                    "PDF policy extraction is not supported by this retriever",
                )

            self._wait_for_document_stability(
                page
            )

            title = _plain_space(
                page.title()
            )

            try:
                body_text = _plain_space(
                    page.locator(
                        "body"
                    ).inner_text(
                        timeout=5_000
                    )
                )

            except Exception:
                body_text = ""

            if not self._looks_like_policy(
                final_url,
                title,
                body_text,
            ):
                return (
                    None,
                    "page did not verify as a privacy policy",
                )

            if not self._is_applicable(
                domain_url=domain_url,
                policy_url=final_url,
                method=candidate.method,
                title=title,
                body_text=body_text,
            ):
                return (
                    None,
                    "policy did not verify as applicable to the supplied domain",
                )

            extracted = self._extract_sections(
                page
            )

            sections = extracted[
                "sections"
            ]

            if not sections:
                return (
                    None,
                    "policy text could not be extracted",
                )

            model_sections = (
                self._select_model_sections(
                    sections,
                    telemetry,
                )
            )

            limitations = []

            if extracted[
                "document_truncated"
            ]:
                limitations.append(
                    "The retrieved policy exceeded the host document limit and was truncated before storage."
                )

            if not extracted[
                "complete"
            ]:
                limitations.append(
                    "Policy extraction coverage was below the preferred completeness threshold."
                )

            if any(
                section.get(
                    "context_truncated"
                )
                for section in model_sections
            ):
                limitations.append(
                    "At least one policy section was shortened to fit the model context budget."
                )

            model_chars = 0

            for section in model_sections:
                model_chars += len(
                    section["text"]
                )

            return ({
                "url":
                    final_url,
                "found":
                    True,
                "applicable":
                    True,
                "complete":
                    extracted["complete"],
                "title":
                    title,
                "retrieved_at":
                    dt.datetime.now(
                        dt.timezone.utc
                    ).isoformat().replace(
                        "+00:00",
                        "Z",
                    ),
                "retrieval_method":
                    candidate.method,

                "document_hash":
                    _document_hash(
                        extracted["full_text"]
                    ),

                "document": {
                    "full_text":
                        extracted["full_text"],
                    "original_chars":
                        extracted["original_chars"],
                    "extracted_chars":
                        extracted["extracted_chars"],
                    "coverage":
                        extracted["coverage"],
                    "extraction_mode":
                        extracted["extraction_mode"],
                    "sections":
                        sections,
                },

                "model_context": {
                    "max_policy_chars":
                        self.max_model_policy_chars,
                    "max_sections":
                        self.max_model_sections,
                    "selected_chars":
                        model_chars,
                    "selected_sections":
                        len(model_sections),
                    "sections":
                        model_sections,
                },

                "limitations":
                    limitations,
            }, None)

        except Exception as exc:
            return (
                None,
                exc.__class__.__name__
                + ": "
                + str(exc),
            )

        finally:
            page.close()

    # -----------------------------------------------------------------------
    # Domain-link discovery
    # -----------------------------------------------------------------------

    def _discover_domain_links(
        self,
        context,
        domain_url,
    ):
        page = context.new_page()

        try:
            target = self._assert_safe_url(
                _origin(
                    domain_url
                )
                + "/"
            )

            response = page.goto(
                target,
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )

            if (
                response is None
                or response.status >= 400
            ):
                return []

            self._wait_for_document_stability(
                page
            )

            self._assert_safe_url(
                page.url
            )

            links = page.eval_on_selector_all(
                "a[href]",
                """
                els => els.map(a => ({
                    href: a.href,
                    text: (
                        a.innerText ||
                        a.textContent ||
                        ''
                    ).trim()
                }))
                """,
            )

            ranked = []

            for link in links:
                if not isinstance(
                    link,
                    dict,
                ):
                    continue

                href = link.get(
                    "href"
                )

                text = (
                    link.get("text")
                    or ""
                )

                if not href:
                    continue

                if not POLICY_LINK_RE.search(
                    text + " " + href
                ):
                    continue

                try:
                    href = self._assert_safe_url(
                        href
                    )

                except ValueError:
                    continue

                score = 0

                if _same_site(
                    domain_url,
                    href,
                ):
                    score += 3

                if POLICY_LINK_RE.search(
                    text
                ):
                    score += 3

                if "privacy" in href.lower():
                    score += 2

                if "/legal/" in href.lower():
                    score += 1

                ranked.append(
                    (
                        score,
                        href,
                    )
                )

            ranked.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                )
            )

            candidates = []

            for score, url in ranked[:8]:
                candidates.append(
                    Candidate(
                        url,
                        "domain_link",
                    )
                )

            return candidates

        except Exception:
            return []

        finally:
            page.close()

    # -----------------------------------------------------------------------
    # Search-based discovery
    # -----------------------------------------------------------------------

    def _search_candidates(
        self,
        context,
        domain_url,
    ):
        if not self.search_enabled:
            return []

        page = context.new_page()

        try:
            hostname = _host(
                domain_url
            )

            candidates = []

            tokens = _service_tokens(
                domain_url
            )

            if tokens:
                service_name = tokens[0]
            else:
                service_name = hostname

            queries = [
                (
                    'site:'
                    + hostname
                    + ' "privacy policy" OR "privacy notice"'
                ),
                (
                    '"'
                    + service_name
                    + '" "privacy policy" OR "privacy notice"'
                ),
            ]

            for raw_query in queries:
                search_url = self._assert_safe_url(
                    "https://html.duckduckgo.com/html/?q="
                    + quote_plus(
                        raw_query
                    )
                )

                response = page.goto(
                    search_url,
                    wait_until="domcontentloaded",
                    timeout=self.timeout_ms,
                )

                if (
                    response is None
                    or response.status >= 400
                ):
                    continue

                self._wait_for_document_stability(
                    page
                )

                results = page.eval_on_selector_all(
                    "a.result__a, a[data-testid='result-title-a']",
                    """
                    els => els.map(a => ({
                        href: a.href,
                        text: (
                            a.innerText ||
                            a.textContent ||
                            ''
                        ).trim()
                    }))
                    """,
                )

                for result in results:
                    if not isinstance(
                        result,
                        dict,
                    ):
                        continue

                    href = (
                        result.get("href")
                        or ""
                    )

                    text = (
                        result.get("text")
                        or ""
                    )

                    href = _unwrap_search_url(
                        href
                    )

                    if not POLICY_LINK_RE.search(
                        text + " " + href
                    ):
                        continue

                    try:
                        href = self._assert_safe_url(
                            href
                        )

                    except ValueError:
                        continue

                    candidates.append(
                        Candidate(
                            href,
                            "web_search",
                        )
                    )

                    if len(candidates) >= 8:
                        break

                if len(candidates) >= 8:
                    break

            return candidates

        except Exception:
            return []

        finally:
            page.close()

    # -----------------------------------------------------------------------
    # Candidate deduplication
    # -----------------------------------------------------------------------

    def _dedupe(
        self,
        candidates,
    ):
        output = []
        seen = set()

        for candidate in candidates:
            key = _candidate_key(
                candidate.url
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            output.append(
                candidate
            )

        return output

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def retrieve(
        self,
        domain_url,
        privacy_policy_url=None,
        telemetry=None,
    ):
        """Retrieve, verify, extract, and prepare an applicable policy.

        `telemetry` is optional.

        When supplied, it is used only to rank policy sections for the
        bounded model context. It does not affect the canonical policy
        extraction.
        """

        domain_url = self._assert_safe_url(
            domain_url
        )

        supplied = None

        supplied_failed = (
            privacy_policy_url is not None
        )

        if privacy_policy_url:
            try:
                supplied = self._assert_safe_url(
                    privacy_policy_url
                )

            except (
                TypeError,
                ValueError,
            ):
                supplied = None

        try:
            from playwright.sync_api import (
                sync_playwright,
            )

        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. "
                "Run `pip install playwright` and "
                "`playwright install chromium`."
            ) from exc

        attempts = []

        with sync_playwright() as playwright:
            browser_type = getattr(
                playwright,
                self.browser_name,
                None,
            )

            if browser_type is None:
                raise ValueError(
                    "unsupported browser "
                    + repr(
                        self.browser_name
                    )
                    + "; use chromium, firefox, or webkit"
                )

            browser = browser_type.launch(
                headless=self.headless
            )

            context = browser.new_context(
                service_workers="block",
                java_script_enabled=True,
                locale="en-US",
            )

            context.set_default_navigation_timeout(
                self.timeout_ms
            )

            context.set_default_timeout(
                min(
                    self.timeout_ms,
                    5_000,
                )
            )

            context.route(
                "**/*",
                self._route,
            )

            try:
                # -----------------------------------------------------------
                # Try explicitly supplied policy URL first.
                # -----------------------------------------------------------

                if supplied:
                    candidate = Candidate(
                        supplied,
                        "supplied_url",
                    )

                    document, error = (
                        self._fetch_candidate(
                            context,
                            candidate,
                            domain_url,
                            telemetry,
                        )
                    )

                    if document:
                        return document

                    attempts.append({
                        "url":
                            candidate.url,
                        "method":
                            candidate.method,
                        "error":
                            error
                            or "unknown retrieval error",
                    })

                # -----------------------------------------------------------
                # Discover links from the actual site.
                # -----------------------------------------------------------

                candidates = (
                    self._discover_domain_links(
                        context,
                        domain_url,
                    )
                )

                # -----------------------------------------------------------
                # Try conventional privacy-policy paths.
                # -----------------------------------------------------------

                origin = _origin(
                    domain_url
                )

                for path in COMMON_POLICY_PATHS:
                    candidates.append(
                        Candidate(
                            urljoin(
                                origin + "/",
                                path,
                            ),
                            "common_path",
                        )
                    )

                # -----------------------------------------------------------
                # Search only for candidate discovery.
                # -----------------------------------------------------------

                candidates.extend(
                    self._search_candidates(
                        context,
                        domain_url,
                    )
                )

                candidates = self._dedupe(
                    candidates
                )

                candidates = candidates[
                    :self.max_candidates
                ]

                # -----------------------------------------------------------
                # Retrieve and verify candidates.
                # -----------------------------------------------------------

                for candidate in candidates:
                    if (
                        supplied
                        and _candidate_key(
                            candidate.url
                        )
                        == _candidate_key(
                            supplied
                        )
                    ):
                        continue

                    document, error = (
                        self._fetch_candidate(
                            context,
                            candidate,
                            domain_url,
                            telemetry,
                        )
                    )

                    if document:
                        if supplied_failed:
                            document[
                                "limitations"
                            ].append(
                                "The supplied privacy-policy URL could not "
                                "be verified. Policy discovery located the "
                                "applicable policy used for this comparison."
                            )

                        return document

                    attempts.append({
                        "url":
                            candidate.url,
                        "method":
                            candidate.method,
                        "error":
                            error
                            or "unknown retrieval error",
                    })

            finally:
                context.close()
                browser.close()

        limitation = (
            "No applicable privacy policy could be retrieved "
            "or verified for the supplied domain."
        )

        if supplied_failed:
            limitation += (
                " The supplied privacy-policy URL could not be used."
            )

        return {
            "url":
                "",
            "found":
                False,
            "applicable":
                False,
            "complete":
                False,
            "title":
                "",
            "retrieved_at":
                dt.datetime.now(
                    dt.timezone.utc
                ).isoformat().replace(
                    "+00:00",
                    "Z",
                ),
            "retrieval_method":
                "none",
            "document_hash":
                "",
            "document": {
                "full_text":
                    "",
                "original_chars":
                    0,
                "extracted_chars":
                    0,
                "coverage":
                    0,
                "extraction_mode":
                    "none",
                "sections":
                    [],
            },
            "model_context": {
                "max_policy_chars":
                    self.max_model_policy_chars,
                "max_sections":
                    self.max_model_sections,
                "selected_chars":
                    0,
                "selected_sections":
                    0,
                "sections":
                    [],
            },
            "limitations": [
                limitation,
            ],
            "attempted_candidates":
                len(attempts),
            "attempts":
                attempts,
        }