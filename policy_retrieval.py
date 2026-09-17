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
import datetime as dt
import hashlib
import ipaddress
import logging
import re
import socket
import sys
import time

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
        max_policy_chars=None,
        max_model_sections=8,
        max_candidates=14,
        search_enabled=True,
        browser_name="chromium",
        headless=True,
        allow_private_network=False,

        # Logging
        log_level="INFO",
        log_file=None,
        log_format=None,
        logger=None,

        **kwargs
    ):
        self.timeout_ms = timeout_ms
        self.max_document_chars = max_document_chars
        self.section_chunk_chars = section_chunk_chars
        self.max_sections = max_sections

        if max_policy_chars is not None:
            max_model_policy_chars = max_policy_chars

        self.max_model_policy_chars = max_model_policy_chars
        self.max_model_sections = max_model_sections
        self.max_candidates = max_candidates
        self.search_enabled = search_enabled
        self.browser_name = browser_name
        self.headless = headless
        self.allow_private_network = allow_private_network
        self._dns_cache = {}

        self.logger = self._configure_logger(
            logger=logger,
            log_level=log_level,
            log_file=log_file,
            log_format=log_format,
        )

        self.logger.info(
            "PolicyRetriever initialized "
            "browser=%s headless=%s timeout_ms=%d "
            "search_enabled=%s max_candidates=%d "
            "max_document_chars=%d max_sections=%d "
            "max_model_policy_chars=%d max_model_sections=%d",
            self.browser_name,
            self.headless,
            self.timeout_ms,
            self.search_enabled,
            self.max_candidates,
            self.max_document_chars,
            self.max_sections,
            self.max_model_policy_chars,
            self.max_model_sections,
        )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_log_level(level):
        if isinstance(level, int):
            return level

        if not isinstance(level, str):
            raise ValueError(
                "log_level must be a logging level name or integer"
            )

        normalized = level.strip().upper()

        value = getattr(logging, normalized, None)

        if not isinstance(value, int):
            raise ValueError(
                f"invalid log level {level!r}; use "
                "DEBUG, INFO, WARNING, ERROR, or CRITICAL"
            )

        return value

    def _configure_logger(
        self,
        logger=None,
        log_level="INFO",
        log_file=None,
        log_format=None,
    ):
        level = self._normalize_log_level(log_level)

        if log_format is None:
            log_format = (
                "%(asctime)s | %(levelname)-8s | "
                "%(name)s | %(message)s"
            )

        formatter = logging.Formatter(
            log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if logger is None:
            logger = logging.getLogger(
                f"{__name__}.PolicyRetriever"
            )

            # Avoid installing our handlers multiple times if multiple
            # PolicyRetriever instances are created.
            if not getattr(logger, "_policy_retriever_configured", False):
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(formatter)
                logger.addHandler(stream_handler)

                if log_file:
                    file_handler = logging.FileHandler(
                        log_file,
                        encoding="utf-8",
                    )
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)

                logger.propagate = False
                logger._policy_retriever_configured = True

        else:
            # Respect caller-provided handlers, but still allow us to set
            # the requested level.
            if log_file:
                file_handler = logging.FileHandler(
                    log_file,
                    encoding="utf-8",
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

        logger.setLevel(level)

        for handler in logger.handlers:
            handler.setLevel(level)

        return logger

    # -------------------------------------------------------------------------
    # URL / network safety
    # -------------------------------------------------------------------------

    def _assert_safe_url(self, url):
        self.logger.debug("Validating URL: %r", url)

        if not isinstance(url, str) or not url.strip():
            self.logger.debug("Rejected empty URL")
            raise ValueError("URL is empty")

        url = url.strip()
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            self.logger.debug(
                "Rejected non-HTTP(S) or non-absolute URL: %s",
                url,
            )
            raise ValueError("only absolute HTTP(S) URLs are allowed")

        if parsed.username or parsed.password:
            self.logger.warning(
                "Rejected URL containing credentials host=%s",
                parsed.hostname,
            )
            raise ValueError("URLs containing credentials are not allowed")

        try:
            port = parsed.port or (
                443 if parsed.scheme == "https" else 80
            )
        except ValueError as exc:
            self.logger.warning(
                "Rejected URL with invalid port: %s",
                url,
            )
            raise ValueError("URL contains an invalid port") from exc

        if port not in {80, 443}:
            self.logger.warning(
                "Rejected URL using disallowed port host=%s port=%s",
                parsed.hostname,
                port,
            )
            raise ValueError("only ports 80 and 443 are allowed")

        hostname = parsed.hostname.lower().rstrip(".")

        if self.allow_private_network:
            self.logger.debug(
                "Private-network checks disabled for host=%s",
                hostname,
            )
            return url

        if hostname in {"localhost", "localhost.localdomain"}:
            self.logger.warning(
                "Rejected localhost URL host=%s",
                hostname,
            )
            raise ValueError("local hosts are not allowed")

        if hostname.endswith(
            (".local", ".internal", ".localhost")
        ):
            self.logger.warning(
                "Rejected internal hostname host=%s",
                hostname,
            )
            raise ValueError(
                "local or internal hosts are not allowed"
            )

        cache_key = (hostname, port)
        addresses = self._dns_cache.get(cache_key)

        if addresses is None:
            self.logger.debug(
                "Resolving DNS hostname=%s port=%d",
                hostname,
                port,
            )

            try:
                info = socket.getaddrinfo(
                    hostname,
                    port,
                    type=socket.SOCK_STREAM,
                )
            except socket.gaierror as exc:
                self.logger.warning(
                    "DNS resolution failed hostname=%s error=%s",
                    hostname,
                    exc,
                )
                raise ValueError(
                    f"host did not resolve: {hostname}"
                ) from exc

            addresses = sorted(
                {row[4][0] for row in info}
            )

            if not addresses:
                self.logger.warning(
                    "DNS returned no addresses hostname=%s",
                    hostname,
                )
                raise ValueError(
                    f"host did not resolve: {hostname}"
                )

            self._dns_cache[cache_key] = addresses

            self.logger.debug(
                "DNS resolved hostname=%s addresses=%s",
                hostname,
                addresses,
            )

        else:
            self.logger.debug(
                "Using cached DNS hostname=%s addresses=%s",
                hostname,
                addresses,
            )

        for address in addresses:
            ip = ipaddress.ip_address(address)

            if not ip.is_global:
                self.logger.warning(
                    "Rejected hostname resolving to non-public address "
                    "hostname=%s address=%s",
                    hostname,
                    address,
                )

                raise ValueError(
                    f"host resolves to a non-public address: {hostname}"
                )

        return url

    def _route(self, route):
        request = route.request
        parsed = urlparse(request.url)

        self.logger.debug(
            "Browser request method=%s type=%s url=%s",
            request.method,
            request.resource_type,
            request.url,
        )

        if parsed.scheme not in {"http", "https"}:
            if parsed.scheme in {"data", "blob", "about"}:
                self.logger.debug(
                    "Allowing internal browser resource scheme=%s",
                    parsed.scheme,
                )
                route.continue_()
            else:
                self.logger.debug(
                    "Blocking unsupported scheme=%s url=%s",
                    parsed.scheme,
                    request.url,
                )
                route.abort()

            return

        try:
            self._assert_safe_url(request.url)
        except ValueError as exc:
            self.logger.debug(
                "Blocking unsafe browser request url=%s reason=%s",
                request.url,
                exc,
            )
            route.abort()
            return

        if request.resource_type in {
            "image",
            "media",
            "font",
        }:
            self.logger.debug(
                "Blocking unnecessary resource type=%s url=%s",
                request.resource_type,
                request.url,
            )
            route.abort()
            return

        route.continue_()

    # -------------------------------------------------------------------------
    # Policy verification
    # -------------------------------------------------------------------------

    def _looks_like_policy(self, url, title, body_text):
        title = _plain_space(title)
        body_text = _plain_space(body_text)

        if len(body_text) < 250:
            self.logger.debug(
                "Policy verification failed: body too small "
                "url=%s chars=%d",
                url,
                len(body_text),
            )
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

        result = score >= 5 and not cookie_only

        self.logger.debug(
            "Policy verification url=%s title=%r chars=%d "
            "score=%d cookie_only=%s accepted=%s",
            url,
            title[:120],
            len(body_text),
            score,
            cookie_only,
            result,
        )

        return result

    def _is_applicable(
        self,
        domain_url,
        policy_url,
        title,
        body_text,
    ):
        # we will assume the policy is applicable instead of assuming domain inference
        self.logger.debug(
            "Policy assume applicability accepted by forced-site match "
            "domain=%s policy=%s",
            domain_url,
            policy_url,
        )
        return True

    # -------------------------------------------------------------------------
    # Page stabilization
    # -------------------------------------------------------------------------

    def _wait_for_document_stability(self, page):
        self.logger.debug(
            "Waiting for document stability url=%s",
            page.url,
        )

        try:
            page.wait_for_load_state(
                "networkidle",
                timeout=min(self.timeout_ms, 4_000),
            )

            self.logger.debug(
                "networkidle reached url=%s",
                page.url,
            )

        except Exception as exc:
            self.logger.debug(
                "networkidle timeout/exception url=%s error=%s",
                page.url,
                exc,
            )

        previous_length = -1
        stable_reads = 0

        for read_number in range(16):
            try:
                current_length = page.evaluate(
                    """
                    () => {
                        if (!document.body) return 0;
                        return (
                            document.body.innerText ||
                            document.body.textContent ||
                            ''
                        ).length;
                    }
                    """
                )

            except Exception as exc:
                self.logger.debug(
                    "Body-length evaluation failed url=%s error=%s",
                    page.url,
                    exc,
                )
                current_length = 0

            self.logger.debug(
                "Document stability read=%d url=%s chars=%d",
                read_number + 1,
                page.url,
                current_length,
            )

            if current_length >= 500 and previous_length >= 0:
                difference = abs(
                    current_length - previous_length
                )

                tolerance = max(
                    25,
                    current_length // 100,
                )

                if difference <= tolerance:
                    stable_reads += 1
                else:
                    stable_reads = 0

                if stable_reads >= 3:
                    self.logger.debug(
                        "Document stable url=%s chars=%d",
                        page.url,
                        current_length,
                    )
                    return

            previous_length = current_length
            page.wait_for_timeout(350)

        self.logger.debug(
            "Document stability loop finished without explicit "
            "stability url=%s last_chars=%d",
            page.url,
            previous_length,
        )

    def _prime_document(self, page):
        self.logger.debug(
            "Priming lazy-loaded document url=%s",
            page.url,
        )

        try:
            page.evaluate(
                """
                () => {
                    window.scrollTo(
                        0,
                        document.body
                            ? document.body.scrollHeight
                            : 0
                    );
                }
                """
            )

            page.wait_for_timeout(250)

            page.evaluate(
                "() => window.scrollTo(0, 0)"
            )

            page.wait_for_timeout(150)

        except Exception as exc:
            self.logger.debug(
                "Document priming failed url=%s error=%s",
                page.url,
                exc,
            )

    # -------------------------------------------------------------------------
    # DOM extraction
    # -------------------------------------------------------------------------

    def _extract_from_frame(self, frame):
        self.logger.debug(
            "Extracting policy content from frame url=%s",
            frame.url,
        )

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

                const policyHint =
                    /privacy|legal|policy|data-protection|data_privacy/i;

                const policyText =
                    /personal (data|information)|information we collect|data we collect|privacy rights|data protection|how we (use|share|collect|process)/i;

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

                const candidates = [
                    ...document.querySelectorAll(selector)
                ];

                let source = null;
                let bestScore = -1;

                for (const candidate of candidates) {
                    const text = normalize(
                        candidate.textContent ||
                        candidate.innerText ||
                        ''
                    );

                    if (text.length < 250) continue;

                    const hint = [
                        candidate.id || '',
                        candidate.className || '',
                        candidate.getAttribute('role') || '',
                        candidate.tagName || ''
                    ].join(' ');

                    let score = Math.min(
                        text.length,
                        60000
                    );

                    if (policyHint.test(hint)) {
                        score += 50000;
                    }

                    if (
                        policyText.test(
                            text.slice(0, 20000)
                        )
                    ) {
                        score += 30000;
                    }

                    if (
                        /^(MAIN|ARTICLE)$/.test(
                            candidate.tagName
                        )
                    ) {
                        score += 5000;
                    }

                    if (
                        candidate.getAttribute('role') ===
                        'main'
                    ) {
                        score += 5000;
                    }

                    if (score > bestScore) {
                        bestScore = score;
                        source = candidate;
                    }
                }

                if (!source) {
                    source = document.body;
                }

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
                ].join(',')).forEach(
                    node => node.remove()
                );

                const fullText = normalize(
                    root.textContent ||
                    root.innerText ||
                    ''
                );

                const blockSelector =
                    'h1,h2,h3,h4,h5,h6,p,li,dt,dd,blockquote,pre';

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
                        node.textContent ||
                        node.innerText ||
                        ''
                    );

                    if (!text) continue;

                    if (
                        /^H[1-6]$/.test(
                            node.tagName
                        )
                    ) {
                        heading = text.slice(0, 500);
                        continue;
                    }

                    if (
                        node.querySelector(
                            blockSelector
                        )
                    ) {
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

                const semanticLength =
                    items.reduce(
                        (total, item) =>
                            total + item.text.length,
                        0
                    );

                if (
                    !items.length ||
                    semanticLength <
                        Math.min(
                            fullText.length * 0.55,
                            5000
                        )
                ) {
                    return {
                        title:
                            document.title || '',
                        fullText,
                        items: [{
                            heading:
                                normalize(
                                    document.title
                                ) ||
                                'Privacy Policy',
                            text: fullText
                        }],
                        semanticLength:
                            fullText.length,
                        extractionMode:
                            'full_text_fallback'
                    };
                }

                return {
                    title:
                        document.title || '',
                    fullText,
                    items,
                    semanticLength,
                    extractionMode:
                        'semantic'
                };
            }
            """
        )

    def _extract_raw_policy(self, page):
        self.logger.info(
            "Extracting policy DOM url=%s frames=%d",
            page.url,
            len(page.frames),
        )

        extractions = []

        for frame_number, frame in enumerate(
            page.frames,
            1,
        ):
            self.logger.debug(
                "Inspecting frame %d/%d url=%s",
                frame_number,
                len(page.frames),
                frame.url,
            )

            try:
                if (
                    frame.url
                    and frame.url != "about:blank"
                ):
                    parsed = urlparse(frame.url)

                    if parsed.scheme in {
                        "http",
                        "https",
                    }:
                        self._assert_safe_url(
                            frame.url
                        )

                extracted = self._extract_from_frame(
                    frame
                )

            except Exception as exc:
                self.logger.debug(
                    "Frame extraction failed "
                    "frame_url=%s error=%s",
                    frame.url,
                    exc,
                    exc_info=True,
                )
                continue

            if not extracted:
                self.logger.debug(
                    "Frame returned no extraction "
                    "frame_url=%s",
                    frame.url,
                )
                continue

            text = (
                extracted.get("fullText")
                or ""
            )

            if len(text) < 250:
                self.logger.debug(
                    "Ignoring short extraction "
                    "frame_url=%s chars=%d",
                    frame.url,
                    len(text),
                )
                continue

            score = min(
                len(text),
                100_000,
            )

            title = (
                extracted.get("title")
                or ""
            )

            if POLICY_LINK_RE.search(title):
                score += 40_000

            if POLICY_TEXT_RE.search(
                text[:20_000]
            ):
                score += 40_000

            if frame == page.main_frame:
                score += 5_000

            extracted["frame_url"] = frame.url
            extracted["selection_score"] = score

            extractions.append(extracted)

            self.logger.debug(
                "Frame extraction candidate "
                "frame_url=%s chars=%d mode=%s "
                "items=%d score=%d",
                frame.url,
                len(text),
                extracted.get(
                    "extractionMode"
                ),
                len(
                    extracted.get("items")
                    or []
                ),
                score,
            )

        if not extractions:
            self.logger.warning(
                "No viable policy extraction found url=%s",
                page.url,
            )

            return {
                "title": "",
                "fullText": "",
                "items": [],
                "semanticLength": 0,
                "extractionMode": "empty",
                "frame_url": "",
            }

        extractions.sort(
            key=lambda item:
                item["selection_score"],
            reverse=True,
        )

        selected = extractions[0]

        self.logger.info(
            "Selected extraction frame=%s "
            "mode=%s chars=%d items=%d score=%d",
            selected.get("frame_url"),
            selected.get("extractionMode"),
            len(
                selected.get("fullText")
                or ""
            ),
            len(
                selected.get("items")
                or []
            ),
            selected.get(
                "selection_score",
                0,
            ),
        )

        return selected

    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------

    def _split_text(self, text):
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

                question = text.rfind(
                    "? ",
                    start,
                    end,
                )

                exclamation = text.rfind(
                    "! ",
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
                    question,
                    exclamation,
                    semicolon,
                )

                minimum_boundary = (
                    start
                    + self.section_chunk_chars
                    // 2
                )

                if boundary > minimum_boundary:
                    end = boundary + 1

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end <= start:
                break

            start = end

        self.logger.debug(
            "Split text chars=%d chunks=%d "
            "chunk_target=%d",
            len(text),
            len(chunks),
            self.section_chunk_chars,
        )

        return chunks

    def _group_items(self, items):
        groups = []
        current_heading = None
        current_parts = []

        def flush():
            nonlocal current_heading
            nonlocal current_parts

            if not current_parts:
                return

            text = _plain_space(
                " ".join(current_parts)
            )

            if text:
                groups.append({
                    "heading":
                        current_heading
                        or "Privacy Policy",
                    "text": text,
                })

            current_parts = []

        for item in items:
            if not isinstance(item, dict):
                continue

            heading = _plain_space(
                item.get("heading")
                or "Privacy Policy"
            )[:500]

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

            current_parts.append(text)

        flush()

        self.logger.debug(
            "Grouped semantic items items=%d groups=%d",
            len(items),
            len(groups),
        )

        return groups

    def _build_sections(self, groups):
        sections = []
        section_limit_hit = False

        for group in groups:
            chunks = self._split_text(
                group["text"]
            )

            total_parts = len(chunks)

            for part_number, chunk in enumerate(
                chunks,
                1,
            ):
                if len(sections) >= self.max_sections:
                    section_limit_hit = True

                    self.logger.warning(
                        "Canonical section limit hit "
                        "limit=%d",
                        self.max_sections,
                    )

                    break

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

            if section_limit_hit:
                break

        self.logger.debug(
            "Built canonical sections groups=%d "
            "sections=%d limit_hit=%s",
            len(groups),
            len(sections),
            section_limit_hit,
        )

        return sections, section_limit_hit

    def _limit_document_sections(
        self,
        sections,
    ):
        if not self.max_document_chars:
            return sections, False

        output = []
        used_chars = 0
        truncated = False

        for section in sections:
            remaining = (
                self.max_document_chars
                - used_chars
            )

            if remaining <= 0:
                truncated = True
                break

            text = section["text"]

            if len(text) <= remaining:
                output.append(
                    section.copy()
                )

                used_chars += len(text)
                continue

            if remaining >= 500:
                shortened = (
                    section.copy()
                )

                shortened["text"] = (
                    self._cut_text(
                        text,
                        remaining,
                    )
                )

                shortened[
                    "document_truncated"
                ] = True

                output.append(shortened)

                used_chars += len(
                    shortened["text"]
                )

            truncated = True
            break

        if truncated:
            self.logger.warning(
                "Canonical document truncated "
                "limit=%d used_chars=%d "
                "input_sections=%d output_sections=%d",
                self.max_document_chars,
                used_chars,
                len(sections),
                len(output),
            )

        return output, truncated

    def _cut_text(self, text, limit):
        if len(text) <= limit:
            return text

        minimum_boundary = max(
            0,
            limit // 2,
        )

        boundaries = (
            text.rfind(
                ". ",
                minimum_boundary,
                limit,
            ),
            text.rfind(
                "? ",
                minimum_boundary,
                limit,
            ),
            text.rfind(
                "! ",
                minimum_boundary,
                limit,
            ),
            text.rfind(
                "; ",
                minimum_boundary,
                limit,
            ),
        )

        boundary = max(boundaries)

        if boundary > 0:
            return text[
                :boundary + 1
            ].strip()

        return text[:limit].strip()

    def _extract_sections(self, page):
        started = time.perf_counter()

        self.logger.info(
            "Building canonical policy document url=%s",
            page.url,
        )

        extracted = self._extract_raw_policy(
            page
        )

        raw_full_text = (
            extracted.get("fullText")
            or ""
        )

        original_chars = len(
            raw_full_text
        )

        items = (
            extracted.get("items")
            or []
        )

        if not raw_full_text or not items:
            self.logger.warning(
                "Canonical policy extraction empty "
                "url=%s original_chars=%d items=%d",
                page.url,
                original_chars,
                len(items),
            )

            return {
                "full_text": "",
                "sections": [],
                "original_chars":
                    original_chars,
                "extracted_chars": 0,
                "uncapped_extracted_chars": 0,
                "coverage": 0,
                "complete": False,
                "document_truncated": False,
                "section_limit_hit": False,
                "extraction_mode":
                    extracted.get(
                        "extractionMode"
                    )
                    or "empty",
                "frame_url":
                    extracted.get(
                        "frame_url"
                    )
                    or "",
            }

        groups = self._group_items(
            items
        )

        (
            uncapped_sections,
            section_limit_hit,
        ) = self._build_sections(
            groups
        )

        uncapped_extracted_chars = sum(
            len(section["text"])
            for section
            in uncapped_sections
        )

        (
            sections,
            document_truncated,
        ) = self._limit_document_sections(
            uncapped_sections
        )

        for index, section in enumerate(
            sections,
            1,
        ):
            section[
                "section_id"
            ] = f"policy-{index:04d}"

        extracted_chars = sum(
            len(section["text"])
            for section
            in sections
        )

        if (
            extracted.get(
                "extractionMode"
            )
            == "full_text_fallback"
        ):
            coverage = (
                1.0
                if original_chars
                else 0
            )

        elif original_chars:
            coverage = min(
                uncapped_extracted_chars
                / original_chars,
                1.0,
            )

        else:
            coverage = 0

        complete = bool(
            sections
            and not document_truncated
            and not section_limit_hit
            and coverage >= 0.60
        )

        canonical_full_text = (
            "\n\n".join(
                section["text"]
                for section
                in sections
            )
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        self.logger.info(
            "Canonical extraction complete "
            "mode=%s original_chars=%d "
            "extracted_chars=%d "
            "uncapped_chars=%d sections=%d "
            "coverage=%.4f complete=%s "
            "document_truncated=%s "
            "section_limit_hit=%s "
            "duration=%.3fs",
            extracted.get(
                "extractionMode"
            )
            or "unknown",
            original_chars,
            extracted_chars,
            uncapped_extracted_chars,
            len(sections),
            coverage,
            complete,
            document_truncated,
            section_limit_hit,
            elapsed,
        )

        return {
            "full_text":
                canonical_full_text,
            "sections":
                sections,
            "original_chars":
                original_chars,
            "extracted_chars":
                extracted_chars,
            "uncapped_extracted_chars":
                uncapped_extracted_chars,
            "coverage":
                round(coverage, 4),
            "complete":
                complete,
            "document_truncated":
                document_truncated,
            "section_limit_hit":
                section_limit_hit,
            "extraction_mode":
                extracted.get(
                    "extractionMode"
                )
                or "unknown",
            "frame_url":
                extracted.get(
                    "frame_url"
                )
                or "",
        }

    # -------------------------------------------------------------------------
    # Telemetry / section selection
    # -------------------------------------------------------------------------

    def _detect_categories(self, telemetry):
        categories = {
            "collection",
            "sharing",
        }

        if not telemetry:
            categories.update({
                "tracking",
                "device",
            })

            result = sorted(categories)

            self.logger.debug(
                "No telemetry supplied; using default categories=%s",
                result,
            )

            return result

        terms = " ".join(
            sorted(
                _active_telemetry_terms(
                    telemetry
                )
            )
        )

        if any(
            term in terms
            for term in (
                "cookie",
                "localstorage",
                "local storage",
                "sessionstorage",
                "session storage",
                "indexeddb",
                "cache storage",
                "storage",
            )
        ):
            categories.add("cookies")

        if any(
            term in terms
            for term in (
                "tracker",
                "thirdparty",
                "third party",
                "analytics",
                "pixel",
                "beacon",
            )
        ):
            categories.add("tracking")

        if any(
            term in terms
            for term in (
                "canvas",
                "webgl",
                "audio",
                "fingerprint",
                "useragent",
                "user agent",
            )
        ):
            categories.add(
                "fingerprinting"
            )

        if any(
            term in terms
            for term in (
                "request",
                "host",
                "network",
                "ip",
                "third party host",
            )
        ):
            categories.add("network")

        if any(
            term in terms
            for term in (
                "permission",
                "camera",
                "microphone",
                "sensor",
                "notification",
            )
        ):
            categories.add(
                "permissions"
            )

        if any(
            term in terms
            for term in (
                "device",
                "screen",
                "browser",
                "useragent",
                "user agent",
                "webgl",
            )
        ):
            categories.add("device")

        if any(
            term in terms
            for term in (
                "location",
                "geolocation",
            )
        ):
            categories.add("location")

        result = sorted(categories)

        self.logger.debug(
            "Detected telemetry categories=%s active_term_chars=%d",
            result,
            len(terms),
        )

        return result

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

        score = 0

        if position == 0:
            score += 6

        elif position < 3:
            score += 2

        for category in categories:
            for term in CATEGORY_TERMS.get(
                category,
                (),
            ):
                if term in heading:
                    score += 8

                if term in text:
                    score += 3

        if re.search(
            r"\bdefinitions?\b|\bscope\b|\babout this policy\b",
            heading,
        ):
            score += 4

        if "collect" in text:
            score += 1

        if "share" in text:
            score += 1

        if (
            "third party" in text
            or "third-party" in text
        ):
            score += 1

        if "cookie" in text:
            score += 1

        if "device" in text:
            score += 1

        return score

    def _select_model_sections(
        self,
        sections,
        telemetry=None,
    ):
        if not sections:
            self.logger.warning(
                "No canonical sections available for model selection"
            )
            return []

        categories = (
            self._detect_categories(
                telemetry
            )
        )

        self.logger.info(
            "Selecting Qwen policy context "
            "canonical_sections=%d "
            "categories=%s "
            "char_budget=%d "
            "section_budget=%d",
            len(sections),
            categories,
            self.max_model_policy_chars,
            self.max_model_sections,
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
                "score": score,
                "position": position,
                "section": section,
            })

            self.logger.debug(
                "Policy section score "
                "id=%s position=%d score=%d "
                "heading=%r chars=%d",
                section.get("section_id"),
                position,
                score,
                (
                    section.get("heading")
                    or ""
                )[:120],
                len(
                    section.get("text")
                    or ""
                ),
            )

        scored.sort(
            key=lambda item: (
                -item["score"],
                item["position"],
            )
        )

        selected = []
        selected_ids = set()
        used_chars = 0

        def add_section(section):
            nonlocal used_chars

            section_id = section[
                "section_id"
            ]

            if section_id in selected_ids:
                return False

            if (
                len(selected)
                >= self.max_model_sections
            ):
                self.logger.debug(
                    "Skipping model section %s: "
                    "section budget reached",
                    section_id,
                )
                return False

            text_length = len(
                section["text"]
            )

            if (
                used_chars
                + text_length
                > self.max_model_policy_chars
            ):
                self.logger.debug(
                    "Skipping model section %s: "
                    "char budget exceeded "
                    "used=%d section=%d limit=%d",
                    section_id,
                    used_chars,
                    text_length,
                    self.max_model_policy_chars,
                )
                return False

            selected.append(
                section.copy()
            )

            selected_ids.add(
                section_id
            )

            used_chars += text_length

            self.logger.debug(
                "Selected model section "
                "id=%s chars=%d total_chars=%d",
                section_id,
                text_length,
                used_chars,
            )

            return True

        add_section(sections[0])

        for item in scored:
            add_section(
                item["section"]
            )

            if (
                len(selected)
                >= self.max_model_sections
            ):
                break

        if (
            len(selected) == 1
            and len(sections) > 1
        ):
            only = selected[0]

            if (
                len(only["text"])
                > self.max_model_policy_chars
                // 2
            ):
                self.logger.debug(
                    "Opening section consumed >50%% "
                    "of context budget; retrying "
                    "selection without forced opener"
                )

                selected = []
                selected_ids = set()
                used_chars = 0

                for item in scored:
                    add_section(
                        item["section"]
                    )

                    if (
                        len(selected)
                        >= self.max_model_sections
                    ):
                        break

        if not selected:
            self.logger.warning(
                "Scored selection produced no sections; "
                "attempting sequential fallback"
            )

            for section in sections:
                if add_section(section):
                    break

        position_map = {
            section["section_id"]:
                position
            for position, section
            in enumerate(sections)
        }

        selected.sort(
            key=lambda section:
                position_map.get(
                    section["section_id"],
                    999999,
                )
        )

        self.logger.info(
            "Qwen policy context selected "
            "sections=%d chars=%d ids=%s",
            len(selected),
            used_chars,
            [
                section["section_id"]
                for section
                in selected
            ],
        )

        return selected

    def validate_policy_section_ids(
        self,
        findings,
        document,
    ):
        valid_ids = {
            section["section_id"]
            for section
            in document.get(
                "sections",
                [],
            )
            if section.get(
                "section_id"
            )
        }

        errors = []

        for finding_index, finding in enumerate(
            findings
        ):
            for section_id in (
                finding.get(
                    "policy_section_ids"
                )
                or []
            ):
                if section_id not in valid_ids:
                    errors.append({
                        "finding_index":
                            finding_index,
                        "section_id":
                            section_id,
                        "error":
                            "unknown policy section ID",
                    })

        if errors:
            self.logger.warning(
                "Policy section ID validation failed "
                "errors=%d valid_ids=%d",
                len(errors),
                len(valid_ids),
            )

            for error in errors:
                self.logger.debug(
                    "Invalid policy citation "
                    "finding_index=%d section_id=%s",
                    error["finding_index"],
                    error["section_id"],
                )

        else:
            self.logger.info(
                "Policy section ID validation passed "
                "findings=%d valid_ids=%d",
                len(findings),
                len(valid_ids),
            )

        return {
            "valid": not errors,
            "errors": errors,
        }

    # -------------------------------------------------------------------------
    # Candidate retrieval
    # -------------------------------------------------------------------------

    def _fetch_candidate(
        self,
        context,
        candidate,
        domain_url,
        telemetry=None,
    ):
        started = time.perf_counter()

        self.logger.info(
            "Trying policy candidate "
            "method=%s url=%s",
            candidate.get("method"),
            candidate.get("url"),
        )

        page = context.new_page()

        try:
            self._assert_safe_url(
                candidate["url"]
            )

            self.logger.debug(
                "Navigating candidate url=%s",
                candidate["url"],
            )

            response = page.goto(
                candidate["url"],
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )

            if response is None:
                self.logger.warning(
                    "Candidate failed: navigation "
                    "returned no response url=%s",
                    candidate["url"],
                )

                return (
                    None,
                    "navigation returned no response",
                )

            self.logger.info(
                "Candidate HTTP response "
                "status=%d requested=%s final=%s",
                response.status,
                candidate["url"],
                page.url,
            )

            if response.status >= 400:
                return (
                    None,
                    f"HTTP {response.status}",
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

            self.logger.debug(
                "Candidate response metadata "
                "final_url=%s content_type=%s",
                final_url,
                content_type,
            )

            if "application/pdf" in content_type:
                self.logger.info(
                    "Candidate rejected because it is a PDF "
                    "url=%s",
                    final_url,
                )

                return (
                    None,
                    "PDF policy extraction is not "
                    "supported by this retriever",
                )

            self._wait_for_document_stability(
                page
            )

            self._prime_document(page)

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

            except Exception as exc:
                self.logger.debug(
                    "Could not read body text "
                    "url=%s error=%s",
                    final_url,
                    exc,
                )

                body_text = ""

            self.logger.info(
                "Candidate loaded "
                "url=%s title=%r body_chars=%d",
                final_url,
                title[:160],
                len(body_text),
            )

            if not self._looks_like_policy(
                final_url,
                title,
                body_text,
            ):
                self.logger.info(
                    "Candidate rejected: page does not "
                    "verify as privacy policy url=%s",
                    final_url,
                )

                return (
                    None,
                    "page did not verify as a "
                    "privacy policy",
                )

            self.logger.info(
                "Candidate verified as privacy-policy-like "
                "url=%s",
                final_url,
            )

            if not self._is_applicable(
                domain_url,
                final_url,
                title,
                body_text,
            ):
                return (
                    None,
                    "policy did not verify as "
                    "applicable to the supplied domain",
                )

            self.logger.info(
                "Candidate applicability verified "
                "domain=%s policy=%s",
                domain_url,
                final_url,
            )

            extracted = (
                self._extract_sections(
                    page
                )
            )

            canonical_sections = (
                extracted["sections"]
            )

            if not canonical_sections:
                self.logger.warning(
                    "Candidate verified but policy text "
                    "could not be extracted url=%s",
                    final_url,
                )

                return (
                    None,
                    "policy text could not be extracted",
                )

            model_sections = (
                self._select_model_sections(
                    canonical_sections,
                    telemetry,
                )
            )

            if not model_sections:
                self.logger.warning(
                    "Canonical extraction succeeded but "
                    "model context selection failed url=%s",
                    final_url,
                )

                return (
                    None,
                    "policy was extracted but no model "
                    "context could be selected",
                )

            limitations = []

            if extracted[
                "document_truncated"
            ]:
                limitations.append(
                    "The policy exceeded the host-side "
                    "canonical document limit."
                )

            if extracted[
                "section_limit_hit"
            ]:
                limitations.append(
                    "The policy exceeded the maximum "
                    "canonical section count."
                )

            if (
                extracted["coverage"]
                < 0.60
            ):
                limitations.append(
                    "The extracted structured text covered "
                    "less than 60% of the selected policy "
                    "container."
                )

            if (
                len(model_sections)
                < len(canonical_sections)
            ):
                limitations.append(
                    "Only the policy sections most relevant "
                    "to the observed visit were supplied to "
                    "the model because of the Qwen context "
                    "budget."
                )

            model_chars = sum(
                len(section["text"])
                for section
                in model_sections
            )

            public_sections = []

            for section in model_sections:
                heading = section["heading"]

                if (
                    section.get(
                        "parts",
                        1,
                    )
                    > 1
                ):
                    heading = (
                        f'{heading} '
                        f'(Part {section["part"]}/'
                        f'{section["parts"]})'
                    )

                public_sections.append({
                    "heading": heading,
                    "text": section["text"],
                })

            elapsed = (
                time.perf_counter()
                - started
            )

            self.logger.info(
                "Policy candidate accepted "
                "url=%s method=%s "
                "canonical_sections=%d "
                "canonical_chars=%d "
                "model_sections=%d "
                "model_chars=%d "
                "coverage=%.4f "
                "complete=%s "
                "limitations=%d "
                "duration=%.3fs",
                final_url,
                candidate["method"],
                len(canonical_sections),
                extracted[
                    "extracted_chars"
                ],
                len(model_sections),
                model_chars,
                extracted["coverage"],
                extracted["complete"],
                len(limitations),
                elapsed,
            )

            return {
                "url": final_url,
                "found": True,
                "applicable": True,
                "complete":
                    extracted["complete"],
                "title": title,
                "retrieved_at":
                    dt.datetime.now(
                        dt.timezone.utc
                    )
                    .isoformat()
                    .replace(
                        "+00:00",
                        "Z",
                    ),
                "retrieval_method":
                    candidate["method"],

                "sections":
                    public_sections,

                "document_hash":
                    _document_hash(
                        extracted[
                            "full_text"
                        ]
                    ),

                "document": {
                    "full_text":
                        extracted[
                            "full_text"
                        ],
                    "original_chars":
                        extracted[
                            "original_chars"
                        ],
                    "extracted_chars":
                        extracted[
                            "extracted_chars"
                        ],
                    "uncapped_extracted_chars":
                        extracted[
                            "uncapped_extracted_chars"
                        ],
                    "coverage":
                        extracted[
                            "coverage"
                        ],
                    "extraction_mode":
                        extracted[
                            "extraction_mode"
                        ],
                    "frame_url":
                        extracted[
                            "frame_url"
                        ],
                    "sections":
                        canonical_sections,
                },

                "model_context": {
                    "max_policy_chars":
                        self.max_model_policy_chars,
                    "max_sections":
                        self.max_model_sections,
                    "selected_chars":
                        model_chars,
                    "selected_sections":
                        len(
                            model_sections
                        ),
                    "categories":
                        self._detect_categories(
                            telemetry
                        ),
                    "sections":
                        model_sections,
                },

                "limitations":
                    limitations,

            }, None

        except Exception as exc:
            elapsed = (
                time.perf_counter()
                - started
            )

            self.logger.warning(
                "Candidate failed "
                "method=%s url=%s "
                "exception=%s error=%s "
                "duration=%.3fs",
                candidate.get("method"),
                candidate.get("url"),
                exc.__class__.__name__,
                exc,
                elapsed,
            )

            self.logger.debug(
                "Candidate exception traceback",
                exc_info=True,
            )

            return (
                None,
                f"{exc.__class__.__name__}: {exc}",
            )

        finally:
            try:
                page.close()
            except Exception as exc:
                self.logger.debug(
                    "Page close failed error=%s",
                    exc,
                )

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    def _discover_domain_links(
        self,
        context,
        domain_url,
    ):
        page = context.new_page()

        self.logger.info(
            "Discovering privacy links from domain homepage "
            "domain=%s",
            domain_url,
        )

        try:
            target = self._assert_safe_url(
                _origin(domain_url) + "/"
            )

            response = page.goto(
                target,
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )

            if response is None:
                self.logger.info(
                    "Domain discovery returned no response "
                    "url=%s",
                    target,
                )
                return []

            if response.status >= 400:
                self.logger.info(
                    "Domain discovery HTTP error "
                    "url=%s status=%d",
                    target,
                    response.status,
                )
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

            self.logger.debug(
                "Domain homepage links found total=%d",
                len(links),
            )

            ranked = []

            for link in links:
                if not isinstance(
                    link,
                    dict,
                ):
                    continue

                href = link.get("href")
                text = (
                    link.get("text")
                    or ""
                )

                if (
                    not href
                    or not POLICY_LINK_RE.search(
                        f"{text} {href}"
                    )
                ):
                    continue

                try:
                    href = (
                        self._assert_safe_url(
                            href
                        )
                    )
                except ValueError as exc:
                    self.logger.debug(
                        "Ignoring unsafe discovered link "
                        "url=%s error=%s",
                        href,
                        exc,
                    )
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
                    (score, href)
                )

                self.logger.debug(
                    "Domain policy-link candidate "
                    "score=%d text=%r url=%s",
                    score,
                    text[:100],
                    href,
                )

            ranked.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                )
            )

            output = [
                {
                    "url": url,
                    "method":
                        "domain_link",
                }
                for _, url
                in ranked[:8]
            ]

            self.logger.info(
                "Domain link discovery complete "
                "candidates=%d",
                len(output),
            )

            return output

        except Exception as exc:
            self.logger.warning(
                "Domain policy-link discovery failed "
                "domain=%s error=%s",
                domain_url,
                exc,
            )

            self.logger.debug(
                "Domain discovery exception traceback",
                exc_info=True,
            )

            return []

        finally:
            try:
                page.close()
            except Exception:
                pass

    def _search_candidates(
        self,
        context,
        domain_url,
    ):
        if not self.search_enabled:
            self.logger.info(
                "Web-search policy discovery disabled"
            )
            return []

        page = context.new_page()

        self.logger.info(
            "Starting web-search policy discovery "
            "domain=%s",
            domain_url,
        )

        try:
            hostname = _host(
                domain_url
            )

            tokens = _service_tokens(
                domain_url
            )

            service_name = (
                tokens[0]
                if tokens
                else hostname
            )

            candidates = []

            queries = (
                f'site:{hostname} '
                f'"privacy policy" OR '
                f'"privacy notice"',

                f'"{service_name}" '
                f'"privacy policy" OR '
                f'"privacy notice"',
            )

            for query_number, raw_query in enumerate(
                queries,
                1,
            ):
                self.logger.info(
                    "Running policy search "
                    "query=%d/%d value=%r",
                    query_number,
                    len(queries),
                    raw_query,
                )

                search_url = (
                    self._assert_safe_url(
                        "https://html.duckduckgo.com/html/?q="
                        + quote_plus(
                            raw_query
                        )
                    )
                )

                response = page.goto(
                    search_url,
                    wait_until="domcontentloaded",
                    timeout=self.timeout_ms,
                )

                if response is None:
                    self.logger.info(
                        "Search returned no response"
                    )
                    continue

                if response.status >= 400:
                    self.logger.info(
                        "Search HTTP failure status=%d",
                        response.status,
                    )
                    continue

                results = (
                    page.eval_on_selector_all(
                        "a.result__a, "
                        "a[data-testid='result-title-a']",
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
                )

                self.logger.debug(
                    "Search results parsed query=%d results=%d",
                    query_number,
                    len(results),
                )

                for result in results:
                    if not isinstance(
                        result,
                        dict,
                    ):
                        continue

                    href = _unwrap_search_url(
                        result.get(
                            "href"
                        )
                        or ""
                    )

                    text = (
                        result.get(
                            "text"
                        )
                        or ""
                    )

                    if not POLICY_LINK_RE.search(
                        f"{text} {href}"
                    ):
                        continue

                    try:
                        href = (
                            self._assert_safe_url(
                                href
                            )
                        )

                    except ValueError as exc:
                        self.logger.debug(
                            "Ignoring unsafe search result "
                            "url=%s error=%s",
                            href,
                            exc,
                        )
                        continue

                    candidates.append({
                        "url": href,
                        "method":
                            "web_search",
                    })

                    self.logger.debug(
                        "Search candidate accepted "
                        "text=%r url=%s",
                        text[:120],
                        href,
                    )

                    if len(candidates) >= 8:
                        break

                if len(candidates) >= 8:
                    break

            self.logger.info(
                "Web-search discovery complete "
                "candidates=%d",
                len(candidates),
            )

            return candidates

        except Exception as exc:
            self.logger.warning(
                "Web-search policy discovery failed "
                "domain=%s error=%s",
                domain_url,
                exc,
            )

            self.logger.debug(
                "Search discovery exception traceback",
                exc_info=True,
            )

            return []

        finally:
            try:
                page.close()
            except Exception:
                pass

    def _dedupe(self, candidates):
        output = []
        seen = set()

        duplicates = 0

        for candidate in candidates:
            key = _candidate_key(
                candidate["url"]
            )

            if key in seen:
                duplicates += 1

                self.logger.debug(
                    "Removing duplicate candidate "
                    "url=%s",
                    candidate["url"],
                )

                continue

            seen.add(key)
            output.append(candidate)

        self.logger.info(
            "Candidate deduplication "
            "input=%d output=%d duplicates=%d",
            len(candidates),
            len(output),
            duplicates,
        )

        return output

    # -------------------------------------------------------------------------
    # Main retrieval
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        domain_url,
        privacy_policy_url=None,
        telemetry=None,
    ):
        """Retrieve, verify, extract, and prepare the applicable policy.

        The top-level `sections` field remains model-ready for compatibility.
        The complete canonical extraction is stored under `document`.
        """

        retrieval_started = time.perf_counter()

        self.logger.info(
            "============================================================"
        )

        self.logger.info(
            "Starting privacy-policy retrieval "
            "domain=%s supplied_policy=%s telemetry=%s",
            domain_url,
            privacy_policy_url or "<none>",
            "yes" if telemetry else "no",
        )

        domain_url = self._assert_safe_url(
            domain_url
        )

        self.logger.info(
            "Domain URL validated domain=%s host=%s",
            domain_url,
            _host(domain_url),
        )

        supplied = None

        supplied_failed = (
            privacy_policy_url
            not in {None, ""}
        )

        if privacy_policy_url:
            try:
                supplied = (
                    self._assert_safe_url(
                        privacy_policy_url
                    )
                )

                self.logger.info(
                    "Supplied policy URL validated url=%s",
                    supplied,
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                self.logger.warning(
                    "Supplied policy URL rejected "
                    "url=%r error=%s; discovery will continue",
                    privacy_policy_url,
                    exc,
                )

                supplied = None

        categories = (
            self._detect_categories(
                telemetry
            )
        )

        self.logger.info(
            "Telemetry policy categories=%s",
            categories,
        )

        try:
            from playwright.sync_api import (
                sync_playwright,
            )

        except ImportError as exc:
            self.logger.error(
                "Playwright import failed",
                exc_info=True,
            )

            raise RuntimeError(
                "Playwright is not installed. "
                "Run `pip install playwright` and "
                "`playwright install chromium`."
            ) from exc

        attempts = []

        self.logger.info(
            "Starting Playwright browser=%s headless=%s",
            self.browser_name,
            self.headless,
        )

        with sync_playwright() as playwright:
            browser_type = getattr(
                playwright,
                self.browser_name,
                None,
            )

            if browser_type is None:
                self.logger.error(
                    "Unsupported Playwright browser=%s",
                    self.browser_name,
                )

                raise ValueError(
                    f"unsupported browser "
                    f"{self.browser_name!r}; "
                    "use chromium, firefox, "
                    "or webkit"
                )

            browser = browser_type.launch(
                headless=self.headless
            )

            self.logger.info(
                "Playwright browser launched"
            )

            context = browser.new_context(
                service_workers="block",
                java_script_enabled=True,
                locale="en-US",
            )

            self.logger.debug(
                "Browser context created "
                "service_workers=block "
                "javascript=true locale=en-US"
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
                # -------------------------------------------------------------
                # Supplied policy
                # -------------------------------------------------------------

                if supplied:
                    self.logger.info(
                        "Trying supplied privacy-policy URL first"
                    )

                    candidate = {
                        "url": supplied,
                        "method":
                            "supplied_url",
                    }

                    (
                        document,
                        error,
                    ) = self._fetch_candidate(
                        context,
                        candidate,
                        domain_url,
                        telemetry,
                    )

                    if document:
                        elapsed = (
                            time.perf_counter()
                            - retrieval_started
                        )

                        self.logger.info(
                            "Retrieval succeeded using supplied URL "
                            "url=%s duration=%.3fs",
                            document["url"],
                            elapsed,
                        )

                        self.logger.info(
                            "============================================================"
                        )

                        return document

                    self.logger.info(
                        "Supplied policy candidate failed "
                        "url=%s error=%s",
                        candidate["url"],
                        error,
                    )

                    attempts.append({
                        "url":
                            candidate["url"],
                        "method":
                            candidate[
                                "method"
                            ],
                        "error":
                            error
                            or "unknown retrieval error",
                    })

                # -------------------------------------------------------------
                # Homepage links
                # -------------------------------------------------------------

                candidates = (
                    self._discover_domain_links(
                        context,
                        domain_url,
                    )
                )

                self.logger.info(
                    "Homepage discovery produced "
                    "%d candidates",
                    len(candidates),
                )

                # -------------------------------------------------------------
                # Common paths
                # -------------------------------------------------------------

                origin = _origin(
                    domain_url
                )

                common_candidates = []

                for path in COMMON_POLICY_PATHS:
                    candidate = {
                        "url":
                            urljoin(
                                origin + "/",
                                path,
                            ),
                        "method":
                            "common_path",
                    }

                    common_candidates.append(
                        candidate
                    )

                candidates.extend(
                    common_candidates
                )

                self.logger.info(
                    "Added %d common-path candidates",
                    len(common_candidates),
                )

                # -------------------------------------------------------------
                # Search
                # -------------------------------------------------------------

                search_candidates = (
                    self._search_candidates(
                        context,
                        domain_url,
                    )
                )

                candidates.extend(
                    search_candidates
                )

                self.logger.info(
                    "Added %d search candidates",
                    len(search_candidates),
                )

                # -------------------------------------------------------------
                # Dedupe / cap
                # -------------------------------------------------------------

                before_dedupe = len(
                    candidates
                )

                candidates = (
                    self._dedupe(
                        candidates
                    )
                )

                candidates = candidates[
                    :self.max_candidates
                ]

                self.logger.info(
                    "Final candidate queue "
                    "before_dedupe=%d after_limit=%d "
                    "max_candidates=%d",
                    before_dedupe,
                    len(candidates),
                    self.max_candidates,
                )

                for number, candidate in enumerate(
                    candidates,
                    1,
                ):
                    self.logger.info(
                        "Candidate queue %d/%d "
                        "method=%s url=%s",
                        number,
                        len(candidates),
                        candidate[
                            "method"
                        ],
                        candidate[
                            "url"
                        ],
                    )

                # -------------------------------------------------------------
                # Candidate evaluation
                # -------------------------------------------------------------

                for candidate_number, candidate in enumerate(
                    candidates,
                    1,
                ):
                    if (
                        supplied
                        and _candidate_key(
                            candidate["url"]
                        )
                        == _candidate_key(
                            supplied
                        )
                    ):
                        self.logger.debug(
                            "Skipping candidate because it "
                            "matches already-tried supplied URL "
                            "url=%s",
                            candidate["url"],
                        )

                        continue

                    self.logger.info(
                        "Evaluating candidate %d/%d",
                        candidate_number,
                        len(candidates),
                    )

                    (
                        document,
                        error,
                    ) = self._fetch_candidate(
                        context,
                        candidate,
                        domain_url,
                        telemetry,
                    )

                    if document:
                        if supplied_failed:
                            document[
                                "limitations"
                            ].append(
                                "The supplied privacy-policy "
                                "URL could not be verified; "
                                "policy discovery located the "
                                "applicable policy used for "
                                "this comparison."
                            )

                        elapsed = (
                            time.perf_counter()
                            - retrieval_started
                        )

                        self.logger.info(
                            "Privacy-policy retrieval succeeded "
                            "method=%s url=%s attempts=%d "
                            "duration=%.3fs",
                            document[
                                "retrieval_method"
                            ],
                            document["url"],
                            len(attempts) + 1,
                            elapsed,
                        )

                        self.logger.info(
                            "Document summary "
                            "title=%r complete=%s "
                            "chars=%d sections=%d "
                            "model_sections=%d coverage=%.4f "
                            "hash=%s",
                            document[
                                "title"
                            ][:160],
                            document[
                                "complete"
                            ],
                            document[
                                "document"
                            ][
                                "extracted_chars"
                            ],
                            len(
                                document[
                                    "document"
                                ][
                                    "sections"
                                ]
                            ),
                            document[
                                "model_context"
                            ][
                                "selected_sections"
                            ],
                            document[
                                "document"
                            ][
                                "coverage"
                            ],
                            document[
                                "document_hash"
                            ],
                        )

                        self.logger.info(
                            "============================================================"
                        )

                        return document

                    self.logger.info(
                        "Candidate rejected "
                        "method=%s url=%s error=%s",
                        candidate[
                            "method"
                        ],
                        candidate[
                            "url"
                        ],
                        error,
                    )

                    attempts.append({
                        "url":
                            candidate[
                                "url"
                            ],
                        "method":
                            candidate[
                                "method"
                            ],
                        "error":
                            error
                            or "unknown retrieval error",
                    })

            finally:
                self.logger.debug(
                    "Closing Playwright browser context"
                )

                try:
                    context.close()
                except Exception as exc:
                    self.logger.warning(
                        "Browser context close failed error=%s",
                        exc,
                    )

                self.logger.debug(
                    "Closing Playwright browser"
                )

                try:
                    browser.close()
                except Exception as exc:
                    self.logger.warning(
                        "Browser close failed error=%s",
                        exc,
                    )

        # ---------------------------------------------------------------------
        # No policy found
        # ---------------------------------------------------------------------

        limitation = (
            "No applicable privacy policy could be "
            "retrieved or verified for the supplied domain."
        )

        if supplied_failed:
            limitation += (
                " The supplied privacy-policy URL "
                "could not be used."
            )

        elapsed = (
            time.perf_counter()
            - retrieval_started
        )

        self.logger.warning(
            "Privacy-policy retrieval failed "
            "domain=%s attempts=%d duration=%.3fs",
            domain_url,
            len(attempts),
            elapsed,
        )

        for index, attempt in enumerate(
            attempts,
            1,
        ):
            self.logger.info(
                "Failed attempt %d/%d "
                "method=%s url=%s error=%s",
                index,
                len(attempts),
                attempt["method"],
                attempt["url"],
                attempt["error"],
            )

        self.logger.info(
            "============================================================"
        )

        return {
            "url": "",
            "found": False,
            "applicable": False,
            "complete": False,
            "title": "",

            "retrieved_at":
                dt.datetime.now(
                    dt.timezone.utc
                )
                .isoformat()
                .replace(
                    "+00:00",
                    "Z",
                ),

            "retrieval_method":
                "none",

            "sections": [],
            "document_hash": "",

            "document": {
                "full_text": "",
                "original_chars": 0,
                "extracted_chars": 0,
                "uncapped_extracted_chars": 0,
                "coverage": 0,
                "extraction_mode":
                    "none",
                "frame_url": "",
                "sections": [],
            },

            "model_context": {
                "max_policy_chars":
                    self.max_model_policy_chars,
                "max_sections":
                    self.max_model_sections,
                "selected_chars": 0,
                "selected_sections": 0,
                "categories":
                    self._detect_categories(
                        telemetry
                    ),
                "sections": [],
            },

            "limitations": [
                limitation
            ],

            "attempted_candidates":
                len(attempts),

            "attempts":
                attempts,
        }
