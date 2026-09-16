"""Safe, host-orchestrated privacy-policy retrieval with Playwright.

The model never launches a browser. The inference host retrieves and verifies the
policy first, then supplies plain policy sections to the model. Search is used only
to discover candidate URLs; search snippets are never treated as policy evidence.
"""

import datetime as dt
import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse


POLICY_LINK_RE = re.compile(
    r"(?:privacy(?:\s+(?:policy|notice|statement))?|data\s+(?:policy|privacy))",
    re.IGNORECASE,
)
POLICY_TEXT_RE = re.compile(
    r"(?:personal (?:data|information)|information we collect|data we collect|"
    r"how we (?:use|share|collect)|your privacy|privacy rights|data protection)",
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
    "/legal/privacy",
    "/legal/privacy-policy",
    "/policies/privacy",
)


@dataclass(frozen=True)
class Candidate:
    url: str
    method: str


def _plain_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower().rstrip(".")


def _same_site(left: str, right: str) -> bool:
    left_host = _host(left)
    right_host = _host(right)
    return bool(
        left_host
        and right_host
        and (
            left_host == right_host
            or left_host.endswith("." + right_host)
            or right_host.endswith("." + left_host)
        )
    )


def _service_tokens(domain_url: str) -> list[str]:
    labels = [
        label
        for label in _host(domain_url).split(".")
        if label not in GENERIC_HOST_LABELS and len(label) >= 3
    ]
    return sorted(labels, key=len, reverse=True)[:3]


def _unwrap_search_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in (parsed.hostname or ""):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return href


class PolicyRetriever:
    def __init__(
        self,
        *,
        timeout_ms: int = 20_000,
        max_policy_chars: int = 24_000,
        max_candidates: int = 14,
        search_enabled: bool = True,
        browser_name: str = "chromium",
        headless: bool = True,
        allow_private_network: bool = False,
    ):
        self.timeout_ms = timeout_ms
        self.max_policy_chars = max_policy_chars
        self.max_candidates = max_candidates
        self.search_enabled = search_enabled
        self.browser_name = browser_name
        self.headless = headless
        self.allow_private_network = allow_private_network
        self._dns_cache: dict[tuple[str, int], tuple[str, ...]] = {}

    def _assert_safe_url(self, url: str) -> str:
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
        if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(
            (".local", ".internal", ".localhost")
        ):
            raise ValueError("local or internal hosts are not allowed")

        cache_key = (hostname, port)
        addresses = self._dns_cache.get(cache_key)
        if addresses is None:
            try:
                info = socket.getaddrinfo(
                    hostname,
                    port,
                    type=socket.SOCK_STREAM,
                )
            except socket.gaierror as exc:
                raise ValueError(f"host did not resolve: {hostname}") from exc
            addresses = tuple(sorted({row[4][0] for row in info}))
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

    def _route(self, route) -> None:
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

    @staticmethod
    def _looks_like_policy(url: str, title: str, body_text: str) -> bool:
        title = _plain_space(title)
        body_text = _plain_space(body_text)
        if len(body_text) < 250:
            return False
        score = 0
        if POLICY_LINK_RE.search(url):
            score += 2
        if POLICY_LINK_RE.search(title):
            score += 4
        if POLICY_LINK_RE.search(body_text[:4_000]):
            score += 2
        if POLICY_TEXT_RE.search(body_text[:12_000]):
            score += 3
        if re.search(r"last updated|effective date", body_text[:4_000], re.I):
            score += 1
        cookie_only = "cookie" in title.lower() and "privacy" not in title.lower()
        return score >= 5 and not cookie_only

    @staticmethod
    def _is_applicable(
        *,
        domain_url: str,
        policy_url: str,
        method: str,
        title: str,
        body_text: str,
    ) -> bool:
        if _same_site(domain_url, policy_url):
            return True
        haystack = re.sub(r"[^a-z0-9]+", "", (title + " " + body_text).lower())
        return any(
            re.sub(r"[^a-z0-9]+", "", token.lower()) in haystack
            for token in _service_tokens(domain_url)
        )

    def _extract_sections(self, page) -> tuple[list[dict], bool]:
        extracted = page.evaluate(
            r"""
            () => {
              const source = document.querySelector('main, article, [role="main"]') || document.body;
              if (!source) return {items: [], bodyLength: 0};
              const root = source.cloneNode(true);
              root.querySelectorAll('script,style,noscript,svg,canvas,nav,header,footer,form,button,input,select,textarea').forEach(n => n.remove());
              const bodyLength = (root.innerText || root.textContent || '').length;
              const nodes = [...root.querySelectorAll('h1,h2,h3,h4,h5,p,li,dt,dd')];
              const items = [];
              let heading = 'Privacy Policy';
              for (const node of nodes) {
                const text = (node.innerText || node.textContent || '').replace(/\s+/g, ' ').trim();
                if (!text) continue;
                if (/^H[1-5]$/.test(node.tagName)) {
                  heading = text.slice(0, 240);
                } else if (text.length >= 20) {
                  items.push({heading, text});
                }
              }
              if (!items.length) {
                const text = (root.innerText || root.textContent || '').replace(/\s+/g, ' ').trim();
                if (text) items.push({heading: document.title || 'Privacy Policy', text});
              }
              return {items, bodyLength};
            }
            """
        )
        items = extracted.get("items", []) if isinstance(extracted, dict) else []
        body_length = extracted.get("bodyLength", 0) if isinstance(extracted, dict) else 0

        sections: list[dict] = []
        total = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            heading = _plain_space(str(item.get("heading") or "Privacy Policy"))[:240]
            text = _plain_space(str(item.get("text") or ""))
            if not text:
                continue
            remaining = self.max_policy_chars - total
            if remaining <= 0:
                break
            text = text[: min(4_000, remaining)]
            if sections and sections[-1]["heading"] == heading:
                room = min(4_000 - len(sections[-1]["text"]), remaining)
                if room > 1:
                    addition = (" " + text)[:room]
                    sections[-1]["text"] += addition
                    total += len(addition)
            else:
                sections.append({"heading": heading, "text": text})
                total += len(text)
            if len(sections) >= 24:
                break
        return sections, bool(body_length > total)

    def _fetch_candidate(
        self,
        context,
        candidate: Candidate,
        domain_url: str,
    ) -> tuple[Optional[dict], Optional[str]]:
        page = context.new_page()
        try:
            self._assert_safe_url(candidate.url)
            response = page.goto(
                candidate.url,
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
            page.wait_for_timeout(500)
            title = _plain_space(page.title())
            body_text = _plain_space(page.locator("body").inner_text(timeout=3_000))
            if not self._looks_like_policy(final_url, title, body_text):
                return None, "page did not verify as a privacy policy"
            if not self._is_applicable(
                domain_url=domain_url,
                policy_url=final_url,
                method=candidate.method,
                title=title,
                body_text=body_text,
            ):
                return None, "policy did not verify as applicable to the supplied domain"
            sections, truncated = self._extract_sections(page)
            if not sections:
                return None, "policy text could not be extracted"
            limitations = []
            if truncated:
                limitations.append(
                    "The retrieved policy was reduced to the extracted sections supplied to the model."
                )
            return {
                "url": final_url,
                "found": True,
                "applicable": True,
                "complete": not truncated,
                "title": title,
                "retrieved_at": dt.datetime.now(dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "retrieval_method": candidate.method,
                "sections": sections,
                "limitations": limitations,
            }, None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        finally:
            page.close()

    def _discover_domain_links(self, context, domain_url: str) -> list[Candidate]:
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
            self._assert_safe_url(page.url)
            links = page.eval_on_selector_all(
                "a[href]",
                """els => els.map(a => ({href: a.href, text: (a.innerText || a.textContent || '').trim(), rel: a.rel || ''}))""",
            )
            ranked = []
            for link in links:
                href = link.get("href") if isinstance(link, dict) else None
                text = link.get("text", "") if isinstance(link, dict) else ""
                if not isinstance(href, str) or not POLICY_LINK_RE.search(
                    f"{text} {href}"
                ):
                    continue
                try:
                    href = self._assert_safe_url(href)
                except ValueError:
                    continue
                score = 2 if _same_site(domain_url, href) else 0
                score += 2 if POLICY_LINK_RE.search(text) else 0
                score += 1 if "privacy" in href.lower() else 0
                ranked.append((score, href))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            return [Candidate(url, "domain_link") for _, url in ranked[:8]]
        except Exception:
            return []
        finally:
            page.close()

    def _search_candidates(self, context, domain_url: str) -> list[Candidate]:
        if not self.search_enabled:
            return []
        page = context.new_page()
        try:
            hostname = _host(domain_url)
            candidates = []
            tokens = _service_tokens(domain_url)
            service_name = tokens[0] if tokens else hostname
            queries = [
                f'site:{hostname} "privacy policy" OR "privacy notice"',
                f'"{service_name}" "privacy policy" OR "privacy notice"',
            ]
            for raw_query in queries:
                search_url = self._assert_safe_url(
                    "https://html.duckduckgo.com/html/?q=" + quote_plus(raw_query)
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
                    "els => els.map(a => ({href: a.href, text: (a.innerText || '').trim()}))",
                )
                for result in results:
                    href = _unwrap_search_url(str(result.get("href") or ""))
                    text = str(result.get("text") or "")
                    if not POLICY_LINK_RE.search(f"{text} {href}"):
                        continue
                    try:
                        href = self._assert_safe_url(href)
                    except ValueError:
                        continue
                    candidates.append(Candidate(href, "web_search"))
                    if len(candidates) >= 8:
                        break
                if len(candidates) >= 8:
                    break
            return candidates
        except Exception:
            return []
        finally:
            page.close()

    @staticmethod
    def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
        output = []
        seen = set()
        for candidate in candidates:
            key = candidate.url.split("#", 1)[0].rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            output.append(candidate)
        return output

    def retrieve(
        self,
        *,
        domain_url: str,
        privacy_policy_url: Optional[str] = None,
    ) -> dict:
        """Retrieve, verify, and extract the applicable policy document."""

        domain_url = self._assert_safe_url(domain_url)
        supplied = None
        supplied_failed = bool(privacy_policy_url)
        if privacy_policy_url:
            try:
                supplied = self._assert_safe_url(privacy_policy_url)
            except (TypeError, ValueError):
                # Invalid and unsafe supplied URLs are never opened. Discovery on
                # the validated domain still runs, as required for stale inputs.
                supplied = None

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Run `pip install -r requirements.txt` "
                "and `playwright install chromium`."
            ) from exc

        attempts: list[tuple[Candidate, str]] = []
        with sync_playwright() as playwright:
            browser_type = getattr(playwright, self.browser_name, None)
            if browser_type is None:
                raise ValueError(
                    f"unsupported browser {self.browser_name!r}; use chromium, firefox, or webkit"
                )
            browser = browser_type.launch(headless=self.headless)
            context = browser.new_context(
                service_workers="block",
                java_script_enabled=True,
                locale="en-US",
                user_agent=(
                    "Mozilla/5.0 (compatible; VeilancePolicyRetriever/1.0; "
                    "+https://veilance.org)"
                ),
            )
            context.set_default_navigation_timeout(self.timeout_ms)
            context.set_default_timeout(min(self.timeout_ms, 5_000))
            context.route("**/*", self._route)
            try:
                if supplied:
                    candidate = Candidate(supplied, "supplied_url")
                    document, error = self._fetch_candidate(
                        context, candidate, domain_url
                    )
                    if document:
                        return document
                    attempts.append((candidate, error or "unknown retrieval error"))

                candidates = self._discover_domain_links(context, domain_url)
                origin = _origin(domain_url)
                candidates.extend(
                    Candidate(urljoin(origin + "/", path), "common_path")
                    for path in COMMON_POLICY_PATHS
                )
                candidates.extend(self._search_candidates(context, domain_url))
                candidates = self._dedupe(candidates)[: self.max_candidates]

                for candidate in candidates:
                    if supplied and candidate.url.rstrip("/") == supplied.rstrip("/"):
                        continue
                    document, error = self._fetch_candidate(
                        context, candidate, domain_url
                    )
                    if document:
                        if supplied_failed:
                            document.setdefault("limitations", []).append(
                                "The supplied privacy-policy URL could not be verified; policy discovery located the applicable policy used for this comparison."
                            )
                        return document
                    attempts.append((candidate, error or "unknown retrieval error"))
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
            "limitations": [limitation],
            "attempted_candidates": len(attempts),
        }
