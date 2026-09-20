from __future__ import annotations

import gzip
import io
import html
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Tuple


_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36 AevumAI/0.2.4"
)

_WMO_WEATHER = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "moderate drizzle",
    55: "dense drizzle", 56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain", 66: "light freezing rain",
    67: "heavy freezing rain", 71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains", 80: "slight rain showers", 81: "moderate rain showers",
    82: "violent rain showers", 85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def _safe_public_url(url: str) -> Tuple[bool, str]:
    """Validate an outbound URL and reject local/private targets.

    This is intentionally conservative because URLs can be proposed by an LLM.
    """
    try:
        parsed = urllib.parse.urlsplit(str(url or "").strip())
    except Exception:
        return False, "invalid URL"
    if parsed.scheme.lower() not in {"http", "https"}:
        return False, "only http/https URLs are allowed"
    if not parsed.hostname:
        return False, "URL is missing a hostname"
    if parsed.username or parsed.password:
        return False, "URLs with embedded credentials are not allowed"
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        return False, "local URLs are not allowed"
    try:
        ip = ipaddress.ip_address(host)
        ips = [ip]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
            ips = []
            for info in infos:
                try:
                    ips.append(ipaddress.ip_address(info[4][0]))
                except Exception:
                    pass
        except socket.gaierror:
            # DNS may be unavailable temporarily. The actual fetch will report that
            # cleanly; a syntactically valid public hostname is not rejected here.
            ips = []
        except Exception:
            ips = []
    for ip in ips:
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False, f"non-public target is not allowed ({ip})"
    return True, ""


def validate_search_query(query: str) -> Tuple[bool, str]:
    q = _collapse_ws(query)
    if not q:
        return False, "empty query"
    if len(q) > 220:
        return False, "query is too long; use a concise targeted search query"
    words = q.split()
    if len(words) > 28:
        return False, "query is too conversational; reduce it to the key entities and fact needed"
    # Prevent the old failure mode where a model copies an entire user prompt into search.
    sentence_marks = sum(q.count(x) for x in ("?", "!", "."))
    if len(words) > 18 and sentence_marks >= 2:
        return False, "query resembles a paragraph; compile a short search-engine query"
    return True, q


class _TextExtractor(HTMLParser):
    BLOCK = {
        "p", "div", "section", "article", "main", "header", "footer", "aside",
        "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "tr", "td", "th",
    }
    IGNORE = {"script", "style", "noscript", "svg", "canvas", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth_ignore = 0
        self.parts: List[str] = []
        self.title_parts: List[str] = []
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t in self.IGNORE:
            self.depth_ignore += 1
            return
        if self.depth_ignore:
            return
        if t == "title":
            self.in_title = True
        if t in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self.IGNORE:
            if self.depth_ignore:
                self.depth_ignore -= 1
            return
        if self.depth_ignore:
            return
        if t == "title":
            self.in_title = False
        if t in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.depth_ignore:
            return
        text = data.strip()
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
        self.parts.append(text + " ")

    def result(self) -> Tuple[str, str]:
        title = _collapse_ws(" ".join(self.title_parts))
        raw = "".join(self.parts)
        lines = []
        for line in raw.splitlines():
            line = _collapse_ws(line)
            if line:
                lines.append(line)
        # Remove exact duplicate adjacent blocks, common in responsive/nav markup.
        deduped: List[str] = []
        for line in lines:
            if not deduped or deduped[-1] != line:
                deduped.append(line)
        return title, "\n".join(deduped)


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[Dict[str, str]] = []
        self.current: Optional[Dict[str, str]] = None
        self.capture_title = False
        self.capture_snippet = False
        self.title_buf: List[str] = []
        self.snippet_buf: List[str] = []

    @staticmethod
    def _attrs(attrs: List[Tuple[str, Optional[str]]]) -> Dict[str, str]:
        return {k.lower(): (v or "") for k, v in attrs}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        a = self._attrs(attrs)
        cls = a.get("class", "")
        if tag.lower() == "a" and "result__a" in cls:
            href = a.get("href", "")
            self.current = {"title": "", "url": href, "snippet": ""}
            self.title_buf = []
            self.capture_title = True
        elif self.current is not None and ("result__snippet" in cls or "result-snippet" in cls):
            self.capture_snippet = True
            self.snippet_buf = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "a" and self.capture_title and self.current is not None:
            self.capture_title = False
            self.current["title"] = _collapse_ws(" ".join(self.title_buf))
            return
        if self.capture_snippet and t in {"a", "div", "span"} and self.current is not None:
            self.capture_snippet = False
            self.current["snippet"] = _collapse_ws(" ".join(self.snippet_buf))
            self.results.append(self.current)
            self.current = None

    def handle_data(self, data: str) -> None:
        if self.capture_title:
            self.title_buf.append(data)
        elif self.capture_snippet:
            self.snippet_buf.append(data)

    def close(self) -> None:
        super().close()
        if self.current is not None and self.current.get("title"):
            self.results.append(self.current)
            self.current = None


class _BingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[Dict[str, str]] = []
        self.in_algo = 0
        self.capture_title = False
        self.capture_snippet = False
        self.current: Optional[Dict[str, str]] = None
        self.title_buf: List[str] = []
        self.snippet_buf: List[str] = []

    @staticmethod
    def _attrs(attrs: List[Tuple[str, Optional[str]]]) -> Dict[str, str]:
        return {k.lower(): (v or "") for k, v in attrs}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        a = self._attrs(attrs)
        cls = a.get("class", "")
        t = tag.lower()
        if t == "li" and "b_algo" in cls:
            self.in_algo += 1
            self.current = {"title": "", "url": "", "snippet": ""}
        elif self.in_algo and t == "a" and self.current is not None and not self.current.get("url"):
            href = a.get("href", "")
            if href.startswith("http"):
                self.current["url"] = href
                self.capture_title = True
                self.title_buf = []
        elif self.in_algo and t in {"p", "div"} and self.current is not None and (t == "p" or "b_caption" in cls):
            self.capture_snippet = True
            self.snippet_buf = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "a" and self.capture_title and self.current is not None:
            self.capture_title = False
            self.current["title"] = _collapse_ws(" ".join(self.title_buf))
        if t in {"p", "div"} and self.capture_snippet and self.current is not None:
            self.capture_snippet = False
            snippet = _collapse_ws(" ".join(self.snippet_buf))
            if snippet and not self.current.get("snippet"):
                self.current["snippet"] = snippet
        if t == "li" and self.in_algo:
            self.in_algo -= 1
            if self.in_algo == 0 and self.current is not None:
                if self.current.get("url") and self.current.get("title"):
                    self.results.append(self.current)
                self.current = None

    def handle_data(self, data: str) -> None:
        if self.capture_title:
            self.title_buf.append(data)
        elif self.capture_snippet:
            self.snippet_buf.append(data)


def _decode_ddg_url(url: str) -> str:
    u = html.unescape(url or "")
    if u.startswith("//"):
        u = "https:" + u
    try:
        parsed = urllib.parse.urlsplit(u)
        params = urllib.parse.parse_qs(parsed.query)
        if "uddg" in params and params["uddg"]:
            return urllib.parse.unquote(params["uddg"][0])
    except Exception:
        pass
    return u


def _dedupe_results(rows: Iterable[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for row in rows:
        url = str(row.get("url", "") or "").strip()
        title = _collapse_ws(str(row.get("title", "") or ""))
        snippet = _collapse_ws(str(row.get("snippet", "") or ""))
        if not url or not title:
            continue
        key = url.split("#", 1)[0]
        if key in seen:
            continue
        ok, _ = _safe_public_url(url)
        if not ok:
            continue
        seen.add(key)
        out.append({"title": title[:300], "url": url[:2000], "snippet": snippet[:1200]})
        if len(out) >= limit:
            break
    return out


class BrowserTools:
    """Small in-process web search/fetch layer for the Executive model.

    No localhost service or API key is required. Search providers are HTML endpoints
    with automatic fallback. The caller is expected to pass results back to the model
    as private evidence rather than as dialogue.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config if config is not None else {}

    def _cfg(self, name: str, default: Any) -> Any:
        # BrowserTools receives the application's runtime config directly.  Accept
        # both browser_* application keys and unprefixed keys so the component can
        # also be unit-tested/embedded independently.
        prefixed = f"browser_{name}"
        if prefixed in self.config:
            return self.config.get(prefixed, default)
        return self.config.get(name, default)

    @property
    def enabled(self) -> bool:
        return bool(self._cfg("enabled", True))

    def _timeout(self) -> float:
        return max(3.0, min(45.0, float(self._cfg("timeout_seconds", 12.0) or 12.0)))

    def _ua(self) -> str:
        return str(self._cfg("user_agent", _DEFAULT_UA) or _DEFAULT_UA)

    def _read_url(self, url: str, *, max_bytes: int) -> Tuple[bytes, Dict[str, str], str]:
        ok, reason = _safe_public_url(url)
        if not ok:
            raise ValueError(reason)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self._ua(),
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.2",
                "Accept-Language": "en-US,en;q=0.8",
                "Accept-Encoding": "gzip",
                "Connection": "close",
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout()) as resp:
            final_url = str(resp.geturl() or url)
            ok, reason = _safe_public_url(final_url)
            if not ok:
                raise ValueError(f"redirected to blocked URL: {reason}")
            raw = resp.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raw = raw[:max_bytes]
            headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            if "gzip" in headers.get("content-encoding", "").lower():
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                        raw = gz.read(max_bytes + 1)
                    if len(raw) > max_bytes:
                        raw = raw[:max_bytes]
                except Exception:
                    pass
            return raw, headers, final_url

    @staticmethod
    def _decode(raw: bytes, headers: Dict[str, str]) -> str:
        ctype = headers.get("content-type", "")
        m = re.search(r"charset=([\w.\-]+)", ctype, flags=re.I)
        charset = m.group(1) if m else "utf-8"
        try:
            return raw.decode(charset, errors="replace")
        except LookupError:
            return raw.decode("utf-8", errors="replace")

    def search(self, query: str, *, limit: int = 8) -> Dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "browser tools are disabled", "query": query, "results": []}
        valid, cleaned_or_error = validate_search_query(query)
        if not valid:
            return {"ok": False, "error": cleaned_or_error, "query": _collapse_ws(query), "results": []}
        q = cleaned_or_error
        limit = max(1, min(12, int(limit or 8)))
        provider = str(self._cfg("search_provider", "auto") or "auto").strip().lower()
        order = [provider] if provider in {"duckduckgo", "bing"} else ["duckduckgo", "bing"]
        errors: List[str] = []
        for name in order:
            try:
                if name == "duckduckgo":
                    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(q)
                    raw, headers, _ = self._read_url(url, max_bytes=1_500_000)
                    text = self._decode(raw, headers)
                    parser = _DuckDuckGoParser()
                    parser.feed(text)
                    parser.close()
                    rows = []
                    for row in parser.results:
                        r = dict(row)
                        r["url"] = _decode_ddg_url(r.get("url", ""))
                        rows.append(r)
                else:
                    url = "https://www.bing.com/search?q=" + urllib.parse.quote_plus(q) + f"&count={limit}"
                    raw, headers, _ = self._read_url(url, max_bytes=1_500_000)
                    text = self._decode(raw, headers)
                    parser = _BingParser()
                    parser.feed(text)
                    parser.close()
                    rows = parser.results
                results = _dedupe_results(rows, limit)
                if results:
                    return {"ok": True, "query": q, "provider": name, "results": results}
                errors.append(f"{name}: no parseable results")
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        return {"ok": False, "query": q, "provider": order[-1] if order else "", "results": [], "error": "; ".join(errors)[:1200]}

    def current_weather(self, location: str) -> Dict[str, Any]:
        """Retrieve current conditions from Open-Meteo's geocoding/forecast APIs."""
        if not self.enabled:
            return {"ok": False, "location": location, "error": "browser tools are disabled"}
        place = _collapse_ws(location)[:200]
        if not place:
            return {"ok": False, "location": place, "error": "location is required"}
        try:
            lookup_name = re.split(r",|\b(?:new hampshire|nh|massachusetts|ma)\b", place, maxsplit=1, flags=re.I)[0].strip() or place
            geo_url = (
                "https://geocoding-api.open-meteo.com/v1/search?name="
                + urllib.parse.quote_plus(lookup_name)
                + "&count=5&language=en&format=json"
            )
            raw, headers, _ = self._read_url(geo_url, max_bytes=300_000)
            geo = json.loads(self._decode(raw, headers))
            candidates = list(geo.get("results", []) or [])
            if not candidates:
                return {"ok": False, "location": place, "error": "location was not found"}
            # Prefer US administrative matches when the request explicitly includes a
            # state name/abbreviation; otherwise retain provider ranking.
            lowered = f" {place.lower()} "
            state_hints = {
                " nh ": "new hampshire", " new hampshire ": "new hampshire",
                " ma ": "massachusetts", " massachusetts ": "massachusetts",
            }
            wanted_admin = next((name for hint, name in state_hints.items() if hint in lowered), "")
            chosen = candidates[0]
            if wanted_admin:
                chosen = next((row for row in candidates if str(row.get("admin1", "")).lower() == wanted_admin), chosen)
            lat = float(chosen["latitude"]); lon = float(chosen["longitude"])
            forecast_url = (
                "https://api.open-meteo.com/v1/forecast?latitude=" + urllib.parse.quote(str(lat))
                + "&longitude=" + urllib.parse.quote(str(lon))
                + "&current=temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m"
                + "&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto"
            )
            raw, headers, final_url = self._read_url(forecast_url, max_bytes=300_000)
            payload = json.loads(self._decode(raw, headers))
            current = dict(payload.get("current", {}) or {})
            units = dict(payload.get("current_units", {}) or {})
            if "temperature_2m" not in current:
                return {"ok": False, "location": place, "error": "weather service returned no current temperature"}
            resolved = ", ".join(x for x in (
                str(chosen.get("name", "") or ""), str(chosen.get("admin1", "") or ""),
                str(chosen.get("country", "") or ""),
            ) if x)
            return {
                "ok": True,
                "location": place,
                "resolved_location": resolved,
                "latitude": lat,
                "longitude": lon,
                "timezone": payload.get("timezone", ""),
                "observed_at": current.get("time", ""),
                "temperature_f": current.get("temperature_2m"),
                "apparent_temperature_f": current.get("apparent_temperature"),
                "relative_humidity_percent": current.get("relative_humidity_2m"),
                "weather_code": current.get("weather_code"),
                "condition": _WMO_WEATHER.get(int(current.get("weather_code") if current.get("weather_code") is not None else -1), "unknown"),
                "wind_speed_mph": current.get("wind_speed_10m"),
                "units": units,
                "source": "Open-Meteo",
                "source_url": final_url,
            }
        except Exception as exc:
            return {"ok": False, "location": place, "error": str(exc)[:1200]}

    def fetch(self, url: str, *, max_chars: Optional[int] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "url": url, "error": "browser tools are disabled"}
        ok, reason = _safe_public_url(url)
        if not ok:
            return {"ok": False, "url": url, "error": reason}
        max_bytes = max(64_000, min(5_000_000, int(self._cfg("fetch_max_bytes", 2_000_000) or 2_000_000)))
        max_chars_i = max(2_000, min(60_000, int(max_chars or self._cfg("fetch_max_chars", 18_000) or 18_000)))
        try:
            raw, headers, final_url = self._read_url(url, max_bytes=max_bytes)
            ctype = headers.get("content-type", "").lower()
            text = self._decode(raw, headers)
            if "html" in ctype or "<html" in text[:1000].lower() or "<!doctype" in text[:1000].lower():
                parser = _TextExtractor()
                parser.feed(text)
                parser.close()
                title, body = parser.result()
            elif ctype.startswith("text/") or not ctype:
                title, body = "", _collapse_ws(text)
            else:
                return {
                    "ok": False, "url": final_url, "content_type": ctype,
                    "error": "web_fetch currently supports HTML and text pages only",
                }
            body = body[:max_chars_i]
            if not body.strip():
                return {"ok": False, "url": final_url, "title": title, "content_type": ctype, "error": "page contained no readable text"}
            return {
                "ok": True,
                "url": final_url,
                "title": title[:500],
                "content_type": ctype,
                "text": body,
                "truncated": len(body) >= max_chars_i,
            }
        except Exception as exc:
            return {"ok": False, "url": url, "error": str(exc)[:1200]}
