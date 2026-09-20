import json
import unittest
from unittest.mock import patch

from core.browser import BrowserTools, validate_search_query, _safe_public_url


DDG_HTML = b'''<!doctype html><html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Falpha">Alpha Result</a>
  <a class="result__snippet">Useful alpha snippet.</a>
</div>
</body></html>'''

PAGE_HTML = b'''<!doctype html><html><head><title>Example Page</title></head>
<body><main><h1>Heading</h1><p>This is useful page content.</p><script>ignore me</script></main></body></html>'''


class BrowserTests(unittest.TestCase):
    def test_runtime_prefixed_browser_config_is_honored(self):
        browser = BrowserTools({
            "browser_enabled": False,
            "browser_timeout_seconds": 7.5,
            "browser_search_provider": "bing",
            "browser_fetch_max_bytes": 123456,
            "browser_fetch_max_chars": 4321,
        })
        self.assertFalse(browser.enabled)
        self.assertEqual(browser._timeout(), 7.5)
        self.assertEqual(browser._cfg("search_provider", "auto"), "bing")
        self.assertEqual(browser._cfg("fetch_max_bytes", 0), 123456)
        self.assertEqual(browser._cfg("fetch_max_chars", 0), 4321)

    def test_search_query_gate_rejects_prompt_dump(self):
        ok, cleaned = validate_search_query("Granite 4.2 3B model card")
        self.assertTrue(ok)
        self.assertEqual(cleaned, "Granite 4.2 3B model card")
        long_prompt = "Please research this whole thing for me. " * 20
        ok, error = validate_search_query(long_prompt)
        self.assertFalse(ok)
        self.assertTrue("long" in error or "conversational" in error or "paragraph" in error)

    def test_private_and_local_urls_are_blocked(self):
        for url in (
            "http://localhost:8000/",
            "http://127.0.0.1/",
            "http://10.1.2.3/",
            "file:///etc/passwd",
        ):
            ok, _ = _safe_public_url(url)
            self.assertFalse(ok, url)
        with patch("core.browser.socket.getaddrinfo", return_value=[]):
            ok, _ = _safe_public_url("https://example.com/page")
        self.assertTrue(ok)

    def test_duckduckgo_search_parser_and_redirect_decode(self):
        browser = BrowserTools({"browser_search_provider": "duckduckgo"})
        with patch.object(browser, "_read_url", return_value=(DDG_HTML, {"content-type": "text/html; charset=utf-8"}, "https://html.duckduckgo.com/html/")), \
             patch("core.browser._safe_public_url", return_value=(True, "")):
            result = browser.search("alpha query", limit=5)
        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "duckduckgo")
        self.assertEqual(result["results"][0]["url"], "https://example.com/alpha")
        self.assertIn("Useful alpha snippet", result["results"][0]["snippet"])

    def test_fetch_extracts_readable_text_and_title(self):
        browser = BrowserTools({})
        with patch("core.browser._safe_public_url", return_value=(True, "")), \
             patch.object(browser, "_read_url", return_value=(PAGE_HTML, {"content-type": "text/html; charset=utf-8"}, "https://example.com/page")):
            result = browser.fetch("https://example.com/page")
        self.assertTrue(result["ok"])
        self.assertEqual(result["title"], "Example Page")
        self.assertIn("This is useful page content", result["text"])
        self.assertNotIn("ignore me", result["text"])

    def test_current_weather_resolves_nh_and_returns_live_measurement_fields(self):
        browser = BrowserTools({})
        geo = {"results": [
            {"name": "Manchester", "admin1": "England", "country": "United Kingdom", "latitude": 53.48, "longitude": -2.24},
            {"name": "Manchester", "admin1": "New Hampshire", "country": "United States", "latitude": 42.9956, "longitude": -71.4548},
        ]}
        forecast = {
            "timezone": "America/New_York",
            "current": {"time": "2026-09-15T18:00", "temperature_2m": 71.4,
                        "apparent_temperature": 70.1, "relative_humidity_2m": 48,
                        "weather_code": 1, "wind_speed_10m": 6.2},
            "current_units": {"temperature_2m": "°F", "wind_speed_10m": "mp/h"},
        }
        replies = [
            (json.dumps(geo).encode(), {"content-type": "application/json"}, "https://geocoding-api.open-meteo.com/v1/search"),
            (json.dumps(forecast).encode(), {"content-type": "application/json"}, "https://api.open-meteo.com/v1/forecast"),
        ]
        with patch.object(browser, "_read_url", side_effect=replies):
            result = browser.current_weather("Manchester, NH")
        self.assertTrue(result["ok"])
        self.assertIn("New Hampshire", result["resolved_location"])
        self.assertEqual(result["temperature_f"], 71.4)
        self.assertEqual(result["source"], "Open-Meteo")


if __name__ == "__main__":
    unittest.main()
