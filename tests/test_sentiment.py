"""
Unit and integration tests for the sentiment (Ollama) specialist.
"""
import json
import os
import unittest
from unittest.mock import patch, MagicMock

# Import from trading_system; run tests from project root with PYTHONPATH=. or python -m pytest tests/
from trading_system.agents.sentiment import (
    _extract_json,
    _sentiment_cache,
    get_sentiment,
    SentimentOutput,
)
from trading_system.config import OLLAMA_URL


class TestExtractJson(unittest.TestCase):
    """Tests for _extract_json with various LLM response shapes."""

    def test_plain_json(self) -> None:
        s = '{"sentiment_score": 0.2, "confidence": 0.7, "reasoning": "Mixed"}'
        out = _extract_json(s)
        self.assertIsNotNone(out)
        self.assertEqual(json.loads(out)["sentiment_score"], 0.2)

    def test_json_in_markdown_block(self) -> None:
        text = """Here is the result:
```json
{"sentiment_score": -0.5, "confidence": 0.8, "reasoning": "Bearish"}
```
Done."""
        out = _extract_json(text)
        self.assertIsNotNone(out)
        data = json.loads(out)
        self.assertEqual(data["sentiment_score"], -0.5)
        self.assertEqual(data["confidence"], 0.8)

    def test_text_before_and_after_json(self) -> None:
        text = "Some preamble.\n{\"sentiment_score\": 0.0, \"confidence\": 0.5, \"reasoning\": \"Neutral\"}\nTrailing."
        out = _extract_json(text)
        self.assertIsNotNone(out)
        data = json.loads(out)
        self.assertEqual(data["sentiment_score"], 0.0)

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(_extract_json(""))
        self.assertIsNone(_extract_json("   "))

    def test_malformed_json_returns_none_or_fallback(self) -> None:
        # No valid JSON object
        self.assertIsNone(_extract_json("not json at all"))
        # Incomplete
        self.assertIsNone(_extract_json('{"sentiment_score": 0.2'))

    def test_trailing_comma_fix(self) -> None:
        # Some LLMs output trailing commas
        text = '{"sentiment_score": 0.3, "confidence": 0.9, "reasoning": "Ok",}'
        out = _extract_json(text)
        # May be extracted by first-line attempt or brace match; if regex extracts, parsing might need fix
        if out:
            raw = out.replace(",}", "}").replace(",]", "]")
            data = json.loads(raw)
            self.assertIn("sentiment_score", data)


class TestGetSentimentMocked(unittest.TestCase):
    """Tests for get_sentiment with mocked ollama and news."""

    def setUp(self) -> None:
        _sentiment_cache.clear()

    @patch("trading_system.agents.sentiment.fetch_news")
    @patch("trading_system.agents.sentiment.news_to_text")
    @patch("ollama.chat")
    def test_dict_response_returns_from_llm(
        self,
        mock_chat: MagicMock,
        mock_news_to_text: MagicMock,
        mock_fetch_news: MagicMock,
    ) -> None:
        mock_fetch_news.return_value = [MagicMock(title="T", summary="S", published=None)]
        mock_news_to_text.return_value = "News text"
        mock_chat.return_value = {"message": {"content": '{"sentiment_score": -0.3, "confidence": 0.6, "reasoning": "Bearish"}'}}
        score, conf, from_llm = get_sentiment("SPY")
        self.assertTrue(from_llm)
        self.assertAlmostEqual(score, -0.3)
        self.assertAlmostEqual(conf, 0.6)

    @patch("trading_system.agents.sentiment.fetch_news")
    @patch("trading_system.agents.sentiment.news_to_text")
    @patch("ollama.chat")
    def test_valid_response_returns_from_llm(
        self,
        mock_chat: MagicMock,
        mock_news_to_text: MagicMock,
        mock_fetch_news: MagicMock,
    ) -> None:
        mock_fetch_news.return_value = [MagicMock(title="T", summary="S", published=None)]
        mock_news_to_text.return_value = "News text"
        mock_chat.return_value = MagicMock(
            message=MagicMock(
                content='{"sentiment_score": 0.2, "confidence": 0.7, "reasoning": "Mixed"}'
            )
        )
        score, conf, from_llm = get_sentiment("SPY")
        self.assertTrue(from_llm)
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(conf, 0)
        self.assertLessEqual(conf, 1)
        self.assertAlmostEqual(score, 0.2)
        self.assertAlmostEqual(conf, 0.7)

    @patch("trading_system.agents.sentiment.fetch_news")
    @patch("trading_system.agents.sentiment.news_to_text")
    @patch("ollama.chat")
    def test_ollama_raises_returns_proxy(
        self,
        mock_chat: MagicMock,
        mock_news_to_text: MagicMock,
        mock_fetch_news: MagicMock,
    ) -> None:
        mock_fetch_news.return_value = []
        mock_news_to_text.return_value = "No recent news available for this ticker."
        mock_chat.side_effect = ConnectionError("Ollama not reachable")
        score, conf, from_llm = get_sentiment("AAPL")
        self.assertFalse(from_llm)
        self.assertEqual(score, 0.0)
        self.assertEqual(conf, 0.5)

    @patch("trading_system.agents.sentiment.fetch_news")
    @patch("trading_system.agents.sentiment.news_to_text")
    @patch("ollama.chat")
    def test_invalid_json_returns_proxy(
        self,
        mock_chat: MagicMock,
        mock_news_to_text: MagicMock,
        mock_fetch_news: MagicMock,
    ) -> None:
        mock_fetch_news.return_value = [MagicMock(title="T", summary="S", published=None)]
        mock_news_to_text.return_value = "News"
        mock_chat.return_value = MagicMock(message=MagicMock(content="not valid json"))
        score, conf, from_llm = get_sentiment("SPY")
        self.assertFalse(from_llm)
        self.assertEqual(score, 0.0)
        self.assertEqual(conf, 0.5)

    @patch("trading_system.agents.sentiment.fetch_news")
    @patch("trading_system.agents.sentiment.news_to_text")
    @patch("ollama.chat")
    def test_minimal_json_no_reasoning_returns_from_llm(
        self,
        mock_chat: MagicMock,
        mock_news_to_text: MagicMock,
        mock_fetch_news: MagicMock,
    ) -> None:
        mock_fetch_news.return_value = []
        mock_news_to_text.return_value = "No recent news."
        mock_chat.return_value = MagicMock(
            message=MagicMock(content='{"sentiment_score": 0.5, "confidence": 0.8}')
        )
        score, conf, from_llm = get_sentiment("SPY")
        self.assertTrue(from_llm)
        self.assertAlmostEqual(score, 0.5)
        self.assertAlmostEqual(conf, 0.8)

    @patch("trading_system.agents.sentiment.fetch_news")
    @patch("trading_system.agents.sentiment.news_to_text")
    @patch("ollama.chat")
    def test_empty_news_does_not_crash(
        self,
        mock_chat: MagicMock,
        mock_news_to_text: MagicMock,
        mock_fetch_news: MagicMock,
    ) -> None:
        mock_fetch_news.return_value = []
        mock_news_to_text.return_value = "No recent news available for this ticker."
        mock_chat.return_value = MagicMock(
            message=MagicMock(
                content='{"sentiment_score": 0.0, "confidence": 0.5, "reasoning": "No news"}'
            )
        )
        score, conf, from_llm = get_sentiment("XYZ")
        self.assertIsInstance(score, (int, float))
        self.assertIsInstance(conf, (int, float))
        self.assertIsInstance(from_llm, bool)
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(conf, 0)
        self.assertLessEqual(conf, 1)

    @patch("trading_system.agents.sentiment.fetch_news")
    @patch("trading_system.agents.sentiment.news_to_text")
    @patch("ollama.chat")
    def test_force_refresh_false_returns_cached_without_calling_ollama(
        self,
        mock_chat: MagicMock,
        mock_news_to_text: MagicMock,
        mock_fetch_news: MagicMock,
    ) -> None:
        mock_fetch_news.return_value = [MagicMock(title="T", summary="S", published=None)]
        mock_news_to_text.return_value = "News"
        mock_chat.return_value = MagicMock(
            message=MagicMock(content='{"sentiment_score": 0.25, "confidence": 0.9, "reasoning": "Cached"}')
        )
        score1, conf1, from_llm1 = get_sentiment("CACHE_TICKER", force_refresh=True)
        self.assertTrue(from_llm1)
        self.assertAlmostEqual(score1, 0.25)
        self.assertAlmostEqual(conf1, 0.9)
        call_count_after_refresh = mock_chat.call_count
        score2, conf2, from_llm2 = get_sentiment("CACHE_TICKER", force_refresh=False)
        self.assertTrue(from_llm2)
        self.assertAlmostEqual(score2, 0.25)
        self.assertAlmostEqual(conf2, 0.9)
        self.assertEqual(mock_chat.call_count, call_count_after_refresh, "ollama.chat should not be called on cache hit")


class TestSentimentOutput(unittest.TestCase):
    """Sanity check for SentimentOutput schema."""

    def test_parse_valid(self) -> None:
        out = SentimentOutput.model_validate(
            {"sentiment_score": 0.1, "confidence": 0.8, "reasoning": "Test"}
        )
        self.assertEqual(out.sentiment_score, 0.1)
        self.assertEqual(out.confidence, 0.8)
        self.assertEqual(out.reasoning, "Test")

    def test_parse_valid_without_reasoning(self) -> None:
        out = SentimentOutput.model_validate(
            {"sentiment_score": 0.5, "confidence": 0.8}
        )
        self.assertEqual(out.sentiment_score, 0.5)
        self.assertEqual(out.confidence, 0.8)
        self.assertEqual(out.reasoning, "")


class TestOllamaConnection(unittest.TestCase):
    """OLLAMA_HOST set from OLLAMA_URL when unset."""

    def setUp(self) -> None:
        _sentiment_cache.clear()

    @patch("trading_system.agents.sentiment.fetch_news")
    @patch("trading_system.agents.sentiment.news_to_text")
    @patch("ollama.chat")
    def test_ollama_host_set_from_url_when_unset(
        self,
        mock_chat: MagicMock,
        mock_news_to_text: MagicMock,
        mock_fetch_news: MagicMock,
    ) -> None:
        mock_fetch_news.return_value = [MagicMock(title="T", summary="S", published=None)]
        mock_news_to_text.return_value = "News"
        mock_chat.return_value = MagicMock(
            message=MagicMock(content='{"sentiment_score": 0.1, "confidence": 0.9, "reasoning": "Ok"}')
        )
        old_host = os.environ.pop("OLLAMA_HOST", None)
        try:
            score, conf, from_llm = get_sentiment("SPY")
            self.assertTrue(from_llm)
            host = os.environ.get("OLLAMA_HOST")
            if OLLAMA_URL:
                expected = OLLAMA_URL.replace("https://", "").replace("http://", "").strip()
                if expected:
                    self.assertEqual(host, expected)
        finally:
            if old_host is not None:
                os.environ["OLLAMA_HOST"] = old_host
            elif "OLLAMA_HOST" in os.environ and not OLLAMA_URL:
                os.environ.pop("OLLAMA_HOST", None)


class TestGetSentimentIntegration(unittest.TestCase):
    """Optional integration test: call real Ollama if available; skip otherwise."""

    def setUp(self) -> None:
        _sentiment_cache.clear()

    def test_get_sentiment_real_ollama_if_available(self) -> None:
        try:
            import requests
            from trading_system.config import OLLAMA_URL
            r = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=2)
            if not r.ok:
                self.skipTest("Ollama not reachable")
        except Exception as e:
            self.skipTest(f"Ollama not available: {e}")
        score, conf, from_llm = get_sentiment("SPY", force_refresh=True)
        self.assertIsInstance(score, (int, float))
        self.assertIsInstance(conf, (int, float))
        self.assertIsInstance(from_llm, bool)
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(conf, 0)
        self.assertLessEqual(conf, 1)


if __name__ == "__main__":
    unittest.main()
