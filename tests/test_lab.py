import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import check_citations
import finalize_citations
import research
import tools


class RetryTests(unittest.TestCase):
    def test_retry_after_and_no_final_sleep(self):
        fn = Mock(side_effect=[tools.RetryableError("busy", 7), tools.RetryableError("still busy")])
        with patch("tools.time.sleep") as sleep:
            with self.assertRaises(tools.RetryableError):
                tools.with_retry(fn, attempts=2, cap=5)
        sleep.assert_called_once_with(5)

    def test_exponential_retry_then_success(self):
        fn = Mock(side_effect=[tools.RetryableError("busy"), "ok"])
        with patch("tools.random.uniform", return_value=0.25), patch("tools.time.sleep") as sleep:
            self.assertEqual(tools.with_retry(fn, base=1, cap=10), "ok")
        sleep.assert_called_once_with(1.25)

    def test_non_retryable_error_is_not_retried(self):
        fn = Mock(side_effect=ValueError("bug"))
        with patch("tools.time.sleep") as sleep, self.assertRaises(ValueError):
            tools.with_retry(fn)
        self.assertEqual(fn.call_count, 1)
        sleep.assert_not_called()


class SourceToolTests(unittest.TestCase):
    def test_arxiv_query_and_parsing(self):
        xml = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
          <id>http://arxiv.org/abs/2501.00001v2</id><published>2025-01-02T00:00:00Z</published>
          <title> A   useful\n paper </title><summary> Evidence here. </summary>
        </entry></feed>"""
        response = Mock(content=xml)
        with patch("tools._request", return_value=response) as request, \
                patch("tools.time.monotonic", side_effect=[100.0, 100.0]), \
                patch("tools.time.sleep"):
            tools._ARXIV_LAST_CALL = 0.0
            result = tools.arxiv_search.invoke({"query": '"world:model" AND agents', "max_results": 99})
        record = json.loads(result)[0]
        self.assertEqual(record["id"], "2501.00001")
        self.assertEqual(record["url"], "https://arxiv.org/abs/2501.00001")
        params = request.call_args.kwargs["params"]
        self.assertEqual(params["max_results"], 30)
        self.assertNotIn("all:AND", params["search_query"])

    def test_empty_arxiv_query_does_not_call_network(self):
        with patch("tools._request") as request:
            self.assertEqual(tools.arxiv_search.invoke({"query": '::: ""'}), "NO RESULTS")
        request.assert_not_called()

    def test_hf_search_prefers_ai_summary(self):
        response = Mock()
        response.json.return_value = [{"paper": {"id": "2501.2", "title": "Paper", "summary": "long",
                                                   "ai_summary": "short", "upvotes": 2}}]
        with patch("tools._request", return_value=response):
            result = tools.hf_search_papers.invoke({"query": "agents", "limit": 3})
        self.assertEqual(json.loads(result)[0]["summary"], "short")

    def test_exa_key_is_redacted_from_errors(self):
        secret = "example-secret-value"
        with patch.dict("os.environ", {"EXA_API_KEY": secret}), \
                patch("tools._request", side_effect=RuntimeError(f"bad URL ?exaApiKey={secret}")):
            result = tools._exa_call("web_search_exa", {"query": "x", "objective": "y"})
        self.assertTrue(result.startswith("ERROR:"))
        self.assertNotIn(secret, result)

    def test_exa_rate_limit_metadata(self):
        self.assertTrue(tools._exa_rate_limited({"_meta": {"rateLimited": True}}, ""))
        self.assertTrue(tools._exa_rate_limited({}, "Rate limit exceeded. Try later."))

    def test_remote_metadata_verification(self):
        xml = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
          <id>http://arxiv.org/abs/2501.00001v1</id><published>2025-01-02T00:00:00Z</published>
          <title>A Real Paper</title></entry></feed>"""
        arxiv_response = Mock(content=xml)
        hf_response = Mock()
        hf_response.json.return_value = {
            "id": "2501.00002", "title": "Another Real Paper", "publishedAt": "2025-01-03T00:00:00Z"
        }
        sources = [
            {"n": 1, "id": "2501.00001", "title": "A Real Paper", "date": "2025-01-02",
             "source": "arxiv"},
            {"n": 2, "id": "2501.00002", "title": "Another Real Paper", "date": "2025-01-03",
             "source": "hf-search"},
        ]
        tools._ARXIV_LAST_CALL = 0.0
        with patch("tools._request", side_effect=[arxiv_response, hf_response]), \
                patch("tools.time.monotonic", return_value=100.0), patch("tools.time.sleep"):
            self.assertEqual(tools.verify_source_metadata(sources), [])

    def test_remote_metadata_rejects_wrong_title(self):
        xml = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
          <id>http://arxiv.org/abs/2501.00001v1</id>
          <published>2025-01-02T00:00:00Z</published><title>Gas Chromatography</title>
        </entry></feed>"""
        tools._ARXIV_LAST_CALL = 0.0
        with patch("tools._request", return_value=Mock(content=xml)), \
                patch("tools.time.monotonic", return_value=100.0), patch("tools.time.sleep"):
            problems = tools.verify_source_metadata([{
                "n": 1, "id": "2501.00001", "title": "Reinforcement Learning Reasoning",
                "date": "2025-01-02", "source": "arxiv"
            }])
        self.assertTrue(any("title does not match" in problem for problem in problems))

    def test_title_match_allows_omitted_subtitle(self):
        self.assertTrue(tools._similar_title(
            "Small Language Models: Architectures, Techniques, Evaluation",
            "Small Language Models: Architectures, Techniques, Evaluation, Problems and Future Adaptation",
        ))

    def test_remote_metadata_normalizes_relevant_title_and_date(self):
        response = Mock()
        response.json.return_value = {
            "title": "ToolGym: an Open-world Tool-using Environment for Scalable Agent Testing",
            "publishedAt": "2026-01-12T00:00:00Z",
        }
        sources = [{"n": 1, "id": "2601.06328", "title": "Incorrect merged title", "date": "n.d.",
                    "source": "hf-search"}]
        with patch("tools._request", return_value=response):
            problems = tools.verify_source_metadata(
                sources, topic="survey about LLM agents and tool use", normalize=True)
        self.assertEqual(problems, [])
        self.assertTrue(sources[0]["title"].startswith("ToolGym:"))
        self.assertEqual(sources[0]["date"], "2026-01-12")


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "source": "arxiv",
             "date": "2025-01-02"},
            {"n": 2, "id": "2501.00002", "url": "https://huggingface.co/papers/2501.00002",
             "source": "hf-search", "date": "2025-01-03"},
            {"n": 3, "id": "page-c", "url": "https://example.com/c", "source": "web", "date": "n.d."},
        ]

    def test_valid_grouped_citations_and_ignored_code(self):
        report = """# Survey
## TL;DR
Evidence [1, 2] and more [3]. `example [99]` [8](https://example.com).
## Background
Background [1].
```
[77]
```
## Trends and open problems
Open questions [2].
## References
[1] A. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-01)
[2] B. hf-search. https://huggingface.co/papers/2501.00002 (2025-01-01)
[3] C. web. https://example.com/c (2025-01-01)
"""
        self.assertEqual(check_citations.check(report, self.sources), [])

    def test_reports_missing_and_duplicate_references(self):
        report = """## TL;DR
Claim [1].
## Background
Background [1].
## Trends and open problems
Problems [1].
## References
[1] A. web. https://arxiv.org/abs/2501.00001 https://example.com/extra
[1] Duplicate. web. https://arxiv.org/abs/2501.00001
[4] Unknown. web. https://example.com/d
"""
        problems = check_citations.check(report, self.sources)
        joined = "\n".join(problems)
        self.assertIn("source [2] never cited", joined)
        self.assertIn("source [1] has more than one reference line", joined)
        self.assertIn("reference [4] is not present", joined)

    def test_finalizer_output_passes_validator(self):
        report = ("# Survey\n## TL;DR\nA [4, 2].\n## Background\nB [3].\n"
                  "## Trends and open problems\nC [1].\n")
        sources = [
            {"n": 1, "id": "page-c", "url": "https://example.com/c", "title": "C", "source": "web",
             "date": "2025-01-01"},
            {"n": 2, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "A",
             "source": "arxiv",
             "date": "2025-01-01"},
            {"n": 3, "id": "2501.00002", "url": "https://huggingface.co/papers/2501.00002", "title": "B",
             "source": "hf-search",
             "date": "2024-01-01"},
            {"n": 4, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "A duplicate",
             "source": "arxiv",
             "date": "2025-01-01"},
        ]
        finalized, final_sources, problems = finalize_citations.finalize(report, sources)
        self.assertEqual(problems, [])
        self.assertEqual(len(final_sources), 3)
        self.assertEqual(check_citations.check(finalized, final_sources), [])

    def test_rejects_invented_hugging_face_slug(self):
        sources = [dict(source) for source in self.sources]
        sources[1].update(id="hf-search-topic", url="https://huggingface.co/papers/hf-search-topic")
        problems = check_citations.check("## TL;DR\n[1][2][3]\n## Background\n[1]\n"
                                          "## Trends and open problems\n[2]\n## References\n", sources)
        self.assertTrue(any("canonical Hugging Face" in problem for problem in problems))


class ResearchPipelineTests(unittest.TestCase):
    def test_slugify_is_safe_and_bounded(self):
        self.assertEqual(research.slugify("../../Hello, World!"), "hello-world")
        self.assertEqual(research.slugify("!!!"), "topic")
        self.assertLessEqual(len(research.slugify("a" * 100)), 60)

    def test_summarize_counts_calls_and_tokens(self):
        messages = [{"tool_calls": [{"name": "task"}, {"name": "execute"}],
                     "usage_metadata": {"input_tokens": 10, "output_tokens": 4}},
                    {"tool_calls": [{"name": "task"}],
                     "usage_metadata": {"input_tokens": 3, "output_tokens": 2}}]
        summary = research.summarize(messages, 1.26, "model")
        self.assertEqual(summary["subagent_calls"], 2)
        self.assertEqual(summary["tool_calls"], {"execute": 1, "task": 2})
        self.assertEqual(summary["tokens"], {"input": 13, "output": 6})
        self.assertEqual(summary["elapsed_s"], 1.3)

    def test_save_outputs_writes_complete_triplet(self):
        sources = [{"n": 1, "url": "https://example.com", "source": "web"}]
        downloaded = {research.REPORT_PATH: b"# Report\n", research.SOURCES_PATH: json.dumps(sources).encode()}
        with tempfile.TemporaryDirectory() as directory, patch("research.download", return_value=downloaded):
            path = research.save_outputs(object(), "My Topic", [], 2.0, "model", Path(directory))
            self.assertTrue(path.exists())
            meta = json.loads((Path(directory) / "my-topic.meta.json").read_text())
            self.assertEqual(meta["source_families"], ["web"])

    def test_save_outputs_writes_nothing_for_invalid_sources(self):
        downloaded = {research.REPORT_PATH: b"# Report\n", research.SOURCES_PATH: b"not-json"}
        with tempfile.TemporaryDirectory() as directory, patch("research.download", return_value=downloaded):
            with self.assertRaises(RuntimeError):
                research.save_outputs(object(), "My Topic", [], 2.0, "model", Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
