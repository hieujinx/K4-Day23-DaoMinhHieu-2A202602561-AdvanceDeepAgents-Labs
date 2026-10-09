"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import random
import re
import threading
import time  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)
from difflib import SequenceMatcher
from html import unescape

import httpx  # noqa: F401
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(cap, max(0.0, float(exc.retry_after)))
            else:
                delay = min(cap, base * (2 ** attempt) + random.uniform(0.0, max(0.0, base)))
            time.sleep(delay)


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST_CALL = 0.0


def _retry_after(response):
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _request(method, url, **kwargs):
    """Make one HTTP request and turn only transient failures into RetryableError."""
    try:
        response = httpx.request(method, url, timeout=30.0, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(str(exc)) from exc
    if response.status_code in _RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {response.status_code}", _retry_after(response))
    response.raise_for_status()
    return response


def _compact(value, limit=600):
    return " ".join(str(value or "").split())[:limit]


def _error(exc, secret=""):
    message = f"{type(exc).__name__}: {exc}"
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return f"ERROR: {message}"


def _hf_record(item, prefer_ai=False):
    paper = item.get("paper") if isinstance(item, dict) else None
    if not isinstance(paper, dict):
        return None
    paper_id = str(paper.get("id") or "").strip()
    if not paper_id:
        return None
    summary = paper.get("ai_summary") if prefer_ai else None
    summary = summary or paper.get("summary") or item.get("summary")
    published = paper.get("publishedAt") or item.get("publishedAt") or ""
    return {
        "id": paper_id,
        "url": f"https://huggingface.co/papers/{paper_id}",
        "published": str(published)[:10],
        "title": _compact(paper.get("title") or item.get("title")),
        "summary": _compact(summary),
        "upvotes": paper.get("upvotes") or item.get("upvotes") or 0,
        "github": paper.get("githubRepo") or item.get("githubRepo") or "",
        "stars": paper.get("githubStars") or item.get("githubStars") or 0,
    }


def _similar_title(left, right):
    normalize = lambda value: " ".join(re.findall(r"[\w]+", str(value or "").casefold()))
    left_normalized, right_normalized = normalize(left), normalize(right)
    if min(len(left_normalized), len(right_normalized)) >= 30 \
            and (left_normalized in right_normalized or right_normalized in left_normalized):
        return True
    return SequenceMatcher(None, left_normalized, right_normalized).ratio() >= 0.82


def verify_source_metadata(sources, topic="", normalize=False):
    """Verify structured scholarly sources against their official metadata APIs.

    This runs on the host after sandbox validation. It returns problems rather than
    raising so callers can feed precise corrections back to the lead agent.
    """
    problems = []
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    arxiv_sources = [source for source in sources if source.get("source") == "arxiv"]
    official_arxiv = {}
    arxiv_batch_error = None
    if arxiv_sources:
        ids = [str(source.get("id") or "") for source in arxiv_sources]

        def call_arxiv_batch():
            global _ARXIV_LAST_CALL
            with _ARXIV_LOCK:
                delay = 3.0 - (time.monotonic() - _ARXIV_LAST_CALL)
                if delay > 0:
                    time.sleep(delay)
                _ARXIV_LAST_CALL = time.monotonic()
            return _request("GET", ARXIV_URL, params={"id_list": ",".join(ids), "max_results": len(ids)})

        try:
            response = with_retry(call_arxiv_batch, attempts=2, cap=5.0)
            root = xml.etree.ElementTree.fromstring(response.content)
            for entry in root.findall("atom:entry", namespace):
                raw_id = _compact(entry.findtext("atom:id", default="", namespaces=namespace))
                paper_id = re.sub(r"v\d+$", "", raw_id.rsplit("/", 1)[-1])
                official_arxiv[paper_id] = (
                    _compact(entry.findtext("atom:title", default="", namespaces=namespace), 2000),
                    _compact(entry.findtext("atom:published", default="", namespaces=namespace))[:10],
                )
        except Exception as exc:
            arxiv_batch_error = f"{type(exc).__name__}: {exc}"

    for source in sources:
        family = source.get("source")
        source_id = str(source.get("id") or "")
        try:
            if family == "arxiv":
                if source_id in official_arxiv:
                    actual_title, actual_date = official_arxiv[source_id]
                else:
                    # Shared classroom IPs can remain arXiv-rate-limited for minutes.
                    # Hugging Face mirrors metadata by arXiv ID and is a safe fallback.
                    try:
                        response = with_retry(lambda: _request(
                            "GET", f"https://huggingface.co/api/papers/{source_id}"))
                        data = response.json()
                        actual_title = _compact(data.get("title"), 2000)
                        actual_date = str(data.get("publishedAt") or "")[:10]
                    except Exception as fallback_exc:
                        try:
                            page = with_retry(lambda: _request("GET", source.get("url")), attempts=3, cap=10.0)
                            title_match = re.search(
                                r'<meta\s+name=["\']citation_title["\']\s+content=["\'](.*?)["\']',
                                page.text, re.I | re.S,
                            )
                            date_match = re.search(
                                r'<meta\s+name=["\']citation_date["\']\s+content=["\'](.*?)["\']',
                                page.text, re.I | re.S,
                            )
                            if not title_match:
                                raise ValueError("arXiv page has no citation_title metadata")
                            actual_title = _compact(unescape(title_match.group(1)), 2000)
                            raw_date = date_match.group(1) if date_match else ""
                            actual_date = raw_date.replace("/", "-")[:10]
                        except Exception as page_exc:
                            detail = f"; arXiv batch failed with {arxiv_batch_error}" if arxiv_batch_error else ""
                            problems.append(
                                f"source [{source.get('n')}] arXiv id {source_id} could not be verified: "
                                f"mirror={type(fallback_exc).__name__}: {fallback_exc}; "
                                f"page={type(page_exc).__name__}: {page_exc}{detail}"
                            )
                            continue
            elif family in {"hf-daily", "hf-search"}:
                response = with_retry(lambda: _request(
                    "GET", f"https://huggingface.co/api/papers/{source_id}"))
                data = response.json()
                actual_title = _compact(data.get("title"), 2000)
                actual_date = str(data.get("publishedAt") or "")[:10]
            else:
                continue
            if not _similar_title(source.get("title"), actual_title):
                topic_terms = {term.rstrip("s") for term in re.findall(r"[a-z0-9]+", topic.casefold())
                               if len(term) >= 4 and term not in {"survey", "about", "language", "model"}}
                title_terms = {term.rstrip("s") for term in re.findall(r"[a-z0-9]+", actual_title.casefold())}
                if normalize and topic_terms & title_terms:
                    source["title"] = actual_title
                else:
                    problems.append(
                        f"source [{source.get('n')}] title does not match {family} id {source_id}; "
                        f"official title is {actual_title!r}"
                    )
            if actual_date and str(source.get("date") or "")[:7] != actual_date[:7]:
                if normalize:
                    source["date"] = actual_date
                else:
                    problems.append(
                        f"source [{source.get('n')}] date {source.get('date')!r} does not match "
                        f"official month {actual_date[:7]!r}"
                    )
        except Exception as exc:
            problems.append(f"source [{source.get('n')}] remote metadata verification failed: {type(exc).__name__}: {exc}")
    return problems


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    terms = [term for term in re.findall(r"[\w-]+", query or "", flags=re.UNICODE)
             if term.upper() not in {"AND", "OR", "NOT"}]
    if not terms:
        return "NO RESULTS"

    def call():
        global _ARXIV_LAST_CALL
        with _ARXIV_LOCK:
            delay = 3.0 - (time.monotonic() - _ARXIV_LAST_CALL)
            if delay > 0:
                time.sleep(delay)
            _ARXIV_LAST_CALL = time.monotonic()
        return _request("GET", ARXIV_URL, params={
            "search_query": " AND ".join(f"all:{term}" for term in terms),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max(1, min(int(max_results), 30)),
            "start": 0,
        })

    try:
        response = with_retry(call, attempts=7, cap=60.0)
        root = xml.etree.ElementTree.fromstring(response.content)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", namespace):
            raw_id = _compact(entry.findtext("atom:id", default="", namespaces=namespace))
            paper_id = re.sub(r"v\d+$", "", raw_id.rsplit("/", 1)[-1])
            if not paper_id:
                continue
            published = _compact(entry.findtext("atom:published", default="", namespaces=namespace))
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": published[:10],
                "title": _compact(entry.findtext("atom:title", default="", namespaces=namespace)),
                "summary": _compact(entry.findtext("atom:summary", default="", namespaces=namespace)),
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # tools report failures to the agent instead of crashing it
        return _error(exc)


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    try:
        params = {"limit": max(1, min(int(limit), 100))}
        if date:
            params["date"] = date
        response = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params))
        records = [record for item in response.json() if (record := _hf_record(item))]
        if keyword:
            needle = keyword.casefold()
            records = [record for record in records
                       if needle in f"{record['title']} {record['summary']}".casefold()]
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    if not str(query or "").strip():
        return "NO RESULTS"
    try:
        response = with_retry(lambda: _request("GET", HF_SEARCH_URL, params={
            "q": query,
            "limit": max(1, min(int(limit), 50)),
        }))
        records = [record for item in response.json() if (record := _hf_record(item, prefer_ai=True))]
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


def _sse_json(response):
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    payloads = []
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        payloads.append(json.loads(data))
    if not payloads:
        raise ValueError("Exa returned no JSON data event")
    return next((payload for payload in reversed(payloads)
                 if "result" in payload or "error" in payload), payloads[-1])


def _exa_rate_limited(result, text):
    meta = result.get("_meta") if isinstance(result, dict) else None
    meta_text = json.dumps(meta, ensure_ascii=False).casefold() if meta else ""
    combined = f"{meta_text}\n{text.casefold()}"
    strong_messages = ("rate limit exceeded", "rate-limit exceeded", "too many requests",
                       "rate limited", "rate-limited", "free mcp server rate limit")
    if any(message in combined for message in strong_messages):
        return True
    if isinstance(meta, dict):
        for key, value in meta.items():
            normalized = str(key).replace("_", "").replace("-", "").casefold()
            if "rate" in normalized and "limit" in normalized and value in (True, "true", 1, "1"):
                return True
    return False


def _exa_call(name, arguments):
    secret = (os.getenv("EXA_API_KEY") or "").strip()

    def call():
        response = _request(
            "POST",
            EXA_URL,
            params={"exaApiKey": secret} if secret else None,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": name, "arguments": arguments}},
        )
        payload = _sse_json(response)
        if payload.get("error"):
            raise RuntimeError(json.dumps(payload["error"], ensure_ascii=False))
        result = payload.get("result") or {}
        texts = [part.get("text", "") for part in result.get("content", [])
                 if isinstance(part, dict) and part.get("type") == "text"]
        text = "\n".join(part for part in texts if part).strip()
        if _exa_rate_limited(result, text):
            raise RetryableError("Exa rate limit exceeded")
        return text

    try:
        text = with_retry(call, attempts=7, base=2.0, cap=60.0)
        return text or "NO RESULTS"
    except Exception as exc:
        return _error(exc, secret)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    if not str(query or "").strip():
        return "NO RESULTS"
    try:
        return _exa_call("web_search_exa", {
            "query": query,
            "objective": objective or f"Find authoritative, relevant sources about {query}",
            "numResults": max(1, min(int(num_results), 10)),
        })
    except Exception as exc:
        return _error(exc, (os.getenv("EXA_API_KEY") or "").strip())


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    if not re.match(r"^https?://", str(url or "")):
        return "ERROR: ValueError: url must start with http:// or https://"
    result = _exa_call("web_fetch_exa", {"urls": [url]})
    return result if result.startswith("ERROR:") or result == "NO RESULTS" else result[:12000]


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
