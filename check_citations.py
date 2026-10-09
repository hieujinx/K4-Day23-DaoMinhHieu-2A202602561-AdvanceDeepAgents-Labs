"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    for required_heading in ("## TL;DR", "## Background", "## Trends and open problems"):
        if not re.search(rf"(?m)^{re.escape(required_heading)}[ \t]*$", report_text):
            problems.append(f"missing required heading: {required_heading}")

    by_number = {}
    seen_urls = set()
    families = set()
    allowed_families = {"arxiv", "hf-daily", "hf-search", "web"}
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            problems.append(f"source entry {index + 1} is not an object")
            continue
        number = source.get("n")
        source_id = source.get("id")
        url = source.get("url")
        family = source.get("source")
        date = source.get("date")
        if family not in allowed_families:
            problems.append(f"source [{number}] has invalid source family={family!r}")
        else:
            families.add(family)
        if not isinstance(number, int) or isinstance(number, bool):
            problems.append(f"source entry {index + 1} has invalid n={number!r}")
        elif number in by_number:
            problems.append(f"source number [{number}] appears more than once")
        else:
            by_number[number] = source
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            problems.append(f"source [{number}] has invalid url={url!r}")
        elif url in seen_urls:
            problems.append(f"source url appears more than once: {url}")
        else:
            seen_urls.add(url)
        if date != "n.d." and not (isinstance(date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)):
            problems.append(f"source [{number}] has invalid date={date!r}; use YYYY-MM-DD or n.d.")
        if family == "arxiv" and isinstance(url, str):
            match = re.fullmatch(r"https://arxiv\.org/abs/(\d{4}\.\d{4,5})", url)
            if not match:
                problems.append(f"source [{number}] labeled arxiv must use a canonical arXiv abs URL")
            else:
                canonical_id = match.group(1)
                if source_id != canonical_id:
                    problems.append(f"source [{number}] id does not match its arXiv URL")
        if family in {"hf-daily", "hf-search"} and isinstance(url, str):
            match = re.fullmatch(r"https://huggingface\.co/papers/(\d{4}\.\d{4,5})", url)
            if not match:
                problems.append(f"source [{number}] labeled {family} must use a canonical Hugging Face paper URL")
            elif source_id != match.group(1):
                problems.append(f"source [{number}] id does not match its Hugging Face URL")

    if len(families) < 3:
        problems.append(f"only {len(families)} source families remain; need at least 3")

    heading = re.search(r"(?m)^##[ \t]+References[ \t]*$", report_text)
    if heading is None:
        problems.append("missing ## References heading")
        body, references = report_text, ""
    else:
        body, references = report_text[:heading.start()], report_text[heading.end():]

    # Citations in fenced/inline code are examples, not evidence. Markdown links
    # such as [1](url) are also deliberately excluded.
    without_code = re.sub(r"```.*?```|`[^`\n]*`", "", body, flags=re.DOTALL)
    groups = re.findall(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\](?!\()", without_code)

    def expand(group):
        result = []
        for part in re.split(r"\s*,\s*", group):
            span = re.fullmatch(r"(\d+)\s*[\-–]\s*(\d+)", part)
            if span:
                start, end = int(span.group(1)), int(span.group(2))
                result.extend(range(start, end + 1) if 0 <= end - start <= 200 else (start, end))
            else:
                result.append(int(part))
        return result

    cited = {number for group in groups for number in expand(group)}
    source_numbers = set(by_number)
    for number in sorted(cited - source_numbers):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(source_numbers - cited):
        problems.append(f"source [{number}] never cited")

    reference_lines = {}
    for line in references.splitlines():
        match = re.match(r"^\[(\d+)\](?:\s|$)", line.strip())
        if match:
            reference_lines.setdefault(int(match.group(1)), []).append(line.strip())

    for number in sorted(source_numbers):
        lines = reference_lines.get(number, [])
        if not lines:
            problems.append(f"source [{number}] has no reference line")
            continue
        if len(lines) > 1:
            problems.append(f"source [{number}] has more than one reference line")
        for line in lines:
            urls = re.findall(r"https?://\S+", line)
            expected = by_number[number].get("url")
            if len(urls) != 1:
                problems.append(f"reference [{number}] must contain exactly one URL")
            elif urls[0] != expected:
                problems.append(f"reference [{number}] URL does not match sources.json")
    for number in sorted(set(reference_lines) - source_numbers):
        problems.append(f"reference [{number}] is not present in sources.json")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
