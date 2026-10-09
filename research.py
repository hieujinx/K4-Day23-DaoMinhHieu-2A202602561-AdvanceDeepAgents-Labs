"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401
from tools import verify_source_metadata

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[\W_]+", "-", str(topic or "").strip().lower(), flags=re.UNICODE).strip("-")
    return slug[:60].rstrip("-") or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        "Produce a rigorous deep-research survey for the literal topic below. Follow your full delegated workflow, "
        "write all artifacts to the contracted sandbox paths, and finish only after finalization and validation.\n\n"
        f"TOPIC: {topic.strip()}"
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    calls = Counter()
    input_tokens = 0
    output_tokens = 0
    for message in messages or []:
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls") or []
            usage = message.get("usage_metadata") or {}
        else:
            tool_calls = getattr(message, "tool_calls", None) or []
            usage = getattr(message, "usage_metadata", None) or {}
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                calls[name] += 1
        input_tokens += int(usage.get("input_tokens", usage.get("input", 0)) or 0)
        output_tokens += int(usage.get("output_tokens", usage.get("output", 0)) or 0)
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": calls.get("task", 0),
        "tool_calls": dict(sorted(calls.items())),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("sandbox did not produce a non-empty report.md")
    if not sources_bytes:
        raise RuntimeError("sandbox did not produce sources.json")
    try:
        report_bytes.decode("utf-8")
        sources = json.loads(sources_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"sandbox output is not valid UTF-8/JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources or not all(isinstance(source, dict) for source in sources):
        raise RuntimeError("sources.json must be a non-empty JSON list of objects")

    slug = slugify(topic)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"
    metadata = {
        "topic": topic,
        **summarize(messages, elapsed, model_name),
        "n_sources": len(sources),
        "source_families": sorted({str(source.get("source")) for source in sources if source.get("source")}),
    }
    outputs = {
        report_path: report_bytes,
        sources_path: sources_bytes,
        meta_path: json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
    }
    temporary = {path: path.with_name(path.name + ".tmp") for path in outputs}
    try:
        for path, content in outputs.items():
            temporary[path].write_bytes(content)
        for path in outputs:
            temporary[path].replace(path)
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink()
    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    topic = str(topic or "").strip()
    if not topic:
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        model = make_model()
        model_name = (getattr(model, "model_name", None) or getattr(model, "model", None)
                      or type(model).__name__)
        started = time.monotonic()
        with open_sandbox() as backend:
            created = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if created.exit_code != 0:
                raise RuntimeError(f"cannot create sandbox workspace: {created.output}")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000},
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            for repair_round in range(3):
                finalized = backend.execute(f"python3 {FINALIZER_PATH}")
                if finalized.exit_code == 0:
                    validated = backend.execute(f"python3 {VALIDATOR_PATH}")
                else:
                    validated = None
                remote_problems = []
                if finalized.exit_code == 0 and validated.exit_code == 0:
                    snapshot = download(backend, [SOURCES_PATH]).get(SOURCES_PATH)
                    try:
                        source_snapshot = json.loads(snapshot.decode("utf-8")) if snapshot else []
                    except (UnicodeDecodeError, ValueError) as exc:
                        remote_problems = [f"cannot parse sources for remote verification: {exc}"]
                    else:
                        before_normalization = json.dumps(source_snapshot, ensure_ascii=False, sort_keys=True)
                        remote_problems = verify_source_metadata(source_snapshot, topic=topic, normalize=True)
                        after_normalization = json.dumps(source_snapshot, ensure_ascii=False, sort_keys=True)
                        if before_normalization != after_normalization:
                            upload(backend, {SOURCES_PATH: json.dumps(
                                source_snapshot, ensure_ascii=False, indent=2).encode("utf-8")})
                            finalized = backend.execute(f"python3 {FINALIZER_PATH}")
                            validated = (backend.execute(f"python3 {VALIDATOR_PATH}")
                                         if finalized.exit_code == 0 else None)
                if finalized.exit_code == 0 and validated.exit_code == 0 and not remote_problems:
                    break
                finalizer_output = finalized.output.strip() or "(no output)"
                validator_output = ((validated.output.strip() or "(no output)")
                                    if validated is not None else "not run because finalization failed")
                if remote_problems:
                    validator_output += "\nRemote metadata problems:\n" + "\n".join(remote_problems)
                if repair_round == 2:
                    raise RuntimeError(
                        "sandbox postflight did not pass after 3 repair rounds; "
                        f"finalizer: {finalizer_output}; validator: {validator_output}"
                    )
                repair_prompt = f"""Repair the existing deep-research artifacts for the literal topic: {topic}

The notes, report, and sources from the completed research run are already present at the contracted sandbox paths.
Do not restart the survey. Read those files and make the smallest evidence-grounded corrections needed.

The mandatory sandbox postflight failed.

Finalizer output:
{finalizer_output}

Validator output:
{validator_output}

Inspect the existing notes, report, and sources in the contracted paths. Fix every reported problem without inventing
evidence. If a third source family is missing and the existing notes cannot supply it, delegate one focused researcher
for that missing family. Then rerun the finalizer and validator yourself. Do not finish until the validator prints OK."""
                repaired = agent.invoke(
                    {"messages": [{"role": "user", "content": repair_prompt}]},
                    config={"recursion_limit": 1000},
                )
                if isinstance(repaired, dict):
                    messages = [*messages, *repaired.get("messages", [])]
            elapsed = time.monotonic() - started
            report_path = save_outputs(backend, topic, messages, elapsed, model_name)
    except Exception as exc:  # cleanup is guaranteed by open_sandbox before control reaches here
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Saved report to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
