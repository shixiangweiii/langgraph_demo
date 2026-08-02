# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal learning sandbox for LangGraph 1.0.7, focused on Human-in-the-Loop (HITL) workflows. Each subdirectory is a standalone, self-contained demo script — there is no shared package, no library code, and no import relationships between directories. Comments and printed output are in Chinese; keep that convention when editing.

- `hello_world.py` — minimal `StateGraph` (state → node → END), no LLM.
- `interrupt/simple_hitl_demo.py` — HITL mechanics with no LLM dependency. The heavy comment block in `human_check()` is deliberate: it records design notes on Redis-vs-memory checkpoint persistence and why LangGraph state (serializable) differs from IM long-connection affinity. Preserve it.
- `interrupt/HITL_GUIDE.md` — the conceptual write-up of the `interrupt()` / `Command` / checkpointer trio.
- `multi_agent/director_human_in_loop_claude.py` — full LLM-backed workflow with two HITL checkpoints.

## Environment and running

Dependencies live only in the gitignored `.venv2` (Python 3.12); there is no `requirements.txt` or `pyproject.toml`, so recreating the env means installing by hand (`langgraph`, `langchain-community`, `dashscope`).

```bash
.venv2/bin/python hello_world.py
.venv2/bin/python interrupt/simple_hitl_demo.py          # no API key needed
export LLM_SK='<tongyi_api_key>'                         # required for the next one
.venv2/bin/python multi_agent/director_human_in_loop_claude.py
```

The LLM is Alibaba Tongyi (`ChatTongyi`, model `qwen3.7-max`) keyed off the `LLM_SK` env var. A `.env` file holds that key but **nothing calls `load_dotenv()`** — the scripts read `os.getenv("LLM_SK")` directly, so the variable must be exported in the shell or the demo exits with a warning.

Both HITL demos are interactive (`input()` at the console) and cannot be run non-interactively without stubbing stdin. There is no test suite, linter, or build step.

## The HITL pattern used throughout

Understanding this pattern is the point of the repo; both demos are variations on it.

1. **`interrupt(payload)` inside a node pauses the whole graph.** LangGraph serializes execution context into the checkpointer and the current `graph.stream()` iterator *terminates* — it does not yield further. Compiling with a checkpointer (`MemorySaver()`) is mandatory; without it, interrupt cannot resume.
2. **Resuming requires a brand-new `graph.stream(Command(resume=value), config)` call.** The same `config` (`{"configurable": {"thread_id": ...}}`) is what ties the new stream back to the paused state. Execution restarts inside the interrupted node and `interrupt()` returns `value`.
3. **HITL nodes route dynamically via `Command`, not edges.** They are typed `-> Command[Literal["node_a", "node_b"]]` and return `Command(goto=..., update=...)`. Consequently `add_edge` is intentionally *absent* for these nodes in `create_graph()` — the `Literal` type annotation is what registers the possible targets. Adding an outgoing edge for a `Command` node is a mistake.
4. **The driver loop polls state, it does not catch exceptions.** After a stream ends, `graph.get_state(config)`; a truthy `.next` means a pause is pending, and the payload passed to `interrupt()` is read at `state.tasks[0].interrupts[0].value`. Loop `while state_snapshot.next` to handle several consecutive checkpoints.
5. **Feedback parsing is defensive by design** — nodes accept either a dict (`{"action": ..., "feedback": ...}`) or a bare string and fall back to a safe default, because the resume value's shape is decided by whatever client drives the graph.

`director_human_in_loop_claude.py` flow: `classify_task → generate_plan → human_review_plan ⏸ → execute_task → human_review_result ⏸ → finalize`. Rejection at either checkpoint loops back (`generate_plan` / `execute_task` respectively) rather than aborting.

## Gotchas

- `HITL_GUIDE.md` refers to the full demo as `human_in_loop_demo.py`; that file no longer exists and the code now lives at `multi_agent/director_human_in_loop_claude.py`. Update the guide if you touch it.
- Logging in the multi-agent demo is configured at `level=logging.ERROR`, so every `logging.info` trace in the node functions is silently dropped. Lower the level when debugging graph flow.
- `.log` files, `.env`, `.venv2`, and `.idea` are gitignored. Existing commit messages follow `<中文描述> to #<issue>` (e.g. `init to #000000`).
