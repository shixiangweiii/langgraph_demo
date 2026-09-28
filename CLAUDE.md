# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal learning sandbox for LangGraph 1.2 (with LangChain 1.4), focused on Human-in-the-Loop (HITL) workflows. Every script is standalone — there is no shared package, no library code, and no imports between files. Comments and printed output are in Chinese; keep that convention when editing.

- `hello_world.py` — minimal `StateGraph` (START → node → END), no LLM.
- `interrupt/simple_hitl_demo.py` — HITL mechanics with no LLM dependency. The heavy comment block in `human_check()` is deliberate: it records design notes on Redis-vs-memory checkpoint persistence and why LangGraph state (serializable) differs from IM long-connection affinity. Preserve it.
- `interrupt/HITL_GUIDE.md` — the conceptual write-up of the `interrupt()` / `Command` / checkpointer trio. Keep its snippets in sync with the demos.
- `multi_agent/director_human_in_loop_claude.py` — full LLM-backed workflow with two HITL checkpoints. The `_claude` suffix names the AI assistant that wrote the file (sibling versions by doubao/trae/tongyi were deleted in `1b24c70`); the runtime LLM is Qwen, not Claude.

## Environment and running

Dependencies live in the gitignored `.venv2` (Python 3.12); `requirements.txt` pins the direct dependencies. Installing straight from PyPI fails certificate verification on this machine (the python.org Python has no default CA file, and pip falls back to it when going through the `HTTPS_PROXY`), so the user installs from the Tsinghua mirror:

```bash
.venv2/bin/python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

There is no test suite, linter, or build step. The HITL demos block on `input()`, so drive them by piping stdin; add `-W error::DeprecationWarning` to catch API drift after an upgrade (all three scripts currently run clean with it):

```bash
.venv2/bin/python hello_world.py
printf 'approve\n' | .venv2/bin/python interrupt/simple_hitl_demo.py    # no API key needed
# Makes billed DashScope calls. The happy path reads 3 lines: request, approve, confirm.
set -a; source .env; set +a; printf '帮我规划北京3日游\napprove\nconfirm\n' | .venv2/bin/python multi_agent/director_human_in_loop_claude.py
```

The LLM is Qwen (`qwen3.7-max`) through `ChatQwen` from `langchain-qwq`, which talks to DashScope's OpenAI-compatible endpoint. `ChatQwen` defaults to the international endpoint; the demo pins `api_base` to the mainland one (`https://dashscope.aliyuncs.com/compatible-mode/v1`) because that is where the key is registered. The key is read from `LLM_SK` and passed explicitly, so `ChatQwen`'s own `DASHSCOPE_API_KEY` variable is ignored. `.env` holds only `LLM_SK`, and no script calls `load_dotenv()`: the PyCharm run config `director_human_in_loop_claude` (in the gitignored `.idea/workspace.xml`) injects `.env` via `ENV_FILES` and runs from `multi_agent/`; from a shell, load `.env` in the same command, as above. Without `LLM_SK` the demo prints a warning and exits 1 before building the model.

Don't reintroduce `langchain_community`'s `ChatTongyi`: `langchain-community` has been sunset (it emits a `DeprecationWarning` on import) and LangChain's docs now point to `langchain-qwq` for Qwen. `langchain-community` and `dashscope` are still installed in `.venv2`, but nothing uses them.

## The HITL pattern used throughout

Understanding this pattern is the point of the repo; both demos are variations on it.

1. **`interrupt(payload)` inside a node pauses the whole graph.** Graph state and the pending interrupt are saved to the checkpointer (`InMemorySaver()`, mandatory for HITL); the current `graph.stream()` yields a last update whose data is `{"__interrupt__": (Interrupt(value=payload, ...),)}` and then ends.
2. **Resuming requires a brand-new `graph.stream(Command(resume=value), config)` call**, whose `config` (`{"configurable": {"thread_id": ...}}`) ties it back to the paused state. **The interrupted node then re-runs from its first line**, and this time `interrupt()` returns `value` instead of pausing — no call stack or local variables are saved. Code before `interrupt()` therefore runs twice (the pre-interrupt `print`s in `simple_hitl_demo.py` visibly repeat), so never put an LLM call or other side effect ahead of `interrupt()`.
3. **HITL nodes route dynamically via `Command`, not edges.** They are typed `-> Command[Literal["node_a", "node_b"]]` and return `Command(goto=..., update=...)`, so `create_graph()` intentionally has no outgoing `add_edge` for them. A static edge would fire *in addition to* `goto`, running both branches. The `Literal` annotation isn't needed for routing, since `goto` works without it. It is what lets `get_graph()` draw the edges; without it the node is drawn as going straight to `END`.
4. **Drivers use the v2 stream format and loop on pending interrupts.** Each HITL demo's `stream_until_pause()` calls `graph.stream(..., stream_mode="updates", version="v2")`, whose parts are `{"type", "ns", "data"}` dicts, and returns whatever arrived under `data["__interrupt__"]`; the caller loops `while pending`, because a reject or redo pauses again. `stream()`/`invoke()` still default to `version="v1"`; `invoke(..., version="v2")` returns a `GraphOutput` (`.value`, `.interrupts`), and dict-style access to it is deprecated.
5. **Resume values in the multi-agent demo are typed with `interrupt(..., response_schema=...)`**, which only exists since LangGraph 1.2.12 — anything older fails on that keyword. The `PlanDecision`/`ResultDecision` Pydantic models validate the resume value and `interrupt()` returns the model instance; clients see the JSON Schema on `Interrupt.response_schema` (the console prints its allowed actions). A `mode="before"` validator turns a bare string into an action by substring (`approve`/`reject`/`confirm`; anything else becomes revise/redo with the text as feedback), so console input never fails validation. An invalid dict raises `pydantic.ValidationError` out of `graph.stream()`, and the thread stays paused at the same interrupt, so the client can simply resume again. The simple demo deliberately keeps an untyped string resume.
6. **The LLM is injected through runtime context, not a module global.** `StateGraph(State, context_schema=Context)`; nodes take `runtime: Runtime[Context]` and call `runtime.context.llm`. Context is not checkpointed, so every `stream()` call — including resumes — must pass `context=...`. This also makes it easy to drive the graph with a fake chat model.

`director_human_in_loop_claude.py` flow: `classify_task → generate_plan → human_review_plan ⏸ → execute_task → human_review_result ⏸ → finalize`. Rejection at either checkpoint loops back (`generate_plan` / `execute_task` respectively) rather than aborting. Human comments go into the `feedback` state field and are appended to the next `generate_plan`/`execute_task` prompt; approving a plan clears it.

## Gotchas

- Logging in the multi-agent demo is configured in `main()` at `level=logging.ERROR`, so every `logging.info` trace in the node functions is silently dropped. Lower the level when debugging graph flow. `hitl_demo.log` is written next to the script.
- `.log` files, `.env`, `.venv2`, and `.idea` are gitignored. Existing commit messages follow `<中文描述> to #<issue>` (e.g. `init to #000000`).
