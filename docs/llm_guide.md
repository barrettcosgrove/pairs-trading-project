# ARQ Pairs Trading — Guide to Using AI Coding Assistants

> **Read order for a new session:** `CLAUDE.md` or `AGENTS.md` →
> `docs/architecture.md` → `src/config.py`. `docs/strategy.md` is the
> original v2.0 spec; several sections (percentiles, tiering, Bollinger)
> are not implemented. Live results: `docs/diagnostics.md`.

This guide applies to all AI coding tools your team is using: **Claude Code**, **Cursor**, and **OpenAI Codex**. The principles are the same regardless of tool. The goal is to get consistent, correct, project-aware code without having to re-explain the codebase every session.

---

## The Most Important Rule

**Every AI session must start by reading project files.**

AI assistants have no memory of previous sessions. Without context, they will guess at your architecture, invent parameter values, and produce code that looks right but breaks at integration. Giving context upfront takes 60 seconds and saves hours of debugging.

---

## Your Project Documentation Map

Before writing a single line of code, know which doc answers which question. Open the right file before starting an AI session — you will need to paste it or reference it.

### `CLAUDE.md` — The Master Context File

**Read this first. Every session. No exceptions.**

Contains: project overview, tech stack, file structure, module ownership, critical coding rules, data contracts between modules, key CONFIG parameters, common mistakes to avoid, and git conventions.

**When to use:** Start of every AI session. Paste it in full or tell the AI to read it first.

**Most important sections for your module:**

- Critical Rules (rules 1-7) — non-negotiable, will cause bugs if violated
- Data Contracts — exact function signatures your module must match
- Key Parameters — every CONFIG value you should reference
- Common Mistakes — bugs the AI is likely to make on this project

---

### `docs/strategy.md` — Original v2.0 spec

Design history. Do **not** treat it as the live engine. Implemented
behavior is `src/config.py` + `docs/architecture.md`. If this file and
the code disagree, trust the code.

**When to use:**

- To understand why an earlier design chose a rule
- Not as a checklist for new signal or scoring code

---

### `docs/architecture.md` — How the System is Designed

**The engineering reference.**

Contains: full pipeline diagram, data flow between modules, technology stack with rationale, data contracts, and pipeline execution order.

**When to use:**

- Before implementing any module — understand what feeds into it and what consumes its output
- When debugging integration failures
- When deciding where a new function should live

---

### `docs/data.md` — Everything About Data

**The data reference.**

Contains: what each parquet file contains, column schemas, universe filter thresholds, known data limitations, and troubleshooting steps.

**When to use:**

- Before writing any code that reads data
- When `load_returns()` or `load_prices()` returns unexpected results
- When debugging missing tickers or NaN values
- When a teammate asks why a ticker is missing from the universe

---

### `docs/module_checklist.md` — What Each File Must Do

**Your implementation guide.**

Contains: ordered list of every file to build, who owns it, and a checkbox checklist of exactly what each file must implement.

**When to use:**

- Before starting any file — read your section first
- As a review checklist after Claude writes a file
- To understand what the file above or below yours in the pipeline does

**This is the most important doc for day-to-day implementation work.**

---

### `docs/implementation_plan.md` — Who Does What When

**The team coordination document.**

Contains: phase-by-phase build plan, daily task assignments, milestone checkpoints, blockers log, and open questions.

**When to use:**

- Daily standup — update your task status
- When you finish a file — mark it complete
- When you hit a blocker — log it here
- When a design question comes up — check open questions first

---

### `docs/decisions.md` — Why Things Are the Way They Are

**The decision log.**

Contains: every significant design decision, why it was made, and what alternatives were rejected.

**When to use:**

- Before changing a threshold or architecture decision — check if it was already decided and why
- After making a non-obvious decision — add it here so teammates don't re-debate it
- When a teammate asks "why did we do it this way?"

---

### `src/config.py` — All Strategy Parameters

**The single source of truth for every number.**

Contains: every tunable parameter as a frozen dataclass field with a comment explaining what it does.

**When to use:**

- Every time you write code that uses a number — check if it is in CONFIG
- Never hardcode a number — always use `CONFIG.parameter_name`
- When you need to know the exact value of a threshold

---

## How to Give Good Context

Good context has four components. Include all four in every prompt.

### Component 1: Tell it what to read

Explicitly name the files the AI should read before writing anything. The AI cannot browse your filesystem — you must tell it which files contain the relevant context.

### Component 2: Describe the module's place in the pipeline

- What calls your module (the consumer)
- What your module calls (your dependencies)
- The exact input and output data contracts from `CLAUDE.md`

### Component 3: Give the exact function signature

Copy the signature from `docs/module_checklist.md` or `CLAUDE.md` verbatim. Do not paraphrase it. The signature is a contract between modules — it must match exactly.

### Component 4: List the rules

Always include:

- Use CONFIG for all thresholds
- Use logging not print
- Use load.py for data access — never pd.read_parquet() directly
- Every function needs a docstring with Args and Returns

---

## Prompt Template

Copy this template and fill in the bracketed sections for every file:

```
Before writing any code, read these files in order:
1. CLAUDE.md
2. [docs/strategy.md section X — if your module involves trading logic]
3. [docs/architecture.md — if your module connects to other modules]
4. src/config.py
5. [any other file your module imports from]

After reading, confirm back to me:
- What function(s) will you implement and their exact signatures?
- What CONFIG parameters will you use?
- What does your module receive as input and what does it return?
- What files does it read from or write to?

Wait for my confirmation before writing any code.

---

Write [filename].

This file is part of the [clustering / scoring / tiering / signals /
regime / backtest / metrics] layer of the ARQ pairs trading pipeline.
It is called by [consumer module]. It calls [dependency modules].

It must expose [N] public function(s):

[paste exact signature from module_checklist.md]

Implementation requirements (from docs/module_checklist.md):
[paste the checklist items for this file]

Non-negotiable rules:
- All threshold values come from CONFIG in src/config.py — no hardcoded numbers
- All data access goes through src/data/load.py — never call pd.read_parquet() directly
- Use Python logging not print statements
- Every function (public and private) needs a docstring with Args and Returns
- No imports of yfinance or curl_cffi — those are isolated to src/data/fetch.py
- No look-ahead bias — only use data available on or before the as_of date

```

---

## Worked Example: Good vs Bad Prompt

**Scenario:** Althan is writing `src/clustering/kmeans.py`

---

**Bad prompt — what not to do:**

> "Write the kmeans clustering file for our pairs trading project"

This gives the AI no context about your architecture, no function signatures, no parameter values, and no rules. It will invent everything and produce code that looks plausible but will break at integration.

---

**Good prompt — use this as a model:**

> "Before writing any code, read these files in order:
>
> 1. `CLAUDE.md`
> 2. `src/config.py`
> 3. `src/clustering/correlation.py` (this is what feeds into my module)
>
> After reading, confirm back to me:
>
> - What function will you implement and its exact signature?
> - Which CONFIG parameters control k range and restarts?
> - What does the distance matrix look like as input?
>
> Wait for my confirmation before writing any code.
>
> ---
>
> Write `src/clustering/kmeans.py`.
>
> This file is part of the clustering layer. It is called by `src/scoring/composite.py`. It receives a distance matrix from `src/clustering/correlation.py`.
>
> It must expose one public function: `run_clustering(distance_matrix: pd.DataFrame) -> dict[int, list[str]]`
>
> From docs/module_checklist.md, it must:
>
> - Scan k from CONFIG.k_min (4) to CONFIG.k_max (20)
> - Run CONFIG.kmeans_restarts (10) random restarts per k value
> - Select k with highest average silhouette score
> - Use CONFIG.random_seed (42) for reproducibility
> - Return dict mapping cluster_id (int) to list of ticker strings
> - Log the winning k and silhouette score at INFO level
>
> Non-negotiable rules:
>
> - All values from CONFIG — no hardcoded numbers
> - Use logging not print
> - Every function needs a docstring with Args and Returns
> - The output format must exactly match the contract in CLAUDE.md: dict[int, list[str]] where keys are cluster ids and values are ticker lists"

---

## After the AI Writes the File

Before accepting any file, run through this checklist:

**Code review:**

- [ ] No hardcoded numbers (search for any raw integers or floats in logic)
- [ ] No `pd.read_parquet()` calls (only allowed in `scripts/`)
- [ ] No `import yfinance` or `import curl_cffi` (only in `fetch.py`)
- [ ] No `print()` statements — only `logging.getLogger(__name__)`
- [ ] Every function has a docstring
- [ ] Function signatures match `CLAUDE.md` data contracts exactly
- [ ] No look-ahead bias in rolling windows

**Run the import check:**

```bash
uv run python -c "from src.[module] import [function]; print('import OK')"

```

**Run any relevant tests:**

```bash
uv run pytest tests/test_[module].py -v

```

If any check fails, tell the AI exactly what is wrong and ask it to fix that specific thing. Do not start a new session — fix in the same session.

---

## Updating Documentation After You Build

Every time you complete a file, update two documents:

`docs/implementation_plan.md`**:** Mark your task as `[x]` complete. If you hit any unexpected issues, add them to the Blockers Log. If you made a design decision that differs from the plan, note it.

`docs/decisions.md`**:** If you made any non-obvious design choice — choosing one algorithm over another, using a different threshold, handling an edge case in a specific way — write a short entry explaining what you decided and why. This prevents teammates from re-questioning settled decisions.

`CLAUDE.md` **— add corrections:** If the AI made a mistake that you had to correct, add it to the "Common Mistakes to Avoid" section or a "Corrections" section at the bottom. This prevents the same mistake in future sessions.

**Rule of thumb:** If you spent more than 15 minutes debugging something the AI got wrong, document it. The next person will thank you.

---

## Tool-Specific Notes

### Claude Code

- Start fresh session per file — do not carry over context between files
- Use `/init` in a new project to generate a base CLAUDE.md (already done)
- Claude Code can read your actual filesystem — tell it to `read CLAUDE.md` directly rather than pasting it
- Use Opus model for planning and debugging, Sonnet for implementation

### Cursor

- Paste the contents of `CLAUDE.md` at the start of every chat
- Cursor's codebase indexing helps but does not replace explicit context
- Use `@file` references to pull in specific files: `@CLAUDE.md @src/config.py` at the start of your prompt
- The prompt template above works directly in Cursor chat

### OpenAI Codex

- Codex has no access to your filesystem — paste file contents directly
- At minimum paste: relevant section of `CLAUDE.md`, the function signature from `module_checklist.md`, and the contents of `src/config.py`
- Codex is strongest on well-defined algorithmic tasks (e.g. half-life AR(1) regression, Johansen test wrapper) — give it the math explicitly
- For integration-heavy files (e.g. `engine.py`), Claude Code or Cursor with filesystem access will produce more reliable results

---

## When Things Go Wrong

**AI produces wrong function signature:** Stop immediately. Paste the exact signature from `CLAUDE.md` and say: "The function signature must be exactly this. Rewrite to match."

**AI uses a hardcoded number:** Say: "Replace [number] with CONFIG.[parameter_name] from src/config.py. Never hardcode values — all parameters come from CONFIG."

**AI reads parquet directly:** Say: "Remove that pd.read_parquet() call. All data access must go through src/data/load.py. Use load_returns() / load_prices() / load_vix() instead."

**AI produces code that conflicts with another module:** Check `docs/architecture.md` for the data flow and `CLAUDE.md` for the data contracts. Paste the relevant contract to the AI and say: "Your output must match this contract exactly because [consumer module] depends on it."

**You are unsure if a decision has already been made:** Check `docs/decisions.md` before asking the AI. If it is there, the decision is final — do not re-debate it. If it is not there, log the open question in `docs/implementation_plan.md` and ask at standup.