# Git Commit and PR Formatting Guide

Standards for commits and pull requests on the ARQ Pairs Trading project. Following these conventions keeps the git history readable, makes code review faster, and helps Claude Code understand what changed and why.

---

## Commit Messages

### Format

```
<type>(<scope>): <short summary>

<body — optional>

<footer — optional>
```

### Rules

- **Summary line**: 50 characters or fewer, imperative mood ("add", not "added" or "adds"), no period at the end
- **Body**: Wrap at 72 characters. Explain *what* changed and *why*, not *how* (the code shows how). Leave one blank line between summary and body.
- **Footer**: Reference issues or note breaking changes. Leave one blank line between body and footer.

---

### Types

| Type | When to Use |
|---|---|
| `feat` | A new feature or capability added to the pipeline |
| `fix` | A bug fix |
| `refactor` | Code restructured without changing behavior |
| `test` | Adding or updating tests |
| `docs` | Changes to markdown files, docstrings, or comments only |
| `config` | Changes to `pyproject.toml`, `.env.example`, `config.py`, or CI |
| `data` | Changes to data fetch, clean, or load logic |
| `perf` | Performance improvement (e.g., caching, vectorization) |
| `chore` | Dependency updates, gitignore changes, housekeeping |

---

### Scopes

Use the module directory name as the scope:

`data` · `universe` · `clustering` · `scoring` · `tiering` · `signals` · `regime` · `backtest` · `metrics` · `scripts` · `tests` · `docs` · `config`

---

### Examples

**Simple fix — no body needed:**
```
fix(signals): correct percentile calculation for fat-tail spreads
```

**New feature with explanation:**
```
feat(scoring): add Benjamini-Hochberg FDR correction to Johansen tests

Without multiple-testing correction, roughly 40-60 pairs were passing
Johansen by chance alone across 500 tests at p < 0.10. BH correction
reduces false discovery rate while keeping a practical number of pairs
alive for scoring.
```

**Config change with breaking note:**
```
config: raise default signal_window from 60 to 90 days

BREAKING: existing cached correlation matrices in data/processed/ must
be deleted and rebuilt. Run scripts/02_build_universe.py after pulling.
```

**Docs only:**
```
docs: update strategy.md with Bollinger regime filter clarification
```

**Refactor:**
```
refactor(backtest): extract position sizing into portfolio.py

engine.py was handling sizing logic directly. Moved to portfolio.py
so it can be unit tested independently and reused by walkforward script.
```

---

## Branching

### Branch Naming

```
<type>/<short-description>
```

Use hyphens, all lowercase, no spaces:

```
feat/composite-scorer
fix/halflife-negative-slope
refactor/backtest-engine
docs/update-architecture
test/scoring-edge-cases
data/add-vix-filter
```

### Branch Strategy

```
main
└── develop
    ├── feat/composite-scorer       ← feature branches off develop
    ├── fix/halflife-negative-slope
    └── data/add-vix-filter
```

- `main` — stable, backtest-ready code only. Never commit directly.
- `develop` — integration branch. Merge feature branches here first.
- Feature branches — one branch per module or logical unit of work. Branch off `develop`, merge back into `develop` via PR.

---

## Pull Requests

### PR Title

Follow the same format as a commit message summary:

```
<type>(<scope>): <short summary>
```

Examples:
```
feat(clustering): add silhouette-scored K-means with ARI stability check
fix(regime): prevent stop tightening on positions younger than 1 day
docs: add implementation plan with module ownership table
```

---

### PR Description Template

Copy this template when opening a PR:

```markdown
## What This PR Does
<!-- 1-3 sentences. What changed and why. -->


## Changes
<!-- List the files changed and what was done to each. -->
- `src/scoring/cointegration.py` — added BH FDR correction across cluster pairs
- `src/scoring/composite.py` — updated to use adjusted p-values from cointegration
- `tests/test_scoring.py` — added tests for FDR correction behavior


## How to Test
<!-- Steps a reviewer can follow to verify the change works. -->
1. Pull branch and run `pytest tests/test_scoring.py`
2. Run `python scripts/03_run_backtest.py` and confirm no regressions in Sharpe


## Notes for Reviewer
<!-- Anything that needs extra attention, open questions, or known tradeoffs. -->
- Used alpha=0.10 for BH correction to match our Johansen threshold
- Did not apply global correction across clusters — see open-questions.md


## Related
<!-- Link to issues, tickets, or related PRs if applicable. -->
- Closes #12
- Related to #8 (multiple testing discussion)
```

---

### PR Size Guidelines

| Size | Lines Changed | Guideline |
|---|---|---|
| Small | < 100 | Ideal. Fast to review, easy to reason about. |
| Medium | 100–300 | Acceptable for new modules. Add extra context in description. |
| Large | 300–500 | Split if possible. If not, schedule a sync before review. |
| Too large | 500+ | Always split. No exceptions. |

If a PR is getting large, split it: open one PR for tests and interfaces (no implementation), and a second for the implementation. The first PR defines the contract; the second fulfills it.

---

### Review Checklist

Before marking a PR ready for review, confirm:

**Code**
- [ ] All functions have docstrings with inputs, outputs, and any side effects noted
- [ ] No hardcoded parameter values — everything comes from `src/config.py`
- [ ] No print statements — use Python `logging` module
- [ ] No data files accidentally staged (check `git status` carefully)

**Tests**
- [ ] New code has corresponding unit tests in `tests/`
- [ ] All existing tests pass (`pytest` runs clean)
- [ ] Tests use fixtures from `tests/fixtures/`, not real market data

**Docs**
- [ ] If a design decision was made, it is logged in `docs/decisions.md`
- [ ] If a question was answered, it is removed from `docs/open-questions.md`
- [ ] If behavior changed, `docs/strategy.md` or `docs/architecture.md` is updated

**Data**
- [ ] No parquet files or output files committed
- [ ] If config parameters changed, teammates are notified to rebuild processed data

---

### Review Turnaround

- Reviewers aim to respond within **24 hours** of a PR being marked ready
- Authors address all comments within **24 hours** of review
- A PR needs **1 approval** before merging into `develop`
- Merges into `main` need **2 approvals**

---

## Merge Strategy

- **Squash and merge** for feature branches into `develop` — keeps history clean, one commit per feature
- **Merge commit** for `develop` into `main` — preserves the integration history

After merging, delete the feature branch.

---

## Quick Reference

```bash
# Start a new feature
git checkout develop
git pull origin develop
git checkout -b feat/my-feature

# Stage and commit
git add src/scoring/halflife.py tests/test_scoring.py
git commit -m "feat(scoring): add OU process half-life estimation"

# Push and open PR
git push origin feat/my-feature
# Open PR on GitHub: feat/my-feature → develop

# After PR merges, clean up
git checkout develop
git pull origin develop
git branch -d feat/my-feature
```