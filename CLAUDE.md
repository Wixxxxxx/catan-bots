# catan-bots

Reinforcement-learning bots for the game of Catan: a Gymnasium environment, RL
training pipelines, gameplay analytics, and dashboards.

This file is the source of truth for how code is written in this repo. Every
agent and contributor must follow it. Specialized agents live in
[.claude/agents/](.claude/agents/) and reusable workflows in
[.claude/skills/](.claude/skills/).

## Python Coding Standards

These standards are mandatory. They are intentionally short so they are actually
followed.

### 1. Tooling — always use `uv`

- Run code with `uv run`, never bare `python`.
  - `uv run python main.py`
  - `uv run pytest`
- Manage dependencies with `uv` only — never edit `pyproject.toml` deps by hand
  and never use `pip`.
  - Add: `uv add <package>` (use `uv add --dev <package>` for dev/test tools).
  - Remove: `uv remove <package>`.
  - Sync the environment: `uv sync`.
- The lockfile (`uv.lock`) is committed; let `uv` regenerate it.

### 2. Object-oriented by default

- Model the domain with classes. Group related state and the behavior that acts
  on it into a single class rather than passing loose dicts/tuples around.
- Prefer composition over inheritance; keep inheritance hierarchies shallow.
- Use `@dataclass` for plain data holders and `abc.ABC` for interfaces that have
  multiple implementations (e.g. an `Agent` base class with concrete bots).
- Keep module-level free functions only for genuinely stateless helpers.

### 3. Atomic, modular functions

- One function, one responsibility. A function should do a single thing that its
  name describes; if you need "and" to describe it, split it.
- Keep functions short and flat. Extract nested logic into named helpers and
  prefer early returns over deep nesting.
- No hidden side effects: a function either computes and returns a value or
  performs a clearly named action — not both silently.

### 4. Documentation — docstrings everywhere

- Every module, class, and function has a docstring.
- Use **Google-style** docstrings stating **purpose**, **inputs** (`Args`),
  and **outputs** (`Returns`/`Yields`), plus `Raises` when relevant.
- **When you edit code, update its docstring in the same change.** A docstring
  that no longer matches its signature or behavior is a bug.

```python
def compute_longest_road(board: Board, player: Player) -> int:
    """Compute the length of a player's longest continuous road.

    Args:
        board: Current board state holding all placed roads.
        player: The player whose longest road is being measured.

    Returns:
        The number of edges in the player's longest unbroken road segment.
    """
```

### 5. Supporting conventions

- **Type hints** on all public function signatures and class attributes.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes,
  `UPPER_SNAKE_CASE` for constants.
- **Formatting/linting**: `uv run ruff format .` and `uv run ruff check .`
  before considering work done.
- **Tests** live in `tests/` and run with `uv run pytest`.
- **Imports** at module top, ordered stdlib → third-party → local.

### 6. Notebooks are for research only

- All production code lives in `src/catan_bots/` as importable modules with
  docstrings, type hints, and tests. Notebooks import it and call it.
- Never define a bot, environment, metric, or any other reusable logic in a
  notebook and leave it there. Prototype freely, then promote the code into a
  module and cover it with a test in `tests/`.
- A notebook cell should read as a few calls into the package plus the printing
  or plotting of results.
- The package is installed into the environment by `uv sync`, so
  `from catan_bots import ...` works from any notebook.

## Project Layout

```text
catan-bots/
├── CLAUDE.md            # this file — coding standards
├── pyproject.toml       # project + dependencies (managed by uv)
├── main.py              # entry point
├── src/catan_bots/      # package code
│   ├── games.py         # game construction and seeding
│   ├── bots/            # playable bots
│   └── analytics/       # reports, board stats, tournaments
├── notebooks/           # research notebooks (no production code)
├── tests/               # pytest suites
└── .claude/
    ├── agents/          # specialized subagents
    └── skills/          # reusable workflows
```

## Markdown Conventions (for all `.md` files in this repo)

To keep docs consistent, every Markdown file follows this format:

- Start with a single `# Title` (H1) and a one-line description beneath it.
- Use sentence-case headings and `##`/`###` for structure (no skipped levels).
- Wrap prose at ~80 characters.
- Use fenced code blocks with a language tag.
- Use relative Markdown links for file references, e.g. `[main.py](main.py)`.
- Agent and skill files additionally carry YAML frontmatter (see those files).
