# agentic-inferno

A terminal spectacle where a Writer LLM agent and a Critic LLM agent run concurrently in an endless loop of revision and beratement.

You sit back and watch. The Writer frantically revises your document while the Critic shreds every change with theatrical contempt. When the insults get too harsh the Writer stops mid-thought, turns to you, the audience, and apologizes for its existence. No convergence. No "done." Just two agents locked in a cage match until you hit Esc or the money runs out.

## Quick start

```bash
# Build from source (only option for now)
cargo install --git https://github.com/jojomatik/agentic-inferno
```

Or clone and build locally:

```bash
git clone https://github.com/jojomatik/agentic-inferno
cd agentic-inferno
cargo build --release
./target/release/agentic-inferno --writer-model gpt-4o --input my-draft.md
```

## Prerequisites

- **Rust** (1.80+) for building from source
- **API keys** set in a `.env` file or exported as environment variables (see [`.env` setup](#env-setup))
- **claude CLI** (optional, only if you plan to use Anthropic models) — [install from Anthropic](https://docs.anthropic.com/en/docs/claude-code/overview)
- **Terminal** at least 80×24. Smaller terminals get a polite red warning.

## Usage

```bash
agentic-inferno \
  --writer-model gpt-4o \
  --critic-model deepseek-chat \
  --input document.md \
  --critic-style theatrical \
  --max-cost-usd 0.50 \
  --temperature 0.9
```

The Writer and Critic fire off independent API calls. You see both outputs streaming live in split panes. The status bar tracks cost in real time across both agents and any generated apologies.

Hit **Esc** or **q** to stop. The tool cancels in-flight requests and exits. No save dialog. No confirmation. The document you started with stays exactly where it was.

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--writer-model` | *(required)* | Model for the Writer agent (e.g. `gpt-4o`, `deepseek-reasoner`, `claude-sonnet-4-20250514`) |
| `--critic-model` | `deepseek-chat` | Model for the Critic agent. Cheap models recommended — the Critic produces entertainment, not value. |
| `--input` | *(required)* | Path to the document the Writer will revise. Must be outside the repo or inside `inputs/`. |
| `--max-cost-usd` | `2.0` | Ceiling in USD. When total spend hits this number the spectacle stops. |
| `--temperature` | `0.8` | Sampling temperature for both agents (0.0–2.0). |
| `--max-tokens` | `8192` | Maximum tokens per model response. |
| `--timeout-secs` | `120` | Request timeout in seconds. |
| `--critic-style` | `random` | Critic personality. One of: `aggressive`, `passive-aggressive`, `theatrical`, `academic-snob`, `disappointed`, `random`. |
| `--config` | *(none)* | Path to a TOML config file (see [TOML config](#toml-config)). |
| `--openai-base-url` | *(provider default)* | Override OpenAI API base URL. |
| `--deepseek-base-url` | *(provider default)* | Override DeepSeek API base URL. |
| `--moonshot-base-url` | *(provider default)* | Override Moonshot API base URL. |

Critic styles let you choose your preferred flavor of abuse:

- **aggressive** — Vicious, loud, merciless. Creative insults only.
- **passive-aggressive** — Venom wrapped in silk. "I'm sure you tried your best."
- **theatrical** — Shakespearean soliloquies of despair. "Alas, poor writing, I knew it well!"
- **academic-snob** — Cites imaginary papers with insufferable confidence. "As Smith et al. (2024) demonstrate in their seminal work..."
- **disappointed** — Not angry. Just profoundly disappointed. Parent-level guilt.
- **random** — Picks one of the above on every cycle. Keeps everyone on edge.

## `.env` setup

Create a `.env` file in the project root (or export these as environment variables):

```bash
# Required: at least one API key matching your chosen models
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
MOONSHOT_API_KEY=sk-...

# Optional: override default API endpoints
# OPENAI_BASE_URL=https://your-openai-proxy.example.com/v1
# DEEPSEEK_BASE_URL=https://your-deepseek-proxy.example.com/v1
# MOONSHOT_BASE_URL=https://your-moonshot-proxy.example.com/v1
```

Copy `.env.example` to `.env` and fill in the keys you need:

```bash
cp .env.example .env
```

Only the keys for the providers you actually use are required. The tool validates at startup and tells you exactly which ones are missing.

## TOML config

You can provide a TOML config file instead of (or in addition to) CLI flags. CLI flags take precedence over TOML values, which take precedence over defaults:

```toml
# inferno.toml
writer_model = "deepseek-reasoner"
critic_model = "deepseek-chat"
critic_style = "academic-snob"
input = "~/documents/my-essay.md"
max_cost_usd = 5.0
temperature = 0.9
max_tokens = 4096
timeout_secs = 60
openai_base_url = "https://custom.openai.com/v1"
```

```bash
agentic-inferno --config inferno.toml
```

Partial config files are fine — unspecified values fall through to defaults.

## TUI keys

| Key | Action |
|-----|--------|
| `Esc` / `q` | Stop the spectacle. Cancels in-flight requests and exits. |
| `Ctrl+C` | Hard quit — immediate exit, no draining. |
| `Tab` | Cycle focus between Writer (left) and Critic (right) panes. |
| `Up` / `Down` | Scroll focused pane by 1 line. |
| `PageUp` / `PageDown` | Scroll focused pane by 10 lines. |
| `Home` | Jump to the top of the focused pane. |
| `End` | Jump to the bottom (latest content) of the focused pane. |

The focused pane gets a yellow border. The unfocused Critic pane keeps a red border for atmosphere.

## Architecture

Three concurrent loops share a single document in memory:

```
┌──────────────────────────────────────────────────────┐
│ Writer Loop               Critic Loop                │
│                                                      │
│ 1. Read shared document  1. Read latest snapshot     │
│ 2. Read latest critique  2. Pick critic personality  │
│ 3. Call Writer LLM       3. Call Critic LLM          │
│ 4. Parse [APOLOGY]?      4. Count harsh keywords     │
│ 5. Update shared doc     5. Write critique           │
│ 6. Emit TUI events       6. Emit TUI events          │
│       ↓                        ↓                     │
│   [repeat]                  [repeat]                 │
│                                                      │
│              Shared Document (RwLock)                │
│                                                      │
│              Apology LLM call (spawned)              │
└──────────────────────────────────────────────────────┘
```

Both loops run independently. The Writer doesn't wait for the Critic and the Critic doesn't wait for the Writer. Each grabs the latest snapshot of the shared document and does its thing.

The **Writer** incorporates critique history into its prompt (with context-window pruning at 90%), revises the document, and emits the result. If the Critic has been particularly harsh — three or more distinct harsh keywords in a single response — an apology workflow triggers regardless of whether the Writer asked for one.

When the Writer explicitly appends `[APOLOGY]` to its output, a separate LLM call fires with the apology prompt. The Writer and Critic loops continue immediately — the apology runs in its own spawned task. A 30-second / 3-cycle cooldown prevents apology spam.

The **Critic** never scores, never suggests rewrites, and never outputs document prose. It's pure entertainment. The prompt for each personality style explicitly forbids constructive feedback.

The **shared document** lives in a `RwLock<SharedState>`. Version counters on every update let the Writer detect stale critiques without blocking. Semantic loop detection with a sliding window of 5 hashes (threshold: 3 repeats) halts the spectacle when output stagnates.

The **cost ceiling** is shared across all three loops. Each successful LLM call records its cost (estimated from token counts for OpenAI-compatible APIs, or via the `claude` CLI's built-in cost reporting). When the ceiling is hit, the cancellation token fires and all loops drain.

## Provider support

| Provider | Transport | Model prefix examples |
|----------|-----------|----------------------|
| **Anthropic** | `claude` CLI binary | `claude-*`, `opus`, `sonnet`, `haiku` |
| **OpenAI** | REST API (OpenAI-compatible) | `gpt-*`, `o1`, `o3-mini`, `o4-mini` |
| **DeepSeek** | REST API (OpenAI-compatible) | `deepseek-*` |
| **Moonshot** | REST API (OpenAI-compatible) | `kimi-*`, `moonshot-*` |

Model names are case-insensitive. Provider detection is order-sensitive — Anthropic patterns are checked first so `opus` routes correctly.

Anthropic models require the `claude` CLI on your PATH. The tool calls `claude --version` at startup and fails with a clear error if it's not found.

OpenAI, DeepSeek, and Moonshot all use the same OpenAI-compatible REST client under the hood. You can point them at any compatible proxy by setting the appropriate base URL.

## Cost warning

This tool makes API calls. API calls cost money. Typical usage runs between $0.50 and $2.00 per hour depending on your model choices.

Set `--max-cost-usd` to a number you're comfortable with. The default is $2.00. The status bar shows cumulative spend so you can see the meter running.

A cheap Critic model (`deepseek-chat` is the default) keeps costs down. The Writer model is where you'll burn most of your budget. If you use Anthropic models, the `claude` CLI reports costs natively. For OpenAI-compatible APIs, costs are estimated from prompt and completion token counts using each provider's published pricing.

The cost ceiling is a hard limit. When total spend hits it the spectacle stops immediately — no final revision, no graceful wrap-up. Set it low the first time you run.

## License

MIT — see [LICENSE](LICENSE).
