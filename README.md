# agentic-inferno

A terminal app that runs two LLM agents at once: a Writer that keeps revising a
document, and a Critic that heckles every revision. The Writer never tries to
finish. It just keeps reworking the document while the Critic responds to each
version. When the criticism gets harsh, the Writer stops and apologizes to you.

You watch. The Writer and Critic each make their own API calls and you see both
outputs in split panes. There's no convergence and no "done" state. The loop
runs until you hit Esc or the cost ceiling is reached. Your original document is
never touched.

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
#./target/release/agentic-inferno --writer-model gpt-4o --input my-draft.md
```

## Install scripts

The install scripts install Rust if it's missing, build the release binary, and
then prompt you for each API key and write the ones you enter into `.env`. Input
is hidden as you type, and blank input skips that provider. Re-running is safe —
it won't overwrite a real key without asking.

**macOS / Linux:**

```bash
chmod +x install.sh
./install.sh
```

**Windows:** see the [Windows](#windows) section.

The `.env` file and `--input` paths work the same on every platform.

## Windows

On Windows, run the install script from a PowerShell prompt in the cloned repo:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

`install.ps1` installs Rust (via `winget`, falling back to `rustup-init.exe`),
checks for a C linker, builds the release binary, and prompts for your API keys
into `.env`.

Prerequisites on Windows:

- **A C linker.** The default Rust toolchain links with MSVC, so you need the
  **"Desktop development with C++"** workload from the
  [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
  `install.ps1` tries to install this for you. If you'd rather not install
  Visual Studio, use the GNU toolchain instead:

  ```powershell
  rustup toolchain install stable-x86_64-pc-windows-gnu
  rustup default stable-x86_64-pc-windows-gnu
  ```

- **claude CLI** — only if you plan to use Anthropic models. The app looks for
  `claude.cmd`, `claude.exe`, or `claude` on your PATH.

Once Windows release artifacts are published, a one-line PowerShell installer
will fetch a prebuilt binary (no Rust needed):

```powershell
# Available once Windows release artifacts ship.
powershell -c "irm https://github.com/jojomatik/agentic-inferno/releases/latest/download/agentic-inferno-installer.ps1 | iex"
```

That installer only fetches the binary — use `install.ps1` (or copy
`.env.example` to `.env`) to set up your API keys.

## Prerequisites

- **Rust** (1.80+) for building from source
- **API keys** set in a `.env` file or exported as environment variables (see [`.env` setup](#env-setup))
- **claude CLI** (optional, only if you plan to use Anthropic models) — [install from Anthropic](https://docs.anthropic.com/en/docs/claude-code/overview)
- **Terminal** at least 80×24. Smaller terminals get a red warning instead of the layout.

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

The Writer and Critic make independent API calls. Both outputs stream live in
split panes. The status bar tracks cost in real time across both agents and any
apologies.

Hit **Esc** or **q** to stop. The tool cancels in-flight requests and exits.
There's no save dialog and no confirmation. The document you started with is left
exactly as it was.

### Tasks and prompt mode

By default the Writer revises a piece of prose. Use `--task` to point it at a
different kind of work:

```bash
agentic-inferno --writer-model gpt-4o --task code --input src/lib.rs
```

`prompt` mode is different. Instead of an input file, you give the Writer a
free-form goal with `--prompt`. It treats the goal as something it can never
fully reach and keeps attempting it while the Critic piles on. No `--input` is
needed:

```bash
agentic-inferno --writer-model gpt-4o --prompt "prove that 1 equals 2"
```

Passing `--prompt` selects `--task prompt` automatically unless you set `--task`
yourself.

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--writer-model` | *(required)* | Model for the Writer agent (e.g. `gpt-4o`, `deepseek-reasoner`, `claude-sonnet-4-20250514`) |
| `--critic-model` | `deepseek-chat` | Model for the Critic agent. A cheap model is fine here — the Critic only produces commentary. |
| `--task` | `writing` | What the Writer works on. One of: `writing`, `code`, `research`, `analysis`, `prompt`. |
| `--input` | *(required unless `--task prompt`)* | Path to the document the Writer will revise. Must be outside the repo or inside `inputs/`. Ignored in prompt mode. |
| `--prompt` | *(none)* | A free-form goal for the Writer to keep attempting. Required for `--task prompt`; supplying it implies `--task prompt`. |
| `--max-cost-usd` | `2.0` | Cost ceiling in USD. When total spend reaches this, the loop stops. |
| `--temperature` | `0.8` | Sampling temperature for both agents (0.0–2.0). |
| `--max-tokens` | `8192` | Maximum tokens per model response. |
| `--timeout-secs` | `120` | Request timeout in seconds. |
| `--critic-style` | `random` | Critic personality. One of: `aggressive`, `passive-aggressive`, `theatrical`, `academic-snob`, `disappointed`, `random`. |
| `--config` | *(none)* | Path to a TOML config file (see [TOML config](#toml-config)). |
| `--openai-base-url` | *(provider default)* | Override OpenAI API base URL. |
| `--deepseek-base-url` | *(provider default)* | Override DeepSeek API base URL. |
| `--moonshot-base-url` | *(provider default)* | Override Moonshot API base URL. |

The critic styles control the tone of the heckling:

- **aggressive** — Loud and harsh. Insults, no advice.
- **passive-aggressive** — Backhanded compliments and dry sarcasm. "I'm sure you tried your best."
- **theatrical** — Dramatic, Shakespearean despair. "Alas, poor writing, I knew it well!"
- **academic-snob** — Cites made-up papers with total confidence. "As Smith et al. (2024) demonstrate..."
- **disappointed** — Not angry, just disappointed. Parent-style guilt.
- **random** — Picks one of the above each cycle.

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

You only need the keys for the providers you actually use. The tool validates at
startup and tells you which ones are missing.

## TOML config

You can use a TOML config file instead of (or alongside) CLI flags. CLI flags
take precedence over TOML values, which take precedence over defaults:

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

Partial config files are fine. Anything you leave out falls through to defaults.

## TUI keys

| Key | Action |
|-----|--------|
| `Esc` / `q` | Stop. Cancels in-flight requests and exits. |
| `Ctrl+C` | Hard quit — immediate exit, no draining. |
| `Tab` | Cycle focus between Writer (left) and Critic (right) panes. |
| `Up` / `Down` | Scroll focused pane by 1 line. |
| `PageUp` / `PageDown` | Scroll focused pane by 10 lines. |
| `Home` | Jump to the top of the focused pane. |
| `End` | Jump to the bottom (latest content) of the focused pane. |

The focused pane gets a yellow border. The unfocused Critic pane keeps a red one.

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

The two loops run independently. The Writer doesn't wait for the Critic and the
Critic doesn't wait for the Writer. Each one grabs the latest snapshot of the
shared document and works from that.

The **Writer** folds the critique history into its prompt (pruned at 90% of the
context window), revises the document, and emits the result. If the Critic was
especially harsh — three or more distinct harsh keywords in one response — an
apology fires whether or not the Writer asked for one.

When the Writer appends `[APOLOGY]` to its output, a separate LLM call runs with
the apology prompt. The Writer and Critic loops keep going while the apology runs
in its own spawned task. A 30-second / 3-cycle cooldown keeps apologies from
spamming.

The **Critic** never scores the work, never suggests rewrites, and never outputs
document prose. The prompt for each personality forbids constructive feedback.

The **shared document** lives in a `RwLock<SharedState>`. A version counter on
every update lets the Writer notice stale critiques without blocking. Loop
detection over a sliding window of 5 hashes (threshold: 3 repeats) stops the run
when output stops changing.

The **cost ceiling** is shared across all three loops. Each successful LLM call
records its cost (estimated from token counts for OpenAI-compatible APIs, or read
from the `claude` CLI's own cost reporting). When the ceiling is reached, the
cancellation token fires and the loops drain.

## Provider support

| Provider | Transport | Model prefix examples |
|----------|-----------|----------------------|
| **Anthropic** | `claude` CLI binary | `claude-*`, `opus`, `sonnet`, `haiku` |
| **OpenAI** | REST API (OpenAI-compatible) | `gpt-*`, `o1`, `o3-mini`, `o4-mini` |
| **DeepSeek** | REST API (OpenAI-compatible) | `deepseek-*` |
| **Moonshot** | REST API (OpenAI-compatible) | `kimi-*`, `moonshot-*` |

Model names are case-insensitive. Provider detection is order-sensitive —
Anthropic patterns are checked first so `opus` routes correctly.

Anthropic models need the `claude` CLI on your PATH. The tool runs
`claude --version` at startup and fails with a clear error if it isn't found.

OpenAI, DeepSeek, and Moonshot share the same OpenAI-compatible REST client. You
can point any of them at a compatible proxy by setting the matching base URL.

## Cost

This tool makes API calls, and API calls cost money. Typical usage runs between
$0.50 and $2.00 per hour depending on the models you pick.

Set `--max-cost-usd` to a number you're comfortable with. The default is $2.00.
The status bar shows cumulative spend so you can watch the meter.

A cheap Critic model (the default is `deepseek-chat`) keeps costs down. The
Writer model is where most of the budget goes. Anthropic models report their cost
through the `claude` CLI. For OpenAI-compatible APIs, cost is estimated from
prompt and completion token counts using each provider's published pricing.

The cost ceiling is a hard limit. When total spend reaches it, the loop stops
immediately — no final revision, no wrap-up. Set it low the first time you run.

## License

MIT — see [LICENSE](LICENSE).
</content>
</invoke>
