#!/usr/bin/env bash
#
# install.sh — set up agentic-inferno on macOS / Linux.
#
# Installs Rust (if missing), builds the release binary, and walks you
# through entering API keys into a .env file. Safe to re-run.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh

set -euo pipefail

# Run from the repository root (the directory holding this script).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

KEYS=(ANTHROPIC_API_KEY OPENAI_API_KEY DEEPSEEK_API_KEY MOONSHOT_API_KEY)

info() { printf '\n==> %s\n' "$1"; }
warn() { printf '\n[warn] %s\n' "$1" >&2; }

# --- 1. Ensure Rust -------------------------------------------------------

ensure_rust() {
  if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    info "Rust found: $(rustc --version)"
    return
  fi

  info "Rust not found. Installing via rustup..."
  if ! command -v curl >/dev/null 2>&1; then
    warn "curl is required to install Rust automatically. Install Rust manually from https://rustup.rs and re-run."
    exit 1
  fi

  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # Make cargo available in the current shell for the rest of this run.
  # shellcheck disable=SC1090,SC1091
  if [ -f "$HOME/.cargo/env" ]; then
    . "$HOME/.cargo/env"
  fi

  if ! command -v cargo >/dev/null 2>&1; then
    warn "cargo is still not on PATH. Open a new shell (or source ~/.cargo/env) and re-run this script."
    exit 1
  fi
  rustup default stable >/dev/null 2>&1 || true
  info "Rust installed: $(rustc --version)"
}

# --- 2. Build -------------------------------------------------------------

build_release() {
  info "Building release binary (cargo build --release)..."
  cargo build --release
  info "Build complete: ./target/release/agentic-inferno"
}

# --- 3. API-key setup -----------------------------------------------------

# Look up the current value of KEY in .env, printing it on stdout.
# Empty output means the key is absent or still the placeholder.
current_value() {
  local key="$1"
  local line
  line="$(grep -E "^${key}=" .env 2>/dev/null | head -n1 || true)"
  printf '%s' "${line#"${key}"=}"
}

# Replace (or append) the KEY=value line in .env with a literal value.
# Rebuilds the file via a temp copy so no sed -i portability issues arise
# and so the value is never reinterpreted (API keys may contain $, & etc.).
set_env_value() {
  local key="$1"
  local value="$2"
  local tmp
  tmp="$(mktemp)"
  local replaced=0

  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" == "${key}="* ]]; then
      printf '%s=%s\n' "$key" "$value" >>"$tmp"
      replaced=1
    else
      printf '%s\n' "$line" >>"$tmp"
    fi
  done <.env

  if [ "$replaced" -eq 0 ]; then
    printf '%s=%s\n' "$key" "$value" >>"$tmp"
  fi

  mv "$tmp" .env
}

setup_env() {
  info "Setting up API keys in .env"

  if [ ! -f .env ]; then
    if [ -f .env.example ]; then
      cp .env.example .env
      echo "Created .env from .env.example."
    else
      : >.env
      echo "Created an empty .env (.env.example not found)."
    fi
  fi

  echo "Enter each API key (input hidden). Leave blank to skip a provider."
  echo

  for key in "${KEYS[@]}"; do
    local existing
    existing="$(current_value "$key")"
    local has_real=0
    if [ -n "$existing" ] && [[ "$existing" != sk-...* ]] && [[ "$existing" != sk-ant-...* ]]; then
      has_real=1
    fi

    if [ "$has_real" -eq 1 ]; then
      printf '%s already has a value. Overwrite? [y/N]: ' "$key"
      local ans
      read -r ans
      if [[ ! "$ans" =~ ^[Yy]$ ]]; then
        echo "Keeping existing $key."
        continue
      fi
    fi

    local value
    read -rsp "$key: " value
    echo
    if [ -z "$value" ]; then
      echo "Skipped $key."
      continue
    fi
    set_env_value "$key" "$value"
    echo "Saved $key."
  done

  info ".env updated."
}

# --- 4. Optional: claude CLI check ---------------------------------------

check_claude_cli() {
  if command -v claude >/dev/null 2>&1; then
    info "claude CLI found (needed only for Anthropic models)."
  else
    warn "claude CLI not found. It is only required if you plan to use Anthropic models (claude-*, opus, sonnet, haiku). Install: https://docs.anthropic.com/en/docs/claude-code/overview"
  fi
}

# --- Main -----------------------------------------------------------------

main() {
  ensure_rust
  build_release
  setup_env
  check_claude_cli

  info "Done."
  echo "Run it, for example:"
  echo "  ./target/release/agentic-inferno --writer-model gpt-4o --input my-draft.md"
}

main "$@"
