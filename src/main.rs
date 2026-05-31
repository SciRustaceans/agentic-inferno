use std::process::{ExitCode, Termination};

use agentic_inferno::config::{CliArgs, Config, TomlConfig};
use clap::Parser;

/// Custom exit code wrapper so `main` can return `ExitCode` directly.
struct CliExitCode(ExitCode);

impl Termination for CliExitCode {
    fn report(self) -> ExitCode {
        self.0
    }
}

fn main() -> CliExitCode {
    // Parse CLI args first so we know about --config and all required flags.
    let cli = CliArgs::parse();

    // Load .env — silent if file doesn't exist.
    let _ = dotenvy::from_filename_override(".env");

    // Optionally load TOML config.
    let toml = match cli.config.as_ref() {
        Some(path) => match TomlConfig::from_file(path) {
            Ok(tc) => Some(tc),
            Err(e) => {
                eprintln!("Error: {e}");
                return CliExitCode(ExitCode::from(e));
            }
        },
        None => None,
    };

    // Build and validate the unified config.
    let config = match Config::build(cli, toml) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error: {e}");
            return CliExitCode(ExitCode::from(e));
        }
    };

    // TODO: Launch the full app (orchestrator + TUI).
    // For now, print the resolved config as a smoke test.
    println!("Agentic Inferno — spectacle tool");
    println!("  Writer model: {}", config.writer_model);
    println!("  Critic model: {}", config.critic_model);
    println!("  Critic style: {}", config.critic_style);
    println!("  Input: {}", config.input.display());
    println!("  Max cost: ${:.2}", config.max_cost_usd);
    println!("  Temperature: {}", config.temperature);
    println!("  Max tokens: {}", config.max_tokens);
    println!("  Timeout: {}s", config.timeout_secs);
    println!("  Repo root: {}", config.repo_root.display());

    CliExitCode(ExitCode::SUCCESS)
}
