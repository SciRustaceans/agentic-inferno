use std::process::{ExitCode, Termination};

use agentic_inferno::config::{CliArgs, Config, InfernoTask, TomlConfig};
use agentic_inferno::error::AppError;
use agentic_inferno::orchestrator;
use agentic_inferno::state::SharedState;
use agentic_inferno::tui::{install_panic_hook, Tui};
use clap::Parser;
use tokio_util::sync::CancellationToken;

/// Custom exit code wrapper so `main` can return `ExitCode` directly.
struct CliExitCode(ExitCode);

impl Termination for CliExitCode {
    fn report(self) -> ExitCode {
        self.0
    }
}

fn main() -> CliExitCode {
    let cli = CliArgs::parse();

    let _ = dotenvy::from_filename_override(".env");

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

    let config = match Config::build(cli, toml) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error: {e}");
            return CliExitCode(ExitCode::from(e));
        }
    };

    // Choose the seed source: prompt mode has no input file and starts empty;
    // every other task reads the validated --input file.
    let initial_content = if config.task == InfernoTask::Prompt {
        String::new()
    } else {
        match std::fs::read_to_string(&config.input) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Error: failed to read input file: {e}");
                return CliExitCode(ExitCode::from(AppError::Io(e)));
            }
        }
    };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Error: failed to create async runtime: {e}");
            return CliExitCode(ExitCode::from(64));
        }
    };

    let result = rt.block_on(run_spectacle_app(config, initial_content));

    match result {
        Ok(()) => CliExitCode(ExitCode::SUCCESS),
        Err(e) => {
            eprintln!("Error: {e}");
            CliExitCode(ExitCode::from(e))
        }
    }
}

async fn run_spectacle_app(config: Config, initial_content: String) -> Result<(), AppError> {
    let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
    let cancel_token = CancellationToken::new();
    let state = SharedState::new(initial_content);

    let task_label = config.task.to_string();

    let spectacle_handle = tokio::spawn(orchestrator::run_spectacle(
        config.clone(),
        state.clone(),
        event_tx,
        cancel_token.clone(),
    ));

    install_panic_hook();
    let (mut tui, _guard) = Tui::enter(cancel_token.clone())?;
    tui.run(event_rx, task_label).await?;
    Tui::exit()?;

    spectacle_handle
        .await
        .expect("orchestrator task panicked")?;

    Ok(())
}
