use std::process::ExitCode;

use thiserror::Error;

/// All error types for the agentic-inferno spectacle tool.
///
/// Errors are user-actionable — messages tell the user what went wrong
/// and what to do next. This is library code; `anyhow` is reserved for
/// `main.rs` only.
#[derive(Debug, Error)]
pub enum AppError {
    /// A required API key is missing from the environment.
    #[error("Missing API key: {0}. Set it in .env or export it as an environment variable.")]
    MissingKey(String),

    /// An unknown model name was provided for a specific agent role.
    #[error("Unknown model '{0}' for {1}. Check the model name and try again.")]
    UnknownModel(String, String),

    /// An HTTP error was returned by a provider API.
    #[error("HTTP {status}: {body}")]
    Http {
        status: u16,
        body: String,
    },

    /// A network/transport error occurred (connection refused, DNS failure, etc.).
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// The request exceeded the configured timeout.
    #[error("Request timed out. Increase --timeout-secs or check network connectivity.")]
    Timeout,

    /// The `claude` CLI returned an error (is_error, unexpected subtype, or non-zero exit).
    #[error("Claude CLI error ({subtype}): {message}")]
    ClaudeCli {
        subtype: String,
        message: String,
    },

    /// A shell script exited with a non-zero code (reserved for future use).
    #[error("Script exited with code {code}: {stderr}")]
    Script {
        code: i32,
        stderr: String,
    },

    /// An I/O error occurred (file not found, permission denied, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The `--input` flag was missing or the specified file was not found.
    #[error("Missing input: provide a valid file path with --input <path>")]
    MissingInput,

    /// The input file is inside the repository but not in the `inputs/` directory.
    #[error("Leak guard: input file '{0}' is inside the repo but not in inputs/. Place input files in inputs/ or outside the repo.")]
    LeakGuard(String),

    /// The cost ceiling has been exceeded.
    #[error("Cost ceiling exceeded: spent ${0:.2}, limit ${1:.2}. Increase --max-cost-usd or set a higher limit.")]
    CostCeilingExceeded(f64, f64),

    /// The loop detection mechanism triggered — the output has become repetitive.
    #[error("Loop exhausted: {0}")]
    LoopExhausted(String),

    /// The `claude` CLI binary was not found on PATH but an Anthropic model was selected.
    #[error("claude CLI not found on PATH. Install it from https://claude.ai/download or select a non-Anthropic model.")]
    ClaudeNotFound,

    /// The terminal is too small for the TUI layout.
    #[error("Terminal too small: {0}x{1}. Minimum size is 80x24. Resize your terminal and try again.")]
    TerminalTooSmall(usize, usize),
}

impl From<AppError> for ExitCode {
    fn from(err: AppError) -> Self {
        match err {
            // Configuration / user errors → 64 (EX_USAGE)
            AppError::MissingKey(_)
            | AppError::UnknownModel(_, _)
            | AppError::MissingInput
            | AppError::LeakGuard(_)
            | AppError::ClaudeNotFound
            | AppError::TerminalTooSmall(_, _) => ExitCode::from(64),

            // Runtime / transient errors → 70 (EX_SOFTWARE)
            AppError::Http { .. }
            | AppError::Network(_)
            | AppError::Timeout
            | AppError::ClaudeCli { .. }
            | AppError::Script { .. }
            | AppError::Io(_)
            | AppError::CostCeilingExceeded(_, _)
            | AppError::LoopExhausted(_) => ExitCode::from(70),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_key_message() {
        let err = AppError::MissingKey("OPENAI_API_KEY".into());
        let msg = err.to_string();
        assert!(msg.contains("OPENAI_API_KEY"));
    }

    #[test]
    fn test_unknown_model_includes_agent() {
        let err = AppError::UnknownModel("gpt-7".into(), "Critic".into());
        let msg = err.to_string();
        assert!(msg.contains("gpt-7"));
        assert!(msg.contains("Critic"));
    }

    #[test]
    fn test_http_message() {
        let err = AppError::Http {
            status: 429,
            body: "Too many requests".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("429"));
        assert!(msg.contains("Too many requests"));
    }

    #[test]
    fn test_timeout_message() {
        let err = AppError::Timeout;
        assert!(err.to_string().contains("timed out"));
    }

    #[test]
    fn test_claude_cli_message() {
        let err = AppError::ClaudeCli {
            subtype: "is_error".into(),
            message: "Model not available".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("is_error"));
        assert!(msg.contains("Model not available"));
    }

    #[test]
    fn test_script_message() {
        let err = AppError::Script {
            code: 1,
            stderr: "something failed".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("1"));
        assert!(msg.contains("something failed"));
    }

    #[test]
    fn test_missing_input_message() {
        let err = AppError::MissingInput;
        assert!(err.to_string().contains("--input"));
    }

    #[test]
    fn test_leak_guard_message() {
        let err = AppError::LeakGuard("./src/lib.rs".into());
        assert!(err.to_string().contains("./src/lib.rs"));
    }

    #[test]
    fn test_cost_ceiling_message() {
        let err = AppError::CostCeilingExceeded(2.50, 2.00);
        let msg = err.to_string();
        assert!(msg.contains("2.50"));
        assert!(msg.contains("2.00"));
    }

    #[test]
    fn test_loop_exhausted_message() {
        let err = AppError::LoopExhausted("Semantic similarity threshold exceeded".into());
        assert!(err.to_string().contains("Semantic similarity"));
    }

    #[test]
    fn test_claude_not_found_message() {
        let err = AppError::ClaudeNotFound;
        assert!(err.to_string().contains("PATH"));
    }

    #[test]
    fn test_terminal_too_small_message() {
        let err = AppError::TerminalTooSmall(40, 10);
        let msg = err.to_string();
        assert!(msg.contains("40"));
        assert!(msg.contains("10"));
        assert!(msg.contains("80x24"));
    }

    #[test]
    fn test_io_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let app_err: AppError = io_err.into();
        assert!(app_err.to_string().contains("file not found"));
    }

    #[test]
    fn test_exit_code_mapping() {
        // Helper to avoid ambiguous `.into()` in test assertions.
        fn to_exit_code(err: AppError) -> ExitCode {
            ExitCode::from(err)
        }

        // Config/user errors → 64
        assert_eq!(to_exit_code(AppError::MissingKey("X".into())), ExitCode::from(64));
        assert_eq!(
            to_exit_code(AppError::UnknownModel("x".into(), "Writer".into())),
            ExitCode::from(64)
        );
        assert_eq!(to_exit_code(AppError::MissingInput), ExitCode::from(64));
        assert_eq!(to_exit_code(AppError::LeakGuard("x".into())), ExitCode::from(64));
        assert_eq!(to_exit_code(AppError::ClaudeNotFound), ExitCode::from(64));
        assert_eq!(to_exit_code(AppError::TerminalTooSmall(40, 10)), ExitCode::from(64));

        // Runtime errors → 70
        assert_eq!(
            to_exit_code(AppError::Http {
                status: 500,
                body: "".into()
            }),
            ExitCode::from(70)
        );
        assert_eq!(to_exit_code(AppError::Timeout), ExitCode::from(70));
        assert_eq!(
            to_exit_code(AppError::ClaudeCli {
                subtype: "".into(),
                message: "".into()
            }),
            ExitCode::from(70)
        );
        assert_eq!(
            to_exit_code(AppError::Script {
                code: 1,
                stderr: "".into()
            }),
            ExitCode::from(70)
        );
        assert_eq!(
            to_exit_code(AppError::Io(std::io::Error::new(std::io::ErrorKind::Other, "x"))),
            ExitCode::from(70)
        );
        assert_eq!(
            to_exit_code(AppError::CostCeilingExceeded(1.0, 1.0)),
            ExitCode::from(70)
        );
        assert_eq!(
            to_exit_code(AppError::LoopExhausted("x".into())),
            ExitCode::from(70)
        );
    }
}
