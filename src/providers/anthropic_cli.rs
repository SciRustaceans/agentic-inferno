//! Anthropic LLM provider backed by the local `claude` CLI binary.
//!
//! This module implements `LlmClient::complete()` for the `AnthropicCli` variant.
//! The `claude` binary is spawned as an async subprocess with the user prompt
//! piped on stdin (avoiding command-line argument length limits).  Three
//! complementary subprocess-reaping mechanisms guarantee cleanup:
//!
//! 1. `kill_on_drop(true)` — backstop for panics / early returns.
//! 2. `tokio::time::timeout` with explicit `child.kill().await` on elapse.
//! 3. The `Child` handle is kept alive until `.wait()` returns, so the OS
//!    properly reaps the zombie.
//!
//! Stdout is drained into a `Vec<u8>` via a dedicated reader task, preventing
//! deadlocks when the child produces more than the OS pipe buffer (typically
//! 64 KiB).

use std::process::Stdio;

use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::time::{timeout, Duration};

use crate::error::AppError;
use crate::providers::{ChatReply, ChatRequest, LlmClient};

/// The JSON object emitted by `claude --output-format json`.
#[derive(Debug, Deserialize)]
struct ClaudeCliResponse {
    /// Always `"result"` for message outputs.
    #[serde(rename = "type")]
    response_type: String,
    /// `"success"` on normal completion; other values indicate errors.
    subtype: String,
    /// Whether the CLI considered this an error response.
    #[serde(default)]
    is_error: bool,
    /// The response text (present when `!is_error`).
    result: Option<String>,
    /// Estimated cost in USD (optional; may be absent on error responses).
    total_cost_usd: Option<f64>,
}

impl LlmClient {
    /// Send a chat request to the configured LLM and return the reply.
    ///
    /// # Dispatch
    ///
    /// - `AnthropicCli` → spawns `claude` CLI locally (this module).
    /// - `OpenAiCompat` → REST API (implemented in a separate task).
    pub async fn complete(
        &self,
        request: ChatRequest,
        timeout_secs: u64,
    ) -> Result<ChatReply, AppError> {
        match self {
            LlmClient::OpenAiCompat {
                base_url,
                api_key,
                model,
                http,
            } => {
                let _ = timeout_secs; // HTTP client timeout configured at build time
                super::openai_compat::do_complete(
                    base_url, api_key, model, http, &request,
                )
                .await
            }
            LlmClient::AnthropicCli { model, claude_bin } => {
                anthropic_complete(claude_bin, model, request, timeout_secs).await
            }
        }
    }
}

/// Core implementation: spawn `claude`, pipe the prompt, parse the response.
async fn anthropic_complete(
    claude_bin: &str,
    model: &str,
    request: ChatRequest,
    timeout_secs: u64,
) -> Result<ChatReply, AppError> {
    let duration = Duration::from_secs(timeout_secs);

    // ── Spawn the claude subprocess ──────────────────────────────────

    let mut child = Command::new(claude_bin)
        .args([
            "-p",
            "--model",
            model,
            "--output-format",
            "json",
            "--system-prompt",
            &request.system,
            "--disallowedTools",
            "*",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true) // Mechanism 1: backstop for panic / early return
        .spawn()
        .map_err(|e| AppError::ClaudeCli {
            subtype: "spawn".into(),
            message: format!("Failed to spawn `{claude_bin}`: {e}"),
        })?;

    // ── Pipe user prompt on stdin, then close to signal EOF ─────────

    {
        let mut stdin = child
            .stdin
            .take()
            .expect("stdin was configured as Stdio::piped");
        stdin
            .write_all(request.user.as_bytes())
            .await
            .map_err(|e| AppError::ClaudeCli {
                subtype: "stdin".into(),
                message: format!("Failed to write prompt to claude stdin: {e}"),
            })?;
        // `stdin` is dropped here → EOF sent to the claude subprocess.
    }

    // ── Drain stdout asynchronously (prevents pipe-buffer deadlock) ──
    //
    // If the child produces more than ~64 KiB of output and Rust doesn't
    // read, the child's write(2) blocks on a full pipe.  A dedicated
    // reader task keeps the pipe draining regardless of timeout logic.

    let mut stdout = child.stdout.take().expect("stdout was piped");
    let mut stderr_pipe = child.stderr.take().expect("stderr was piped");

    let stdout_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        stdout.read_to_end(&mut buf).await.map(|_| buf)
    });

    // Drain stderr in parallel for diagnostic purposes.
    let stderr_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf).await;
        String::from_utf8_lossy(&buf).to_string()
    });

    // ── Wait with timeout + explicit kill (Mechanism 2) ──────────────
    //
    // `tokio::time::timeout` does NOT kill the child on elapse — it only
    // cancels the future.  We must call `.kill().await` explicitly.

    let exit_status = match timeout(duration, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(e)) => {
            return Err(AppError::ClaudeCli {
                subtype: "wait".into(),
                message: format!("OS error while waiting for claude process: {e}"),
            });
        }
        Err(_elapsed) => {
            // Timeout fired — kill the child, then block until reaped.
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(AppError::Timeout);
        }
    };

    // Mechanism 3: `child` handle is still alive here; `.wait()` has
    // returned, so the OS has fully reaped the zombie process.

    // ── Collect reader-task results ──────────────────────────────────

    let stdout_buf = stdout_task
        .await
        .map_err(|join_err| AppError::ClaudeCli {
            subtype: "reader_panic".into(),
            message: format!("Stdout reader task panicked: {join_err}"),
        })?
        .map_err(|e| AppError::ClaudeCli {
            subtype: "read".into(),
            message: format!("Failed to read claude stdout: {e}"),
        })?;

    let stderr_str = stderr_task.await.unwrap_or_default();

    // ── Check exit status ───────────────────────────────────────────

    if !exit_status.success() {
        return Err(AppError::ClaudeCli {
            subtype: "exit".into(),
            message: format!(
                "claude CLI exited with {exit_status}. stderr: {stderr_str}"
            ),
        });
    }

    // ── Parse JSON ──────────────────────────────────────────────────

    let response: ClaudeCliResponse = serde_json::from_slice(&stdout_buf).map_err(|e| {
        let snippet = String::from_utf8_lossy(&stdout_buf);
        let snippet = if snippet.len() > 500 {
            format!("{}... [truncated]", &snippet[..500])
        } else {
            snippet.to_string()
        };
        AppError::ClaudeCli {
            subtype: "parse".into(),
            message: format!(
                "Failed to parse claude JSON output ({e}). Raw output: {snippet}"
            ),
        }
    })?;

    // ── Validate response envelope ──────────────────────────────────

    if response.response_type != "result" {
        return Err(AppError::ClaudeCli {
            subtype: "unexpected_type".into(),
            message: format!(
                "Expected response type 'result', got '{}'. stderr: {stderr_str}",
                response.response_type,
            ),
        });
    }

    if response.is_error || response.subtype != "success" {
        return Err(AppError::ClaudeCli {
            subtype: response.subtype,
            message: response.result.unwrap_or_else(|| {
                format!(
                    "claude CLI returned an error with no message. stderr: {stderr_str}"
                )
            }),
        });
    }

    let text = response.result.ok_or_else(|| AppError::ClaudeCli {
        subtype: "empty".into(),
        message: "claude CLI returned success with no result text".into(),
    })?;

    Ok(ChatReply {
        text,
        cost_usd: response.total_cost_usd,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the `ClaudeCliResponse` struct correctly deserializes
    /// a typical success response from the `claude` CLI.
    #[test]
    fn test_deserialize_claude_success() {
        let json = br#"{
            "type": "result",
            "subtype": "success",
            "is_error": false,
            "result": "Hello, world!",
            "total_cost_usd": 0.019
        }"#;

        let resp: ClaudeCliResponse = serde_json::from_slice(json).unwrap();
        assert_eq!(resp.response_type, "result");
        assert_eq!(resp.subtype, "success");
        assert!(!resp.is_error);
        assert_eq!(resp.result.as_deref(), Some("Hello, world!"));
        assert!((resp.total_cost_usd.unwrap() - 0.019).abs() < 1e-9);
    }

    /// Verify that an error response with `is_error: true` deserializes
    /// correctly and the error message is captured.
    #[test]
    fn test_deserialize_claude_error() {
        let json = br#"{
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": true,
            "result": "Something went wrong"
        }"#;

        let resp: ClaudeCliResponse = serde_json::from_slice(json).unwrap();
        assert!(resp.is_error);
        assert_eq!(resp.subtype, "error_during_execution");
        assert_eq!(resp.result.as_deref(), Some("Something went wrong"));
    }

    /// Verify that a response missing `is_error` (defaults to `false`)
    /// deserializes correctly.
    #[test]
    fn test_deserialize_claude_no_is_error() {
        let json = br#"{
            "type": "result",
            "subtype": "success",
            "result": "works",
            "total_cost_usd": 0.0
        }"#;

        let resp: ClaudeCliResponse = serde_json::from_slice(json).unwrap();
        assert!(!resp.is_error);
        assert_eq!(resp.subtype, "success");
    }
}
