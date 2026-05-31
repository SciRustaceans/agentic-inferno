use std::time::Duration;

use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use crate::app::AppEvent;
use crate::config::Config;
use crate::error::AppError;
use crate::prompts::WRITER_SYSTEM_PROMPT;
use crate::providers::{detect_provider, ChatRequest, LlmClient, Provider};
use crate::state::SharedState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Mock critique string used until the real Critic loop is implemented (Task 16).
const MOCK_CRITIQUE: &str =
    "Your writing lacks clarity, structure, and purpose. Try again.";

/// Pause between Writer loop cycles to avoid tight spinning on the LLM API.
const CYCLE_SLEEP_MS: u64 = 500;

// ---------------------------------------------------------------------------
// Orchestrator entry point
// ---------------------------------------------------------------------------

/// Launch the Writer-only spectacle loop.
///
/// Reads the input document, initialises [`SharedState`], builds the Writer
/// [`LlmClient`], and spawns a background task that continuously revises the
/// document.  The Critic is mocked for now — a static critique string is fed
/// to the Writer each cycle.
///
/// # Lifecycle
///
/// 1. Load and validate the input document.
/// 2. Put initial content into `SharedState` and send a `WriterOutput` event
///    so the TUI displays the starting document.
/// 3. Build the LLM client for the Writer model.
/// 4. Spawn the Writer loop as a `tokio::spawn`ed task.
/// 5. Wait for the `cancel_token` (triggered by Esc/q in the TUI).
/// 6. Send `AppEvent::Shutdown` to the TUI.
pub async fn run_spectacle(
    config: Config,
    state: SharedState,
    event_tx: UnboundedSender<AppEvent>,
    cancel_token: CancellationToken,
) -> Result<(), AppError> {
    let initial_content = std::fs::read_to_string(&config.input)?;

    state.update(initial_content.clone());
    let _ = event_tx.send(AppEvent::WriterOutput(initial_content));

    let writer_client = build_llm_client(&config)?;

    let task_event_tx = event_tx.clone();
    let task_state = state.clone();
    let task_cancel = cancel_token.clone();

    tokio::spawn(async move {
        writer_loop(
            writer_client,
            task_state,
            task_event_tx,
            task_cancel,
            config,
        )
        .await;
    });

    cancel_token.cancelled().await;

    let _ = event_tx.send(AppEvent::Shutdown);

    Ok(())
}

// ---------------------------------------------------------------------------
// Writer loop (runs in a spawned task)
// ---------------------------------------------------------------------------

/// Core Writer loop — continuously revises the document with LLM calls.
///
/// Snapshot the current document, build a prompt with the latest critique,
/// call the LLM, update shared state, and emit events to the TUI.  Errors
/// are reported via `AppEvent::Error` but the loop continues.
///
/// # Cancellation
///
/// Uses a `tokio::select!` on the `cancel_token.cancelled()` future and a
/// short sleep so the loop responds to cancellation within ~500ms.
async fn writer_loop(
    client: LlmClient,
    state: SharedState,
    event_tx: UnboundedSender<AppEvent>,
    cancel_token: CancellationToken,
    config: Config,
) {
    let mut total_cost_usd: f64 = 0.0;

    loop {
        if cancel_token.is_cancelled() {
            break;
        }

        let (_version, current_content) = state.snapshot();

        let prompt_text = format!("{}\n\n[CRITIQUE]: {}", current_content, MOCK_CRITIQUE);

        let request = ChatRequest {
            system: WRITER_SYSTEM_PROMPT.to_string(),
            user: prompt_text,
            model: config.writer_model.clone(),
            temperature: config.temperature as f32,
            max_tokens: config.max_tokens,
        };

        match client.complete(request, config.timeout_secs).await {
            Ok(reply) => {
                if let Some(cost) = reply.cost_usd {
                    total_cost_usd += cost;
                    eprintln!(
                        "Writer: cycle cost ${cost:.6}, total ${total_cost_usd:.6}",
                    );
                }

                let new_version = state.update(reply.text.clone());
                let _ = event_tx.send(AppEvent::WriterOutput(reply.text));
                let _ = event_tx.send(AppEvent::WriterDone(new_version));
            }
            Err(err) => {
                let _ = event_tx.send(AppEvent::Error(err));
            }
        }

        tokio::select! {
            _ = cancel_token.cancelled() => {
                break;
            }
            _ = tokio::time::sleep(Duration::from_millis(CYCLE_SLEEP_MS)) => {}
        }
    }
}

// ---------------------------------------------------------------------------
// LLM client construction
// ---------------------------------------------------------------------------

/// Build an [`LlmClient`] for the Writer model based on the resolved configuration.
///
/// Detects the provider from the model name, reads the API key from the
/// environment, resolves the base URL (config override → env var → default),
/// and constructs the appropriate client variant.
fn build_llm_client(config: &Config) -> Result<LlmClient, AppError> {
    let (provider, model_name) = detect_provider(&config.writer_model, "Writer")?;

    match provider {
        Provider::Anthropic => Ok(LlmClient::AnthropicCli {
            model: model_name,
            claude_bin: "claude".to_string(),
        }),
        _ => {
            let env_var = provider.api_key_env_var();
            let api_key =
                std::env::var(env_var).map_err(|_| AppError::MissingKey(env_var.to_string()))?;

            let base_url = resolve_base_url(&provider, config)?;

            let http = reqwest::Client::builder()
                .timeout(Duration::from_secs(config.timeout_secs))
                .build()
                .map_err(AppError::Network)?;

            Ok(LlmClient::OpenAiCompat {
                base_url,
                api_key,
                model: model_name,
                http,
            })
        }
    }
}

/// Resolve the base URL for an OpenAI-compatible provider.
///
/// Resolution order:
/// 1. Config override.
/// 2. Environment variable override.
/// 3. Provider default.
fn resolve_base_url(provider: &Provider, config: &Config) -> Result<reqwest::Url, AppError> {
    let config_url = match provider {
        Provider::OpenAi => config.openai_base_url.as_deref(),
        Provider::DeepSeek => config.deepseek_base_url.as_deref(),
        Provider::Moonshot => config.moonshot_base_url.as_deref(),
        Provider::Anthropic => {
            unreachable!("resolve_base_url called for Anthropic, which uses CLI not HTTP")
        }
    };

    if let Some(url_str) = config_url {
        return reqwest::Url::parse(url_str)
            .map_err(|_| AppError::Validation(format!("Invalid base URL: {url_str}")));
    }

    if let Some(env_var) = provider.base_url_env_var() {
        if let Ok(url_str) = std::env::var(env_var) {
            return reqwest::Url::parse(&url_str).map_err(|_| {
                AppError::Validation(format!("Invalid base URL from {env_var}: {url_str}"))
            });
        }
    }

    let default = provider
        .default_base_url()
        .expect("non-Anthropic providers always have a default base URL");
    reqwest::Url::parse(default)
        .map_err(|_| AppError::Validation(format!("Invalid default base URL: {default}")))
}
