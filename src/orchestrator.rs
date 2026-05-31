use std::time::Duration;

use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use crate::app::AppEvent;
use crate::config::Config;
use crate::error::AppError;
use crate::prompts::{self, WRITER_SYSTEM_PROMPT};
use crate::providers::{detect_provider, ChatRequest, LlmClient, Provider};
use crate::state::SharedState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Pause between Writer loop cycles to avoid tight spinning on the LLM API.
const CYCLE_SLEEP_MS: u64 = 500;

// ---------------------------------------------------------------------------
// Orchestrator entry point
// ---------------------------------------------------------------------------

/// Launch the Writer and Critic spectacle loops concurrently.
///
/// Both loops share the same [`SharedState`] and [`CancellationToken`]. The
/// Writer continuously revises the document while the Critic independently
/// inspects the latest version and produces entertainment-only commentary.
///
/// # Lifecycle
///
/// 1. Load and validate the input document.
/// 2. Put initial content into `SharedState` and send a `WriterOutput` event.
/// 3. Build LLM clients for both the Writer and Critic models.
/// 4. Spawn both loops as independent `tokio::spawn`ed tasks.
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
    let critic_client = build_critic_client(&config)?;

    // Writer task
    {
        let writer_event_tx = event_tx.clone();
        let writer_state = state.clone();
        let writer_cancel = cancel_token.clone();
        let writer_config = config.clone();

        tokio::spawn(async move {
            writer_loop(writer_client, writer_state, writer_event_tx, writer_cancel, writer_config)
                .await;
        });
    }

    // Critic task
    {
        let critic_event_tx = event_tx.clone();
        let critic_state = state.clone();
        let critic_cancel = cancel_token.clone();
        let critic_config = config;

        tokio::spawn(async move {
            critic_loop(critic_client, critic_state, critic_event_tx, critic_cancel, critic_config)
                .await;
        });
    }

    cancel_token.cancelled().await;

    let _ = event_tx.send(AppEvent::Shutdown);

    Ok(())
}

// ---------------------------------------------------------------------------
// Writer loop (runs in a spawned task)
// ---------------------------------------------------------------------------

/// Core Writer loop — continuously revises the document with LLM calls.
///
/// Each cycle:
/// 1. Snapshot the current document content.
/// 2. Read the latest critique from [`SharedState`]. If unavailable (the Critic
///    has not completed a cycle yet), skip critique context entirely.
/// 3. Check version relevance of the critique vs. current document version.
///    Log a warning on mismatch but still incorporate the feedback.
/// 4. Build the prompt: current document + critique (if available).
/// 5. Call the Writer LLM, update shared state, emit events.
///
/// Errors are reported via `AppEvent::Error` but the loop continues.
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

        let (current_version, current_content) = state.snapshot();

        let critique_context = match state.read_critique() {
            Some((critique_version, critique_text)) => {
                if critique_version != current_version {
                    eprintln!(
                        "Writer: critique version {critique_version} is stale (document at {current_version}), applying anyway",
                    );
                }
                format!("\n\nThe Critic said: {}\n\nNow revise the document accordingly.", critique_text)
            }
            None => {
                // No critique available yet — skip critique context entirely.
                String::new()
            }
        };

        let prompt_text = format!("{current_content}{critique_context}");

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
// Critic loop (runs in a spawned task)
// ---------------------------------------------------------------------------

/// Core Critic loop — continuously critiques the current document version.
///
/// Each cycle:
/// 1. Snapshot the current document version + content from [`SharedState`].
/// 2. Build a Critic prompt via [`prompts::critics`] using `config.critic_style`.
/// 3. Call the Critic LLM (cheap model recommended, e.g. deepseek-chat).
/// 4. Write the critique into `SharedState` via `write_critique`.
/// 5. Emit `CriticOutput` and `CritiqueReady` events to the TUI.
/// 6. Sleep 500ms before the next cycle.
///
/// The Critic never produces constructive feedback, rewrites, or scoring —
/// its output is pure entertainment. The document version captured at the
/// start of each cycle is stored alongside the critique text so the Writer
/// can detect stale feedback.
///
/// # Cancellation
///
/// Same `tokio::select!` pattern as the Writer loop — responds within ~500ms.
async fn critic_loop(
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

        let (doc_version, doc_content) = state.snapshot();

        let critic_system = prompts::critics(config.critic_style).to_string();

        let request = ChatRequest {
            system: critic_system,
            user: doc_content,
            model: config.critic_model.clone(),
            temperature: config.temperature as f32,
            max_tokens: config.max_tokens,
        };

        match client.complete(request, config.timeout_secs).await {
            Ok(reply) => {
                if let Some(cost) = reply.cost_usd {
                    total_cost_usd += cost;
                    eprintln!(
                        "Critic: cycle cost ${cost:.6}, total ${total_cost_usd:.6}",
                    );
                }

                state.write_critique(doc_version, reply.text.clone());

                let _ = event_tx.send(AppEvent::CriticOutput(reply.text));
                let _ = event_tx.send(AppEvent::CritiqueReady(doc_version));
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
fn build_llm_client(config: &Config) -> Result<LlmClient, AppError> {
    build_client(&config.writer_model, "Writer", config)
}

/// Build an [`LlmClient`] for the Critic model based on the resolved configuration.
fn build_critic_client(config: &Config) -> Result<LlmClient, AppError> {
    build_client(&config.critic_model, "Critic", config)
}

/// Shared client builder for the Writer and Critic.
///
/// Detects the provider from the model name, reads the API key from the
/// environment, resolves the base URL (config override → env var → default),
/// and constructs the appropriate client variant.
fn build_client(
    model: &str,
    agent_name: &str,
    config: &Config,
) -> Result<LlmClient, AppError> {
    let (provider, model_name) = detect_provider(model, agent_name)?;

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
