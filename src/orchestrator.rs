use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use crate::app::AppEvent;
use crate::config::Config;
use crate::error::AppError;
use crate::guards::{semantic_hash, CostCeiling, LoopDetection};
use crate::prompts::{self, APOLOGY_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT};
use crate::providers::{detect_provider, ChatRequest, LlmClient, Provider};
use crate::state::{ApologyCooldown, SharedState};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Pause between Writer loop cycles to avoid tight spinning on the LLM API.
const CYCLE_SLEEP_MS: u64 = 500;

// ---------------------------------------------------------------------------
// Context window helpers
// ---------------------------------------------------------------------------

/// Rough token estimate: 1 token ≈ 4 characters for English text.
///
/// This is a heuristic, not an exact tokenizer. Good enough for context window
/// management — we only need to stay within ~10% of the model's limit.
fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

/// Estimate the model's context window size in tokens from its name.
///
/// Returns well-known limits for identified models and a conservative 128K
/// default for unrecognised names. Matches are case-insensitive substring
/// checks so that `"gpt-4o"`, `"gpt-4-turbo"`, etc. all resolve correctly.
fn estimate_model_context_window(model: &str) -> usize {
    let lowered = model.to_ascii_lowercase();
    if lowered.contains("claude") {
        200_000
    } else if lowered.contains("deepseek") {
        65_536
    } else {
        128_000
    }
}

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

    // ── Cost ceiling guard ───────────────────────────────────────────
    // Shared across Writer, Critic, and Apology loops. Each successful
    // LLM call records its cost. If the ceiling is exceeded the token is
    // cancelled and the spectacle stops.

    let cost_ceiling = Arc::new(CostCeiling::new(config.max_cost_usd));
    let writer_cost: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));
    let critic_cost: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));
    let _apology_cost: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));

    // ── Spawn concurrent Writer + Critic loops ───────────────────────
    // JoinHandles are retained so we can await graceful shutdown.

    // Per-spectacle loop detection: window=5, min_repeats=3.
    let loop_detector = Arc::new(Mutex::new(LoopDetection::new(5, 3)));

    let writer_handle = {
        let writer_event_tx = event_tx.clone();
        let writer_state = state.clone();
        let writer_cancel = cancel_token.clone();
        let writer_config = config.clone();
        let writer_cc = Arc::clone(&cost_ceiling);
        let writer_ac = Arc::clone(&writer_cost);
        let writer_ld = Arc::clone(&loop_detector);

        tokio::spawn(async move {
            writer_loop(
                writer_client,
                writer_state,
                writer_event_tx,
                writer_cancel,
                writer_config,
                writer_cc,
                writer_ac,
                writer_ld,
            )
            .await;
        })
    };

    let critic_handle = {
        let critic_event_tx = event_tx.clone();
        let critic_state = state.clone();
        let critic_cancel = cancel_token.clone();
        let critic_config = config;
        let critic_cc = Arc::clone(&cost_ceiling);
        let critic_ac = Arc::clone(&critic_cost);

        tokio::spawn(async move {
            critic_loop(
                critic_client,
                critic_state,
                critic_event_tx,
                critic_cancel,
                critic_config,
                critic_cc,
                critic_ac,
            )
            .await;
        })
    };

    // ── Wait for cancellation (Esc / q in TUI) ──────────────────────

    cancel_token.cancelled().await;

    // Notify TUI that shutdown has been initiated.
    let _ = event_tx.send(AppEvent::Shutdown);

    // ── 3-tier subprocess shutdown ──────────────────────────────────
    //
    // Tier 1 (graceful, up to 3s): Both loops have already seen the
    // cancelled token at their next checkpoint (`cancel_token.cancelled()`
    // select branch or top-of-loop `is_cancelled()` check).  We give
    // them 3 seconds to finish any in-progress LLM response processing
    // and exit cleanly.
    //
    // Tier 2 (escalation): If the loops haven't returned within 3s,
    // the timeout elapses.  Any in-flight `claude` subprocesses inside
    // `anthropic_complete()` are bounded by their own `timeout_secs`.
    // Once that expires, `child.kill().await` fires explicitly.
    // The orchestrator returns anyway — `kill_on_drop(true)` on every
    // subprocess `Command` provides the Tier 3 backstop when the
    // tokio runtime drops at process exit.
    //
    // Tier 3 (panic backstop): `kill_on_drop(true)` on every
    // `tokio::process::Command` ensures no orphan `claude` processes
    // survive a panic or early return.

    const SHUTDOWN_GRACE_SECS: u64 = 3;

    let join_result = tokio::time::timeout(
        Duration::from_secs(SHUTDOWN_GRACE_SECS),
        async {
            let writer_result = writer_handle.await;
            let critic_result = critic_handle.await;
            (writer_result, critic_result)
        },
    )
    .await;

    match join_result {
        Ok((Ok(()), Ok(()))) => {
            // Both loops exited cleanly — normal fast path.
        }
        Ok((Err(writer_join_err), _)) => {
            eprintln!(
                "Orchestrator: Writer loop panicked during shutdown: {writer_join_err}"
            );
        }
        Ok((_, Err(critic_join_err))) => {
            eprintln!(
                "Orchestrator: Critic loop panicked during shutdown: {critic_join_err}"
            );
        }
        Err(_elapsed) => {
            // Tier 2 escalation: loops didn't exit within the grace period.
            // In-flight LLM calls will be killed on timeout or by kill_on_drop.
            eprintln!(
                "Orchestrator: loops did not exit within {SHUTDOWN_GRACE_SECS}s grace period — escalating"
            );
        }
    }

    // Drop event_tx last — the TUI loop sees None from recv() after all
    // senders are dropped, which transitions to Done if it hasn't already.
    drop(event_tx);

    Ok(())
}

// ---------------------------------------------------------------------------
// Writer loop (runs in a spawned task)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Apology cooldown helpers
// ---------------------------------------------------------------------------

/// Return the remaining cooldown time in seconds, if cooldown is active.
///
/// Returns `Some(secs)` while the cooldown is in effect, `None` once both
/// the time and cycle conditions are satisfied (or no apology has occurred).
fn cooldown_remaining_secs(cooldown: &ApologyCooldown) -> Option<u64> {
    let last_time = cooldown.last_apology_time?;
    let elapsed_secs = last_time.elapsed().as_secs();

    if elapsed_secs < 30 {
        Some(30 - elapsed_secs)
    } else if cooldown.cycles_since_apology < 3 {
        // Time condition met, still waiting on cycles.
        Some(0)
    } else {
        None
    }
}

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
/// # Apology detection
///
/// After each successful LLM response, the reply is scanned for the `[APOLOGY]`
/// marker. If found and the cooldown has expired, a separate apology LLM call
/// is made and the result is emitted as `AppEvent::ApologyReady`. If the
/// cooldown is still active, the trigger is suppressed and logged.
///
/// # Cancellation
///
/// Uses a `tokio::select!` on the `cancel_token.cancelled()` future and a
/// short sleep so the loop responds to cancellation within ~500ms.
#[allow(clippy::too_many_arguments)]
async fn writer_loop(
    client: LlmClient,
    state: SharedState,
    event_tx: UnboundedSender<AppEvent>,
    cancel_token: CancellationToken,
    config: Config,
    cost_ceiling: Arc<CostCeiling>,
    writer_cost: Arc<Mutex<f64>>,
    loop_detector: Arc<Mutex<LoopDetection>>,
) {
    // ── Context window management ─────────────────────────────────────────
    let model_context_window = estimate_model_context_window(&config.writer_model);
    let mut critique_history: VecDeque<(String, usize)> = VecDeque::new();

    loop {
        if cancel_token.is_cancelled() {
            break;
        }

        let (current_version, current_content) = state.snapshot();

        // ── Collect latest critique into history ──────────────────────────
        if let Some((critique_version, critique_text)) = state.read_critique() {
            if critique_version != current_version {
                eprintln!(
                    "Writer: critique version {critique_version} is stale (document at {current_version}), applying anyway",
                );
            }
            // Only add to history if it's actually new text (avoid duplicating
            // the same critique across fast Writer cycles).
            let is_new = critique_history
                .back()
                .map(|(text, _)| text != &critique_text)
                .unwrap_or(true);
            if is_new {
                let crit_tokens = estimate_tokens(&critique_text);
                critique_history.push_back((critique_text, crit_tokens));
            }
        }

        // ── Context window check — warn at 80%, prune oldest at 90% ──────
        let system_tokens = estimate_tokens(WRITER_SYSTEM_PROMPT);
        let doc_tokens = estimate_tokens(&current_content);
        let crit_tokens_total: usize = critique_history.iter().map(|(_, t)| t).sum();
        let current_estimate = system_tokens + doc_tokens + crit_tokens_total;

        let pct = current_estimate as f64 / model_context_window as f64;
        if pct > 0.90 {
            if let Some((_dropped_text, dropped_tokens)) = critique_history.pop_front() {
                eprintln!(
                    "Writer: context window at {pct:.1}% ({current_estimate}/{model_context_window}) \
                     — dropping oldest critique ({dropped_tokens} tokens)",
                );
            }
        } else if pct > 0.80 {
            eprintln!(
                "Writer: context window at {pct:.1}% ({current_estimate}/{model_context_window})",
            );
        }

        // ── Build prompt with full critique history ───────────────────────
        let mut critique_context = String::new();
        if !critique_history.is_empty() {
            critique_context.push_str("\n\nThe Critic has said:\n");
            for (i, (text, _)) in critique_history.iter().enumerate() {
                critique_context.push_str(&format!("--- Critique {} ---\n{}\n", i + 1, text));
            }
            critique_context.push_str("Now revise the document accordingly.");
        }

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
                    // Track per-agent cost (only on success).
                    *writer_cost.lock().expect("writer_cost mutex poisoned") += cost;

                    eprintln!(
                        "Writer: cycle cost ${cost:.6}, writer total ${:.6}",
                        *writer_cost.lock().expect("writer_cost mutex poisoned"),
                    );

                    // Enforce cost ceiling. If exceeded, cancel and exit.
                    if let Err(AppError::CostCeilingExceeded(spent, limit)) =
                        cost_ceiling.record(cost)
                    {
                        let _ = event_tx.send(AppEvent::Error(AppError::CostCeilingExceeded(
                            spent, limit,
                        )));
                        cancel_token.cancel();
                        break;
                    }

                    // Send cost update to TUI.
                    let _ = event_tx.send(AppEvent::CostWarning {
                        spent: cost_ceiling.spent(),
                        limit: cost_ceiling.limit(),
                        writer_cost: *writer_cost.lock().expect("writer_cost mutex poisoned"),
                        critic_cost: 0.0, // will be updated by critic loop
                        apology_cost: 0.0,
                    });
                }

                // Scan for [APOLOGY] marker (case-insensitive).  If present,
                // split the response: the document part goes into shared state
                // and the apology text is sent to the apology bar via ApologyReady,
                // gated by the cooldown.  The full text (including the marker)
                // always goes to the Writer pane for audience spectacle.
                let new_version = if let Some(marker_idx) = find_apology_marker(&reply.text) {
                    let document_text = reply.text[..marker_idx].trim().to_string();

                    // Semantic loop detection on the document content.
                    let text_hash = semantic_hash(&document_text);
                    if let Err(err) = loop_detector
                        .lock()
                        .expect("LoopDetection mutex poisoned")
                        .check(text_hash)
                    {
                        let _ = event_tx.send(AppEvent::LoopExhausted);
                        cancel_token.cancel();
                        eprintln!("Writer: loop exhausted — {err}");
                        break;
                    }

                    let new_ver = state.update(document_text);

                    // The full text (including the marker) goes to the TUI for
                    // display — the audience should see the theatrical apology.
                    let _ = event_tx.send(AppEvent::WriterOutput(reply.text));
                    let _ = event_tx.send(AppEvent::WriterDone(new_ver));
                    let _ = event_tx.send(AppEvent::ApologyTriggered);

                    // ── Non-blocking apology LLM call ──────────────────
                    //
                    // Spawn a separate LLM call using the writer model with
                    // APOLOGY_SYSTEM_PROMPT and the latest critique as user
                    // prompt. Writer and Critic loops continue immediately.
                    {
                        let apology_state = state.clone();
                        let apology_event_tx = event_tx.clone();
                        let apology_config = config.clone();
                        let apology_ceiling = Arc::clone(&cost_ceiling);

                        tokio::spawn(async move {
                            match build_client(
                                &apology_config.writer_model,
                                "Apology",
                                &apology_config,
                            ) {
                                Ok(client) => {
                                    let critique_text = apology_state
                                        .read_critique()
                                        .map(|(_, text)| text)
                                        .unwrap_or_default();

                                    let request = ChatRequest {
                                        system: APOLOGY_SYSTEM_PROMPT.to_string(),
                                        user: critique_text,
                                        model: apology_config.writer_model.clone(),
                                        temperature: apology_config.temperature as f32,
                                        max_tokens: apology_config.max_tokens,
                                    };

                                    match client
                                        .complete(request, apology_config.timeout_secs)
                                        .await
                                    {
                                        Ok(reply) => {
                                            if let Some(cost) = reply.cost_usd {
                                                if let Err(err) = apology_ceiling.record(cost) {
                                                    let _ = apology_event_tx
                                                        .send(AppEvent::Error(err));
                                                    return;
                                                }
                                                let _ = apology_event_tx.send(
                                                    AppEvent::CostWarning {
                                                        spent: apology_ceiling.spent(),
                                                        limit: apology_ceiling.limit(),
                                                        writer_cost: 0.0,
                                                        critic_cost: 0.0,
                                                        apology_cost: cost,
                                                    },
                                                );
                                            }
                                            let _ = apology_event_tx
                                                .send(AppEvent::ApologyReady(reply.text));
                                        }
                                        Err(err) => {
                                            let _ = apology_event_tx
                                                .send(AppEvent::Error(err));
                                        }
                                    }
                                }
                                Err(err) => {
                                    let _ = apology_event_tx.send(AppEvent::Error(err));
                                }
                            }
                        });
                    }

                    eprintln!("Apology triggered: marker (apology follows marker in text)");
                    new_ver
                } else {
                    // Semantic loop detection on the document content.
                    let text_hash = semantic_hash(&reply.text);
                    if let Err(err) = loop_detector
                        .lock()
                        .expect("LoopDetection mutex poisoned")
                        .check(text_hash)
                    {
                        let _ = event_tx.send(AppEvent::LoopExhausted);
                        cancel_token.cancel();
                        eprintln!("Writer: loop exhausted — {err}");
                        break;
                    }

                    let new_ver = state.update(reply.text.clone());
                    let _ = event_tx.send(AppEvent::WriterOutput(reply.text));
                    let _ = event_tx.send(AppEvent::WriterDone(new_ver));
                    new_ver
                };

                let _ = new_version;

                // Send cooldown status for the TUI status bar.
                let current_cooldown = state.read_apology_cooldown();
                let remaining = cooldown_remaining_secs(&current_cooldown);
                let _ = event_tx.send(AppEvent::ApologyCooldown(remaining));
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
    cost_ceiling: Arc<CostCeiling>,
    critic_cost: Arc<Mutex<f64>>,
) {
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
                    // Track per-agent cost (only on success).
                    *critic_cost.lock().expect("critic_cost mutex poisoned") += cost;

                    eprintln!(
                        "Critic: cycle cost ${cost:.6}, critic total ${:.6}",
                        *critic_cost.lock().expect("critic_cost mutex poisoned"),
                    );

                    // Enforce cost ceiling. If exceeded, cancel and exit.
                    if let Err(AppError::CostCeilingExceeded(spent, limit)) =
                        cost_ceiling.record(cost)
                    {
                        let _ = event_tx.send(AppEvent::Error(AppError::CostCeilingExceeded(
                            spent, limit,
                        )));
                        cancel_token.cancel();
                        break;
                    }

                    // Send cost update to TUI.
                    let _ = event_tx.send(AppEvent::CostWarning {
                        spent: cost_ceiling.spent(),
                        limit: cost_ceiling.limit(),
                        writer_cost: 0.0,
                        critic_cost: *critic_cost.lock().expect("critic_cost mutex poisoned"),
                        apology_cost: 0.0,
                    });
                }

                // Keyword-based harshness detection: if the critique contains
                // ≥3 harsh keywords, trigger the apology workflow regardless
                // of whether the Writer's output contained a marker.
                let kw_count = count_harsh_keywords(&reply.text);
                if kw_count >= 3 {
                    let _ = event_tx.send(AppEvent::ApologyTriggered);
                    eprintln!("Apology triggered: keywords ({kw_count})");
                }

                state.write_critique(doc_version, reply.text.clone());

                let _ = event_tx.send(AppEvent::CriticOutput(reply.text));
                let _ = event_tx.send(AppEvent::CritiqueReady(doc_version));

                // Increment apology cooldown cycle counter — each successful
                // critic cycle brings us closer to satisfying the 3-cycle minimum.
                state.increment_critique_cycles();
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

// ---------------------------------------------------------------------------
// Apology marker parser
// ---------------------------------------------------------------------------

/// Find a case-insensitive `[APOLOGY]` marker in `text`.
///
/// Returns the byte index of the opening `[` if the marker is found as a
/// complete token (`[APOLOGY]`, `[apology]`, `[Apology]`, etc.).  Partial
/// markers like `[APOL` without the closing `]` are NOT matched.
fn find_apology_marker(text: &str) -> Option<usize> {
    let marker_lower = "[apology]";
    let text_lower = text.to_ascii_lowercase();
    text_lower.find(marker_lower)
}

/// Count how many distinct harsh keywords appear in `text` (case-insensitive).
///
/// Harsh keywords: "incompetent", "worthless", "pathetic", "garbage",
/// "useless", "hopeless", "embarrassing", "disgrace".
fn count_harsh_keywords(text: &str) -> usize {
    const KEYWORDS: [&str; 8] = [
        "incompetent",
        "worthless",
        "pathetic",
        "garbage",
        "useless",
        "hopeless",
        "embarrassing",
        "disgrace",
    ];
    let lower = text.to_ascii_lowercase();
    KEYWORDS.iter().filter(|kw| lower.contains(*kw)).count()
}
