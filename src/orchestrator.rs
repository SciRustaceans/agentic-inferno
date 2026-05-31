use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use crate::app::AppEvent;
use crate::config::{Config, InfernoTask, RuntimeSettings};
use crate::error::AppError;
use crate::guards::{semantic_hash, CostCeiling, LoopDetection};
use crate::prompts::{self, APOLOGY_SYSTEM_PROMPT};
use crate::providers::{detect_provider, resolve_claude_bin, ChatRequest, LlmClient, Provider};
use crate::state::{ApologyCooldown, SharedState};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Pause between Writer loop cycles to avoid tight spinning on the LLM API.
const CYCLE_SLEEP_MS: u64 = 500;

/// Extra dwell after a reply has finished typing out, in milliseconds, so the
/// reader gets a beat to absorb the fully-revealed text before the next reply.
const DWELL_MS: u64 = 800;

/// How long an agent loop should wait before its next API call so the current
/// reply has had time to fully type out in the TUI, plus a short dwell.
///
/// = `(text_chars / reveal_cps) seconds + DWELL_MS`, floored at
/// [`CYCLE_SLEEP_MS`]. `reveal_cps` is clamped to at least 1 so a zero rate
/// never divides by zero.
///
/// Pure function of its inputs — unit-tested.
fn reveal_dwell_ms(text_chars: usize, reveal_cps: u32) -> u64 {
    let type_ms = text_chars as u64 * 1000 / reveal_cps.max(1) as u64;
    (type_ms + DWELL_MS).max(CYCLE_SLEEP_MS)
}

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

/// Resolve the token count for a call: use the provider-reported value when
/// present, otherwise estimate from the request text plus the reply text.
///
/// `user_est` is the pre-computed `estimate_tokens(request_user)` (computed
/// before the request is moved), so the estimate stays consistent with the
/// context-window heuristic.
fn resolve_call_tokens(reply_tokens: Option<u64>, user_est: u64, reply_text: &str) -> u64 {
    reply_tokens.unwrap_or(user_est + estimate_tokens(reply_text) as u64)
}

// ---------------------------------------------------------------------------
// Token accounting
// ---------------------------------------------------------------------------

/// Per-agent cumulative token totals, shared across the Writer, Critic, and
/// Apology tasks via `Arc<Mutex<TokenTotals>>`.
///
/// Unlike cost (which has a shared ceiling tracking the true total), token
/// totals have no shared running sum, so each agent must update only its own
/// field and read all three at emit time to send the true cumulative figures.
#[derive(Debug, Default, Clone, Copy)]
pub struct TokenTotals {
    pub writer: u64,
    pub critic: u64,
    pub apology: u64,
}

impl TokenTotals {
    /// Sum of all per-agent totals.
    pub fn total(&self) -> u64 {
        self.writer + self.critic + self.apology
    }
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

/// Launch the Writer and Critic loops concurrently.
///
/// Both loops share the same [`SharedState`] and [`CancellationToken`]. The
/// Writer continuously revises the document while the Critic independently
/// reads the latest version and produces commentary.
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
    runtime: Arc<RwLock<RuntimeSettings>>,
    state: SharedState,
    event_tx: UnboundedSender<AppEvent>,
    cancel_token: CancellationToken,
) -> Result<(), AppError> {
    // Seed the shared document. Non-prompt tasks read the input file; prompt
    // mode has no input file, so the document starts empty and evolves as the
    // Writer attempts the prompt.
    let initial_content = if config.task == InfernoTask::Prompt {
        String::new()
    } else {
        std::fs::read_to_string(&config.input)?
    };

    state.update(initial_content.clone());
    let _ = event_tx.send(AppEvent::WriterOutput(initial_content));

    let writer_client = build_llm_client(&config)?;
    let critic_client = build_critic_client(&config)?;

    // ── Cost ceiling guard ───────────────────────────────────────────
    // Shared across Writer, Critic, and Apology loops. Each successful
    // LLM call records its cost. If the ceiling is exceeded the token is
    // cancelled and the loops stop.

    let cost_ceiling = Arc::new(CostCeiling::new(config.max_cost_usd));
    let writer_cost: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));
    let critic_cost: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));
    let _apology_cost: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));

    // ── Token accounting ─────────────────────────────────────────────
    // One shared accumulator. Each agent updates its own field and reads all
    // three at emit time so every `TokenUsage` event carries the true totals.
    let token_totals: Arc<Mutex<TokenTotals>> = Arc::new(Mutex::new(TokenTotals::default()));

    // ── Spawn concurrent Writer + Critic loops ───────────────────────
    // JoinHandles are retained so we can await graceful shutdown.

    // Per-run loop detection: window=5, min_repeats=3.
    let loop_detector = Arc::new(Mutex::new(LoopDetection::new(5, 3)));

    let writer_handle = {
        let writer_event_tx = event_tx.clone();
        let writer_state = state.clone();
        let writer_cancel = cancel_token.clone();
        let writer_config = config.clone();
        let writer_cc = Arc::clone(&cost_ceiling);
        let writer_ac = Arc::clone(&writer_cost);
        let writer_ld = Arc::clone(&loop_detector);
        let writer_tt = Arc::clone(&token_totals);
        let writer_runtime = Arc::clone(&runtime);

        tokio::spawn(async move {
            writer_loop(
                writer_client,
                writer_state,
                writer_event_tx,
                writer_cancel,
                writer_config,
                writer_runtime,
                writer_cc,
                writer_ac,
                writer_ld,
                writer_tt,
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
        let critic_tt = Arc::clone(&token_totals);
        let critic_runtime = Arc::clone(&runtime);

        tokio::spawn(async move {
            critic_loop(
                critic_client,
                critic_state,
                critic_event_tx,
                critic_cancel,
                critic_config,
                critic_runtime,
                critic_cc,
                critic_ac,
                critic_tt,
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

    let join_result = tokio::time::timeout(Duration::from_secs(SHUTDOWN_GRACE_SECS), async {
        let writer_result = writer_handle.await;
        let critic_result = critic_handle.await;
        (writer_result, critic_result)
    })
    .await;

    match join_result {
        Ok((Ok(()), Ok(()))) => {
            // Both loops exited cleanly — normal fast path.
        }
        Ok((Err(writer_join_err), _)) => {
            eprintln!("Orchestrator: Writer loop panicked during shutdown: {writer_join_err}");
        }
        Ok((_, Err(critic_join_err))) => {
            eprintln!("Orchestrator: Critic loop panicked during shutdown: {critic_join_err}");
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
pub fn cooldown_remaining_secs(cooldown: &ApologyCooldown) -> Option<u64> {
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
/// Add `delta` tokens to the agent's slot in the shared accumulator and emit a
/// `TokenUsage` event carrying the true cumulative totals for all three agents.
///
/// `which` selects the slot: `0` writer, `1` critic, `2` apology.
fn record_and_emit_tokens(
    totals: &Arc<Mutex<TokenTotals>>,
    event_tx: &UnboundedSender<AppEvent>,
    which: u8,
    delta: u64,
) {
    let snapshot = {
        let mut t = totals.lock().expect("token_totals mutex poisoned");
        match which {
            0 => t.writer += delta,
            1 => t.critic += delta,
            _ => t.apology += delta,
        }
        *t
    };
    let _ = event_tx.send(AppEvent::TokenUsage {
        writer: snapshot.writer,
        critic: snapshot.critic,
        apology: snapshot.apology,
        total: snapshot.total(),
    });
}

#[allow(clippy::too_many_arguments)]
async fn writer_loop(
    mut client: LlmClient,
    state: SharedState,
    event_tx: UnboundedSender<AppEvent>,
    cancel_token: CancellationToken,
    config: Config,
    runtime: Arc<RwLock<RuntimeSettings>>,
    cost_ceiling: Arc<CostCeiling>,
    writer_cost: Arc<Mutex<f64>>,
    loop_detector: Arc<Mutex<LoopDetection>>,
    token_totals: Arc<Mutex<TokenTotals>>,
) {
    // ── Context window management ─────────────────────────────────────────
    // `task` is fixed for the run, so the system prompt and model context
    // window are computed once. The model can change live, but model-window
    // staleness after a switch is out of scope.
    let model_context_window = estimate_model_context_window(&config.writer_model);
    let writer_system_prompt = prompts::writer_system(config.task);
    let mut critique_history: VecDeque<(String, usize)> = VecDeque::new();
    // Track the model the current client was built with so a live change
    // triggers a rebuild.
    let mut current_model = config.writer_model.clone();

    loop {
        if cancel_token.is_cancelled() {
            break;
        }

        // ── Re-read live settings (copy out, drop the guard before .await) ──
        let (want_model, speed, prompt, cap) = {
            let s = runtime.read().expect("runtime settings RwLock poisoned");
            (
                s.writer_model.clone(),
                s.speed,
                s.prompt.clone(),
                s.max_cost_usd,
            )
        };
        let reveal_cps = speed.cps();
        cost_ceiling.set_limit(cap);

        if want_model != current_model {
            match build_client(&want_model, "Writer", &config) {
                Ok(c) => {
                    client = c;
                    current_model = want_model;
                }
                Err(e) => {
                    let _ = event_tx.send(AppEvent::Error(e));
                }
            }
        }

        // Char count of the most recent reply, used to pace the next cycle so
        // the reply has time to type out in the TUI. 0 on error / first iter →
        // the dwell floors at CYCLE_SLEEP_MS.
        let mut last_reply_chars = 0usize;

        let (_current_version, current_content) = state.snapshot();

        // ── Collect latest critique into history ──────────────────────────
        if let Some((_critique_version, critique_text)) = state.read_critique() {
            // A stale critique (its version differs from the current document)
            // is still incorporated — the Writer keeps moving forward.
            //
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
        let system_tokens = estimate_tokens(writer_system_prompt);
        let doc_tokens = estimate_tokens(&current_content);
        let crit_tokens_total: usize = critique_history.iter().map(|(_, t)| t).sum();
        let current_estimate = system_tokens + doc_tokens + crit_tokens_total;

        let pct = current_estimate as f64 / model_context_window as f64;
        if pct > 0.90 {
            // At 90% the oldest critique is pruned to keep the prompt within the
            // model's context window.
            let _ = critique_history.pop_front();
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

        // In prompt mode the user message frames the prompt as the goal and the
        // shared document as the evolving attempt. Other tasks pass the document
        // itself as the thing being revised. The prompt text is re-read live from
        // settings each cycle.
        let prompt_text = match (config.task, prompt.as_deref()) {
            (InfernoTask::Prompt, Some(goal)) => format!(
                "Task: {goal}\n\nCurrent attempt:\n{current_content}\n\nKeep working on it.{critique_context}"
            ),
            _ => format!("{current_content}{critique_context}"),
        };

        // Pre-compute the user-side token estimate before `prompt_text` is
        // moved into the request — used for the fallback when the API omits
        // usage data.
        let user_est = estimate_tokens(&prompt_text) as u64;

        let request = ChatRequest {
            system: writer_system_prompt.to_string(),
            user: prompt_text,
            model: current_model.clone(),
            temperature: config.temperature as f32,
            max_tokens: config.max_tokens,
        };

        match client.complete(request, config.timeout_secs).await {
            Ok(reply) => {
                // Record token usage unconditionally (cost may be absent, e.g.
                // OpenAI omits `total_cost_usd`, but token usage is the meter's
                // whole point). Compute before `reply.text` is moved.
                let call_tokens = resolve_call_tokens(reply.tokens, user_est, &reply.text);
                record_and_emit_tokens(&token_totals, &event_tx, 0, call_tokens);

                // Capture the reply length (before `reply.text` is moved) so
                // the cycle pause waits for it to finish typing out.
                last_reply_chars = reply.text.chars().count();

                if let Some(cost) = reply.cost_usd {
                    // Track per-agent cost (only on success).
                    *writer_cost.lock().expect("writer_cost mutex poisoned") += cost;

                    // Enforce cost ceiling. If exceeded, cancel and exit.
                    if let Err(AppError::CostCeilingExceeded(spent, limit)) =
                        cost_ceiling.record(cost)
                    {
                        let _ = event_tx
                            .send(AppEvent::Error(AppError::CostCeilingExceeded(spent, limit)));
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
                // always goes to the Writer pane.
                let new_version = if let Some(marker_idx) = find_apology_marker(&reply.text) {
                    let document_text = reply.text[..marker_idx].trim().to_string();

                    // Semantic loop detection on the document content.
                    let text_hash = semantic_hash(&document_text);
                    if loop_detector
                        .lock()
                        .expect("LoopDetection mutex poisoned")
                        .check(text_hash)
                        .is_err()
                    {
                        let _ = event_tx.send(AppEvent::LoopExhausted);
                        cancel_token.cancel();
                        break;
                    }

                    let new_ver = state.update(document_text);

                    // The full text (including the marker) goes to the TUI so
                    // the apology is shown alongside the revision.
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
                        let apology_tt = Arc::clone(&token_totals);
                        // Use the Writer's currently-active model (which may have
                        // been switched live) for the apology call.
                        let apology_model = current_model.clone();

                        tokio::spawn(async move {
                            match build_client(&apology_model, "Apology", &apology_config) {
                                Ok(client) => {
                                    let critique_text = apology_state
                                        .read_critique()
                                        .map(|(_, text)| text)
                                        .unwrap_or_default();

                                    let apology_user_est = estimate_tokens(&critique_text) as u64;

                                    let request = ChatRequest {
                                        system: APOLOGY_SYSTEM_PROMPT.to_string(),
                                        user: critique_text,
                                        model: apology_model.clone(),
                                        temperature: apology_config.temperature as f32,
                                        max_tokens: apology_config.max_tokens,
                                    };

                                    match client
                                        .complete(request, apology_config.timeout_secs)
                                        .await
                                    {
                                        Ok(reply) => {
                                            let call_tokens = resolve_call_tokens(
                                                reply.tokens,
                                                apology_user_est,
                                                &reply.text,
                                            );
                                            record_and_emit_tokens(
                                                &apology_tt,
                                                &apology_event_tx,
                                                2,
                                                call_tokens,
                                            );
                                            if let Some(cost) = reply.cost_usd {
                                                if let Err(err) = apology_ceiling.record(cost) {
                                                    let _ =
                                                        apology_event_tx.send(AppEvent::Error(err));
                                                    return;
                                                }
                                                let _ =
                                                    apology_event_tx.send(AppEvent::CostWarning {
                                                        spent: apology_ceiling.spent(),
                                                        limit: apology_ceiling.limit(),
                                                        writer_cost: 0.0,
                                                        critic_cost: 0.0,
                                                        apology_cost: cost,
                                                    });
                                            }
                                            let _ = apology_event_tx
                                                .send(AppEvent::ApologyReady(reply.text));
                                        }
                                        Err(err) => {
                                            let _ = apology_event_tx.send(AppEvent::Error(err));
                                        }
                                    }
                                }
                                Err(err) => {
                                    let _ = apology_event_tx.send(AppEvent::Error(err));
                                }
                            }
                        });
                    }

                    new_ver
                } else {
                    // Semantic loop detection on the document content.
                    let text_hash = semantic_hash(&reply.text);
                    if loop_detector
                        .lock()
                        .expect("LoopDetection mutex poisoned")
                        .check(text_hash)
                        .is_err()
                    {
                        let _ = event_tx.send(AppEvent::LoopExhausted);
                        cancel_token.cancel();
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
            _ = tokio::time::sleep(Duration::from_millis(reveal_dwell_ms(last_reply_chars, reveal_cps))) => {}
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
#[allow(clippy::too_many_arguments)]
async fn critic_loop(
    mut client: LlmClient,
    state: SharedState,
    event_tx: UnboundedSender<AppEvent>,
    cancel_token: CancellationToken,
    config: Config,
    runtime: Arc<RwLock<RuntimeSettings>>,
    cost_ceiling: Arc<CostCeiling>,
    critic_cost: Arc<Mutex<f64>>,
    token_totals: Arc<Mutex<TokenTotals>>,
) {
    // Track the model the current client was built with so a live change
    // triggers a rebuild.
    let mut current_model = config.critic_model.clone();

    loop {
        if cancel_token.is_cancelled() {
            break;
        }

        // ── Re-read live settings (copy out, drop the guard before .await) ──
        let (want_model, style, speed, cap) = {
            let s = runtime.read().expect("runtime settings RwLock poisoned");
            (
                s.critic_model.clone(),
                s.critic_style,
                s.speed,
                s.max_cost_usd,
            )
        };
        let reveal_cps = speed.cps();
        cost_ceiling.set_limit(cap);

        if want_model != current_model {
            match build_client(&want_model, "Critic", &config) {
                Ok(c) => {
                    client = c;
                    current_model = want_model;
                }
                Err(e) => {
                    let _ = event_tx.send(AppEvent::Error(e));
                }
            }
        }

        // Char count of the most recent reply, used to pace the next cycle so
        // the critique has time to type out in the TUI. 0 on error / first
        // iter → the dwell floors at CYCLE_SLEEP_MS.
        let mut last_reply_chars = 0usize;

        let (doc_version, doc_content) = state.snapshot();

        let critic_system = prompts::critics(style).to_string();

        // Pre-compute the user-side estimate before `doc_content` is moved.
        let user_est = estimate_tokens(&doc_content) as u64;

        let request = ChatRequest {
            system: critic_system,
            user: doc_content,
            model: current_model.clone(),
            temperature: config.temperature as f32,
            max_tokens: config.max_tokens,
        };

        match client.complete(request, config.timeout_secs).await {
            Ok(reply) => {
                // Record token usage unconditionally (before `reply.text` moves).
                let call_tokens = resolve_call_tokens(reply.tokens, user_est, &reply.text);
                record_and_emit_tokens(&token_totals, &event_tx, 1, call_tokens);

                // Capture the reply length (before `reply.text` is moved) so
                // the cycle pause waits for it to finish typing out.
                last_reply_chars = reply.text.chars().count();

                if let Some(cost) = reply.cost_usd {
                    // Track per-agent cost (only on success).
                    *critic_cost.lock().expect("critic_cost mutex poisoned") += cost;

                    // Enforce cost ceiling. If exceeded, cancel and exit.
                    if let Err(AppError::CostCeilingExceeded(spent, limit)) =
                        cost_ceiling.record(cost)
                    {
                        let _ = event_tx
                            .send(AppEvent::Error(AppError::CostCeilingExceeded(spent, limit)));
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
                }

                state.write_critique(doc_version, reply.text.clone());

                // Emit CritiqueReady first so `app.critic_version` is current
                // when `apply_critic_output` builds the `── vN ──` header.
                let _ = event_tx.send(AppEvent::CritiqueReady(doc_version));
                let _ = event_tx.send(AppEvent::CriticOutput(reply.text));

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
            _ = tokio::time::sleep(Duration::from_millis(reveal_dwell_ms(last_reply_chars, reveal_cps))) => {}
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
pub(crate) fn build_client(
    model: &str,
    agent_name: &str,
    config: &Config,
) -> Result<LlmClient, AppError> {
    let (provider, model_name) = detect_provider(model, agent_name)?;

    match provider {
        Provider::Anthropic => Ok(LlmClient::AnthropicCli {
            model: model_name,
            claude_bin: resolve_claude_bin(),
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

/// Validate that a model string can be turned into a working [`LlmClient`].
///
/// Reuses [`build_client`] so it honors every per-provider rule (e.g. the
/// Anthropic CLI path needs no API key, while OpenAI-compatible providers do).
/// The built client is discarded — only success/failure matters. Intended for
/// a settings menu to pre-validate a model change before applying it.
pub(crate) fn validate_model(
    model: &str,
    agent_name: &str,
    config: &Config,
) -> Result<(), AppError> {
    build_client(model, agent_name, config).map(|_| ())
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
pub fn find_apology_marker(text: &str) -> Option<usize> {
    let marker_lower = "[apology]";
    let text_lower = text.to_ascii_lowercase();
    text_lower.find(marker_lower)
}

/// Count how many distinct harsh keywords appear in `text` (case-insensitive).
///
/// Harsh keywords: "incompetent", "worthless", "pathetic", "garbage",
/// "useless", "hopeless", "embarrassing", "disgrace".
pub fn count_harsh_keywords(text: &str) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CliArgs, Config};
    use std::sync::Mutex as StdMutex;

    /// Serialise the env-var window so parallel tests don't clobber the key.
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    /// Build a minimal validated `Config` for in-crate orchestrator tests.
    /// Creates a temp input file outside any repo so the leak guard passes and
    /// sets a dummy `DEEPSEEK_API_KEY` for the duration of the build.
    ///
    /// Callers must already hold `ENV_LOCK` — this helper does not acquire it
    /// (the `std::sync::Mutex` is non-reentrant, so a second lock on the same
    /// thread would deadlock).
    fn test_config() -> Config {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let input_path = tmp.path().join("input.txt");
        std::fs::write(&input_path, "content").expect("write input");

        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY", "sk-test-fake-key");
        }

        let cli = CliArgs {
            writer_model: "deepseek-reasoner".into(),
            critic_model: Some("deepseek-chat".into()),
            input: Some(input_path),
            task: None,
            prompt: None,
            max_cost_usd: Some(1.0),
            temperature: Some(0.8),
            max_tokens: Some(256),
            timeout_secs: Some(10),
            config: None,
            critic_style: None,
            speed: None,
            openai_base_url: None,
            deepseek_base_url: None,
            moonshot_base_url: None,
        };

        let config = Config::build(cli, None).expect("config build");

        unsafe {
            std::env::remove_var("DEEPSEEK_API_KEY");
        }
        let _ = tmp;
        config
    }

    #[test]
    fn validate_model_accepts_known_model_with_key() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = test_config();
        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY", "sk-test-fake-key");
        }
        let ok = validate_model("deepseek-chat", "Critic", &config);
        unsafe {
            std::env::remove_var("DEEPSEEK_API_KEY");
        }
        assert!(ok.is_ok(), "deepseek-chat with key should validate: {ok:?}");
    }

    #[test]
    fn validate_model_rejects_unknown_model() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = test_config();
        let err = validate_model("nonexistent-model-xyz", "Writer", &config);
        assert!(
            matches!(err, Err(AppError::UnknownModel(_, _))),
            "unknown model should fail with UnknownModel, got {err:?}"
        );
    }

    #[test]
    fn reveal_dwell_ms_grows_with_length() {
        // Longer replies take longer to type out, so the dwell grows.
        let short = reveal_dwell_ms(100, 40);
        let long = reveal_dwell_ms(2000, 40);
        assert!(long > short, "dwell must grow with reply length");
        // 2000 chars / 40 cps = 50s = 50_000ms, + 800 dwell.
        assert_eq!(long, 2000 * 1000 / 40 + DWELL_MS);
    }

    #[test]
    fn reveal_dwell_ms_floors_at_cycle_sleep() {
        // A zero-length reply still waits at least CYCLE_SLEEP_MS.
        assert!(reveal_dwell_ms(0, 40) >= CYCLE_SLEEP_MS);
        // And a faster speed never drops below the floor either.
        assert!(reveal_dwell_ms(0, 80) >= CYCLE_SLEEP_MS);
    }

    #[test]
    fn reveal_dwell_ms_handles_zero_cps_safely() {
        // cps=0 must not divide by zero; clamps to 1 cps internally.
        let v = reveal_dwell_ms(40, 0);
        assert!(v >= CYCLE_SLEEP_MS);
        // 40 chars / 1 cps = 40_000ms + DWELL_MS.
        assert_eq!(v, 40 * 1000 + DWELL_MS);
    }

    #[test]
    fn reveal_dwell_ms_faster_speed_is_shorter() {
        // The same reply types out faster at a higher cps, so the dwell is
        // shorter (down to the floor).
        let slow = reveal_dwell_ms(4000, 20);
        let fast = reveal_dwell_ms(4000, 80);
        assert!(fast < slow, "higher cps should shorten the dwell");
    }

    #[test]
    fn resolve_call_tokens_uses_provider_value_when_present() {
        // When the provider reports tokens, use them verbatim and ignore the
        // estimate.
        let n = resolve_call_tokens(Some(1234), 999, "this text is ignored");
        assert_eq!(n, 1234);
    }

    #[test]
    fn resolve_call_tokens_falls_back_to_estimate() {
        // 40 chars / 4 = 10 reply tokens; plus the user estimate.
        let reply = "x".repeat(40);
        let n = resolve_call_tokens(None, 7, &reply);
        assert_eq!(n, 7 + estimate_tokens(&reply) as u64);
        assert_eq!(n, 17);
    }

    #[test]
    fn token_totals_sum_is_correct() {
        let t = TokenTotals {
            writer: 100,
            critic: 50,
            apology: 25,
        };
        assert_eq!(t.total(), 175);
    }

    #[test]
    fn record_and_emit_tokens_carries_true_cumulative_totals() {
        // Locks the no-flicker contract: updating one agent's slot leaves the
        // others intact, and each emit carries the true cumulative totals + sum.
        let totals = Arc::new(Mutex::new(TokenTotals::default()));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        record_and_emit_tokens(&totals, &tx, 0, 100); // writer += 100
        record_and_emit_tokens(&totals, &tx, 1, 50); // critic += 50
        record_and_emit_tokens(&totals, &tx, 2, 25); // apology += 25
        record_and_emit_tokens(&totals, &tx, 0, 10); // writer += 10 → 110

        // The final accumulator state reflects all four updates.
        let snapshot = *totals.lock().unwrap();
        assert_eq!(snapshot.writer, 110);
        assert_eq!(snapshot.critic, 50);
        assert_eq!(snapshot.apology, 25);

        // The last event must carry the true cumulative totals, not just the
        // one agent that changed.
        let mut last = None;
        while let Ok(ev) = rx.try_recv() {
            last = Some(ev);
        }
        match last.expect("at least one TokenUsage event") {
            AppEvent::TokenUsage {
                writer,
                critic,
                apology,
                total,
            } => {
                assert_eq!(writer, 110);
                assert_eq!(critic, 50);
                assert_eq!(apology, 25);
                assert_eq!(total, 185);
            }
            other => panic!("expected TokenUsage, got {other:?}"),
        }
    }
}
