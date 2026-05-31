//! Integration tests for the orchestrator — concurrent agent behavior,
//! version stamping, stop mechanism, and error resilience.
//!
//! All tests use [`wiremock`] to simulate LLM API endpoints — no real API
//! keys or network calls are required. Each test constructs a minimal
//! [`Config`] pointing to a local mock server, spawns the spectacle via
//! [`run_spectacle`], and verifies behaviour through shared-state inspection
//! and event-channel collection.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::json;
use tokio::sync::mpsc::{self, UnboundedReceiver};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use agentic_inferno::app::AppEvent;
use agentic_inferno::config::{CliArgs, Config};
use agentic_inferno::orchestrator::run_spectacle;
use agentic_inferno::state::SharedState;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Scoped environment-variable setter.  Restores original values (or removes
/// the key) on drop, so test functions never leak env state to each other.
struct EnvGuard(Vec<(&'static str, Option<String>)>);

impl EnvGuard {
    fn new() -> Self {
        Self(Vec::new())
    }

    fn set(&mut self, key: &'static str, value: &str) {
        let old = std::env::var(key).ok();
        std::env::set_var(key, value);
        self.0.push((key, old));
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, val) in self.0.iter().rev() {
            match val {
                Some(old) => std::env::set_var(key, old),
                None => std::env::remove_var(key),
            }
        }
    }
}

/// Build a validated [`Config`] suitable for testing with a mock server.
///
/// The input file is created in a temp directory outside the repository so
/// the leak guard passes.  `deepseek_base_url` points to the wiremock URI.
fn build_test_config(input_content: &str, mock_uri: &str) -> (tempfile::TempDir, Config) {
    let tmp = tempfile::TempDir::new().expect("failed to create temp dir");
    let input_path = tmp.path().join("test_input.txt");
    std::fs::write(&input_path, input_content).expect("failed to write test input");

    let cli = CliArgs {
        writer_model: "deepseek-reasoner".into(),
        critic_model: Some("deepseek-chat".into()),
        input: Some(input_path),
        task: None,
        prompt: None,
        max_cost_usd: Some(100.0),
        temperature: Some(0.8),
        max_tokens: Some(256),
        timeout_secs: Some(10),
        config: None,
        critic_style: None,
        openai_base_url: None,
        deepseek_base_url: Some(mock_uri.to_string()),
        moonshot_base_url: None,
    };

    let config = Config::build(cli, None).expect("test Config::build must succeed");
    (tmp, config)
}

/// Mount a mock on the given [`MockServer`] that returns a fixed success
/// response.  Matches requests whose JSON body contains `body_substr`.
async fn mount_success_mock(server: &MockServer, body_substr: &str, response_text: String) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains(body_substr))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"content": response_text}}],
            "total_cost_usd": 0.001
        })))
        .mount(server)
        .await;
}

/// Mount a mock that returns a fixed HTTP error status (no retries on 4xx).
#[allow(dead_code)]
async fn mount_error_mock(server: &MockServer, body_substr: &str, status: u16) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains(body_substr))
        .respond_with(ResponseTemplate::new(status).set_body_string("mock error"))
        .mount(server)
        .await;
}

/// Mount a mock whose response text is produced by a counter, giving each
/// call a distinct, identifiable output.
async fn mount_counting_mock(
    server: &MockServer,
    body_substr: &str,
    label: &str,
) -> Arc<AtomicUsize> {
    let counter = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&counter);
    let label = label.to_string();
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains(body_substr))
        .respond_with(move |_: &wiremock::Request| {
            let n = c.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"content": format!("{label}-{n}")}}],
                "total_cost_usd": 0.001
            }))
        })
        .mount(server)
        .await;
    counter
}

/// Mount a mock that blocks (sleeps) for `delay` before responding, then
/// returns the given status.
async fn mount_delayed_mock(
    server: &MockServer,
    body_substr: &str,
    delay: Duration,
    status: u16,
    response_text: String,
) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains(body_substr))
        .respond_with(
            ResponseTemplate::new(status)
                .set_body_json(json!({
                    "choices": [{"message": {"content": response_text}}],
                    "total_cost_usd": 0.001
                }))
                .set_delay(delay),
        )
        .mount(server)
        .await;
}

/// Drain all buffered events from the receiver within `timeout`.
/// Returns the collected events.
async fn drain_events(rx: &mut UnboundedReceiver<AppEvent>, timeout: Duration) -> Vec<AppEvent> {
    let mut events = Vec::new();
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, rx.recv()).await {
            Ok(Some(event)) => events.push(event),
            Ok(None) => break, // channel closed
            Err(_) => break,   // timed out
        }
    }
    events
}

/// Count how many events of a specific variant were received.
macro_rules! count_events {
    ($events:expr, $pat:pat) => {
        $events.iter().filter(|e| matches!(e, $pat)).count()
    };
}

// =========================================================================
// Test 1 — Concurrent output: both Writer and Critic produce output within
//          the test timeout (10 s).
// =========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_writer_and_critic_produce_output_within_10s() {
    let mut env = EnvGuard::new();
    env.set("DEEPSEEK_API_KEY", "sk-test-mock-key-123456789");

    let server = MockServer::start().await;
    let (_tmp, config) = build_test_config("Initial document.", &server.uri());

    mount_success_mock(&server, "\"deepseek-reasoner\"", "Writer revision".into()).await;
    mount_success_mock(&server, "\"deepseek-chat\"", "Critic remark".into()).await;

    let state = SharedState::new("Initial document.".into());
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let cancel_token = CancellationToken::new();

    let handle = {
        let config = config.clone();
        let cancel = cancel_token.clone();
        tokio::spawn(async move {
            run_spectacle(config, state, event_tx, cancel)
                .await
                .expect("run_spectacle should not error")
        })
    };

    // Let the loops run for several cycles.
    tokio::time::sleep(Duration::from_secs(3)).await;
    cancel_token.cancel();

    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;

    let events = drain_events(&mut event_rx, Duration::from_millis(500)).await;

    let writer_outputs = count_events!(events, AppEvent::WriterOutput(_));
    let critic_outputs = count_events!(events, AppEvent::CriticOutput(_));

    assert!(
        writer_outputs >= 1,
        "expected at least 1 WriterOutput, got {writer_outputs}"
    );
    assert!(
        critic_outputs >= 1,
        "expected at least 1 CriticOutput, got {critic_outputs}"
    );

    eprintln!("Spectacle ran: {writer_outputs} writer outputs, {critic_outputs} critic outputs");
}

// =========================================================================
// Test 2 — Version stamping: Writer increments document version, Critic
//          snapshots the correct version.
// =========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_version_stamping_writer_increments_critic_snapshots() {
    let mut env = EnvGuard::new();
    env.set("DEEPSEEK_API_KEY", "sk-test-mock-key-123456789");

    let server = MockServer::start().await;
    let (_tmp, config) = build_test_config("v0", &server.uri());

    let writer_ctr = mount_counting_mock(&server, "\"deepseek-reasoner\"", "W").await;
    let _critic_ctr = mount_counting_mock(&server, "\"deepseek-chat\"", "C").await;

    let state = SharedState::new("v0".into());
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let cancel_token = CancellationToken::new();

    let handle = {
        let config = config.clone();
        let cancel = cancel_token.clone();
        tokio::spawn(async move {
            run_spectacle(config, state, event_tx, cancel)
                .await
                .expect("run_spectacle should not error")
        })
    };

    tokio::time::sleep(Duration::from_secs(4)).await;
    cancel_token.cancel();

    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
    let events = drain_events(&mut event_rx, Duration::from_millis(500)).await;

    // Collect WriterDone versions — they must be strictly increasing.
    let writer_done_versions: Vec<u64> = events
        .iter()
        .filter_map(|e| {
            if let AppEvent::WriterDone(v) = e {
                Some(*v)
            } else {
                None
            }
        })
        .collect();

    assert!(
        !writer_done_versions.is_empty(),
        "expected at least one WriterDone event"
    );

    // Versions should be strictly increasing.  The first WriterDone starts
    // at 2 because `run_spectacle` does a `state.update(initial_content)`
    // (→ v1) before spawning the writer loop, and the first LLM cycle
    // produces v2.  No WriterDone event is emitted for the initial update.
    for window in writer_done_versions.windows(2) {
        assert!(
            window[0] < window[1],
            "WriterDone versions must be strictly increasing, got {:?}",
            writer_done_versions
        );
    }
    assert!(
        writer_done_versions[0] >= 2,
        "first WriterDone version should be >= 2 (initial update + 1st LLM cycle), got {}",
        writer_done_versions[0]
    );

    // Collect CritiqueReady versions — they should be ≤ the current max
    // Writer version at their time.
    let critique_ready_versions: Vec<u64> = events
        .iter()
        .filter_map(|e| {
            if let AppEvent::CritiqueReady(v) = e {
                Some(*v)
            } else {
                None
            }
        })
        .collect();

    assert!(
        !critique_ready_versions.is_empty(),
        "expected at least one CritiqueReady event"
    );

    // Every critic version should be ≤ the max writer version seen so far.
    let max_writer = *writer_done_versions.last().unwrap_or(&0);
    for &cv in &critique_ready_versions {
        assert!(
            cv <= max_writer,
            "CritiqueReady version {cv} exceeds max Writer version {max_writer}"
        );
    }

    // Verify that the Writer counter was actually incremented.
    let writer_calls = writer_ctr.load(Ordering::SeqCst);
    assert!(
        writer_calls >= 2,
        "expected at least 2 writer LLM calls, got {writer_calls}"
    );

    eprintln!(
        "Version stamping: {writer_calls} writer calls, \
         {wd} WriterDone events, {cr} CritiqueReady events",
        wd = writer_done_versions.len(),
        cr = critique_ready_versions.len(),
    );
}

// =========================================================================
// Test 3 — Stale critique: Critic critiques v3 while Writer is already at
//          v5 → warning logged, no crash.
// =========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_stale_critique_warning_no_crash() {
    let mut env = EnvGuard::new();
    env.set("DEEPSEEK_API_KEY", "sk-test-mock-key-123456789");

    let server = MockServer::start().await;
    let (_tmp, config) = build_test_config("v0", &server.uri());

    // Writer responds very fast (no delay) — will race ahead.
    mount_counting_mock(&server, "\"deepseek-reasoner\"", "W").await;

    // Critic is artificially slow — 3 s delay per response.
    mount_delayed_mock(
        &server,
        "\"deepseek-chat\"",
        Duration::from_secs(3),
        200,
        "Slow critique".into(),
    )
    .await;

    let state = SharedState::new("v0".into());
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let cancel_token = CancellationToken::new();

    let handle = {
        let config = config.clone();
        let cancel = cancel_token.clone();
        let state = state.clone();
        tokio::spawn(async move {
            run_spectacle(config, state, event_tx, cancel)
                .await
                .expect("run_spectacle should not error")
        })
    };

    // Run long enough for the Writer to get several cycles ahead (fast)
    // while the Critic is still on its first slow cycle.
    tokio::time::sleep(Duration::from_secs(6)).await;
    cancel_token.cancel();

    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
    let events = drain_events(&mut event_rx, Duration::from_millis(500)).await;

    // The Writer should have produced several revisions.
    let writer_outputs = count_events!(events, AppEvent::WriterOutput(_));
    assert!(
        writer_outputs >= 3,
        "expected at least 3 WriterOutput events (writer should race ahead while critic is slow), got {writer_outputs}"
    );

    // No errors should have been emitted.
    let errors = count_events!(events, AppEvent::Error(_));
    assert_eq!(
        errors, 0,
        "expected no errors during stale-critique scenario"
    );

    eprintln!("Stale critique test: {writer_outputs} writer outputs — no crash or error events");
}

// =========================================================================
// Test 4 — Stop mechanism: cancel_token.cancelled() → both loops exit
//          within the orchestrator's 3 s shutdown grace period.
// =========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_stop_mechanism_loops_exit_promptly() {
    let mut env = EnvGuard::new();
    env.set("DEEPSEEK_API_KEY", "sk-test-mock-key-123456789");

    let server = MockServer::start().await;
    let (_tmp, config) = build_test_config("Initial.", &server.uri());

    // Use long delays so the loops are likely inside an LLM call when we
    // cancel.  This exercises the select! path at loop bottom rather than
    // the top-of-loop `is_cancelled()` check.
    mount_delayed_mock(
        &server,
        "\"deepseek-reasoner\"",
        Duration::from_secs(4),
        200,
        "writer".into(),
    )
    .await;
    mount_delayed_mock(
        &server,
        "\"deepseek-chat\"",
        Duration::from_secs(4),
        200,
        "critic".into(),
    )
    .await;

    let state = SharedState::new("Initial.".into());
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let cancel_token = CancellationToken::new();

    let start = Instant::now();
    let handle = {
        let config = config.clone();
        let cancel = cancel_token.clone();
        tokio::spawn(async move {
            run_spectacle(config, state, event_tx, cancel)
                .await
                .expect("run_spectacle should not error")
        })
    };

    // Give loops a chance to enter their first LLM call (~500 ms is plenty).
    tokio::time::sleep(Duration::from_millis(800)).await;

    // Cancel — loops should notice at their next cancellation checkpoint
    // (select! branch or top-of-loop check after the LLM call finishes).
    cancel_token.cancel();

    // The orchestrator uses a 3 s shutdown grace period.  With 4 s mock
    // delays, the LLM calls will still be in-flight when cancel fires.
    // The loops return from complete(), see the cancelled token at the
    // next select!, and exit.
    let result = tokio::time::timeout(Duration::from_secs(8), handle).await;
    let elapsed = start.elapsed();

    // Should complete well within 8 s (4 s LLM delay + 3 s grace + overhead).
    assert!(
        result.is_ok(),
        "run_spectacle should return within 8 s, elapsed {elapsed:.1?}"
    );
    assert!(
        elapsed < Duration::from_secs(8),
        "shutdown took {elapsed:.1?}, expected < 8 s"
    );

    let _events = drain_events(&mut event_rx, Duration::from_millis(200)).await;

    eprintln!("Stop mechanism: shutdown completed in {elapsed:.1?}");
}

// =========================================================================
// Test 5a — Error in Writer does not crash Critic.
// =========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_writer_error_does_not_crash_critic() {
    let mut env = EnvGuard::new();
    env.set("DEEPSEEK_API_KEY", "sk-test-mock-key-123456789");

    let server = MockServer::start().await;
    let (_tmp, config) = build_test_config("Initial.", &server.uri());

    // Writer mock: succeed on call 1, fail on call 2+
    let writer_counter = Arc::new(AtomicUsize::new(0));
    let wc = Arc::clone(&writer_counter);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains("\"deepseek-reasoner\""))
        .respond_with(move |_: &wiremock::Request| {
            let n = wc.fetch_add(1, Ordering::SeqCst);
            if n == 1 {
                ResponseTemplate::new(200).set_body_json(json!({
                    "choices": [{"message": {"content": "Writer rev 1"}}],
                    "total_cost_usd": 0.001
                }))
            } else {
                // 400 is NOT retried by do_complete → immediate error
                ResponseTemplate::new(400).set_body_string("mock writer error")
            }
        })
        .mount(&server)
        .await;

    // Critic mock: always succeeds
    let critic_counter = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&critic_counter);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains("\"deepseek-chat\""))
        .respond_with(move |_: &wiremock::Request| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"content": format!("Critique {n}")}}],
                "total_cost_usd": 0.0005
            }))
        })
        .mount(&server)
        .await;

    let state = SharedState::new("Initial.".into());
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let cancel_token = CancellationToken::new();

    let handle = {
        let config = config.clone();
        let cancel = cancel_token.clone();
        tokio::spawn(async move {
            run_spectacle(config, state, event_tx, cancel)
                .await
                .expect("run_spectacle should not error")
        })
    };

    tokio::time::sleep(Duration::from_secs(4)).await;
    cancel_token.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;

    let events = drain_events(&mut event_rx, Duration::from_millis(500)).await;

    // Writer should have produced at least one error event.
    let writer_errors = events
        .iter()
        .filter(|e| matches!(e, AppEvent::Error(_)))
        .count();
    assert!(
        writer_errors >= 1,
        "expected at least one Writer error event, got {writer_errors}"
    );

    // Critic should continue producing output regardless.
    let critic_outputs = count_events!(events, AppEvent::CriticOutput(_));
    assert!(
        critic_outputs >= 2,
        "expected at least 2 CriticOutput events (critic should continue after writer errors), got {critic_outputs}"
    );

    let critic_calls = critic_counter.load(Ordering::SeqCst);
    assert!(
        critic_calls >= 2,
        "critic should have made at least 2 LLM calls, got {critic_calls}"
    );

    eprintln!(
        "Writer-error resilience: {writer_errors} writer errors, \
         {critic_outputs} critic outputs ({critic_calls} critic calls)"
    );
}

// =========================================================================
// Test 5b — Error in Critic does not crash Writer.
// =========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_critic_error_does_not_crash_writer() {
    let mut env = EnvGuard::new();
    env.set("DEEPSEEK_API_KEY", "sk-test-mock-key-123456789");

    let server = MockServer::start().await;
    let (_tmp, config) = build_test_config("Initial.", &server.uri());

    // Writer mock: always succeeds
    let writer_counter = Arc::new(AtomicUsize::new(0));
    let wc = Arc::clone(&writer_counter);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains("\"deepseek-reasoner\""))
        .respond_with(move |_: &wiremock::Request| {
            let n = wc.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"content": format!("Writer rev {n}")}}],
                "total_cost_usd": 0.001
            }))
        })
        .mount(&server)
        .await;

    // Critic mock: succeed on call 1, fail on call 2+
    let critic_counter = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&critic_counter);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains("\"deepseek-chat\""))
        .respond_with(move |_: &wiremock::Request| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            if n == 1 {
                ResponseTemplate::new(200).set_body_json(json!({
                    "choices": [{"message": {"content": "Critique 1"}}],
                    "total_cost_usd": 0.0005
                }))
            } else {
                // 400 — no retry
                ResponseTemplate::new(400).set_body_string("mock critic error")
            }
        })
        .mount(&server)
        .await;

    let state = SharedState::new("Initial.".into());
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let cancel_token = CancellationToken::new();

    let handle = {
        let config = config.clone();
        let cancel = cancel_token.clone();
        tokio::spawn(async move {
            run_spectacle(config, state, event_tx, cancel)
                .await
                .expect("run_spectacle should not error")
        })
    };

    tokio::time::sleep(Duration::from_secs(4)).await;
    cancel_token.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;

    let events = drain_events(&mut event_rx, Duration::from_millis(500)).await;

    // Critic should have produced at least one error event.
    let critic_errors = events
        .iter()
        .filter(|e| matches!(e, AppEvent::Error(_)))
        .count();
    assert!(
        critic_errors >= 1,
        "expected at least one Critic error event, got {critic_errors}"
    );

    // Writer should continue producing output.
    let writer_outputs = count_events!(events, AppEvent::WriterOutput(_));
    assert!(
        writer_outputs >= 2,
        "expected at least 2 WriterOutput events (writer should continue after critic errors), got {writer_outputs}"
    );

    let writer_calls = writer_counter.load(Ordering::SeqCst);
    assert!(
        writer_calls >= 2,
        "writer should have made at least 2 LLM calls, got {writer_calls}"
    );

    eprintln!(
        "Critic-error resilience: {critic_errors} critic errors, \
         {writer_outputs} writer outputs ({writer_calls} writer calls)"
    );
}

// =========================================================================
// Test 6 — Shared state interactions are correct across concurrent agents.
// =========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_shared_state_snapshot_and_update_interactions() {
    let mut env = EnvGuard::new();
    env.set("DEEPSEEK_API_KEY", "sk-test-mock-key-123456789");

    let server = MockServer::start().await;
    let (_tmp, config) = build_test_config("v0", &server.uri());

    mount_counting_mock(&server, "\"deepseek-reasoner\"", "W").await;
    mount_counting_mock(&server, "\"deepseek-chat\"", "C").await;

    let state = SharedState::new("v0".into());
    let initial_snapshot = state.snapshot();
    assert_eq!(initial_snapshot.0, 0);
    assert_eq!(initial_snapshot.1, "v0");

    // No critique yet.
    assert!(state.read_critique().is_none());

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let cancel_token = CancellationToken::new();

    let handle = {
        let config = config.clone();
        let cancel = cancel_token.clone();
        let state = state.clone();
        tokio::spawn(async move {
            run_spectacle(config, state, event_tx, cancel)
                .await
                .expect("run_spectacle should not error")
        })
    };

    tokio::time::sleep(Duration::from_secs(3)).await;
    cancel_token.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
    let _events = drain_events(&mut event_rx, Duration::from_millis(500)).await;

    // After the spectacle runs, the document version should have advanced.
    let final_snapshot = state.snapshot();
    assert!(
        final_snapshot.0 >= 1,
        "document version should be >= 1 after spectacle, got {}",
        final_snapshot.0
    );
    assert!(
        final_snapshot.1.starts_with("W-"),
        "document content should be a writer revision, got {:?}",
        final_snapshot.1
    );

    // A critique should have been stored.
    let critique = state.read_critique();
    assert!(
        critique.is_some(),
        "critique should have been written by the critic"
    );
    if let Some((ver, text)) = critique {
        assert!(
            ver <= final_snapshot.0,
            "critique version {ver} should be <= final document version {}",
            final_snapshot.0
        );
        assert!(!text.is_empty(), "critique text should not be empty");
        eprintln!("Critique stored: version={ver}, text_len={}", text.len());
    }
}

// =========================================================================
// Test 7 — Cancel token cancels mid-flight LLM call without data corruption.
// =========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_cancel_during_llm_call_no_corruption() {
    let mut env = EnvGuard::new();
    env.set("DEEPSEEK_API_KEY", "sk-test-mock-key-123456789");

    let server = MockServer::start().await;
    let (_tmp, config) = build_test_config("Initial.", &server.uri());

    // Use very long delay — loops will be mid-call when we cancel.
    mount_delayed_mock(
        &server,
        "\"deepseek-reasoner\"",
        Duration::from_secs(10),
        200,
        "slow writer".into(),
    )
    .await;
    mount_delayed_mock(
        &server,
        "\"deepseek-chat\"",
        Duration::from_secs(10),
        200,
        "slow critic".into(),
    )
    .await;

    let state = SharedState::new("Initial.".into());
    let (event_tx, _event_rx) = mpsc::unbounded_channel();
    let cancel_token = CancellationToken::new();

    let handle = {
        let config = config.clone();
        let cancel = cancel_token.clone();
        let state = state.clone();
        tokio::spawn(async move {
            run_spectacle(config, state, event_tx, cancel)
                .await
                .expect("run_spectacle should not error")
        })
    };

    // Wait just enough for the first LLM calls to start.
    tokio::time::sleep(Duration::from_millis(500)).await;
    cancel_token.cancel();

    let start_cancel = Instant::now();
    let result = tokio::time::timeout(Duration::from_secs(15), handle).await;
    let cancel_elapsed = start_cancel.elapsed();

    assert!(
        result.is_ok(),
        "run_spectacle should return after cancellation within timeout"
    );

    // The shutdown should complete once the in-flight calls timeout or return.
    // With 10 s mock delays and the reqwest Client timeout at 10 s,
    // the LLM calls will be cut short by the HTTP timeout.
    eprintln!("Cancel during LLM call: shutdown completed in {cancel_elapsed:.1?}");

    // Shared state should still be coherent — no panics.
    // `run_spectacle` does a `state.update(initial_content)` at startup
    // (→ v1), but no LLM revision should have completed.
    let snapshot = state.snapshot();
    assert_eq!(
        snapshot.0, 1,
        "only the initial update (v1) should have completed, no LLM revision"
    );
    assert_eq!(
        snapshot.1, "Initial.",
        "content should be unchanged from initial"
    );
    assert!(
        state.read_critique().is_none(),
        "no critique should have been stored"
    );
}
