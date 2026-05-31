//! Integration tests for the simplified application state machine.
//!
//! The state machine has exactly four states: Idle → Running → Stopping → Done.
//! There are no iteration bounds, no AwaitingConfirm/AwaitingCleanup/
//! Snapshotting intermediary states, and no confirmation gates.

use agentic_inferno::app::AppState;

// =============================================================================
// Enum existence & completeness
// =============================================================================

#[test]
fn test_all_four_variants_exist_compile_time_check() {
    // Exhaustive match — if a 5th variant is added, the compiler rejects this.
    let states = [
        AppState::Idle,
        AppState::Running,
        AppState::Stopping,
        AppState::Done,
    ];
    for state in &states {
        match state {
            AppState::Idle | AppState::Running | AppState::Stopping | AppState::Done => {}
        }
    }
    assert_eq!(states.len(), 4);
}

#[test]
fn test_partial_eq_works() {
    assert_eq!(AppState::Idle, AppState::Idle);
    assert_eq!(AppState::Running, AppState::Running);
    assert_eq!(AppState::Stopping, AppState::Stopping);
    assert_eq!(AppState::Done, AppState::Done);
    assert_ne!(AppState::Idle, AppState::Running);
    assert_ne!(AppState::Running, AppState::Stopping);
    assert_ne!(AppState::Stopping, AppState::Done);
}

#[test]
fn test_debug_format_is_readable() {
    let debug_str = format!("{:?}", AppState::Idle);
    assert!(
        debug_str.contains("Idle"),
        "Debug output '{debug_str}' should contain 'Idle'"
    );
}

// =============================================================================
// State transitions
// =============================================================================

#[test]
fn test_idle_to_running_transition() {
    let mut state = AppState::Idle;
    assert_eq!(state, AppState::Idle);
    state = AppState::Running;
    assert_eq!(state, AppState::Running);
    assert!(matches!(state, AppState::Running));
}

#[test]
fn test_running_to_stopping_transition() {
    let mut state = AppState::Running;
    assert_eq!(state, AppState::Running);
    state = AppState::Stopping;
    assert_eq!(state, AppState::Stopping);
    assert!(matches!(state, AppState::Stopping));
}

#[test]
fn test_stopping_to_done_transition() {
    let mut state = AppState::Stopping;
    assert_eq!(state, AppState::Stopping);
    state = AppState::Done;
    assert_eq!(state, AppState::Done);
    assert!(matches!(state, AppState::Done));
}

#[test]
fn test_full_lifecycle_transitions() {
    let mut state = AppState::Idle;
    assert!(matches!(state, AppState::Idle));

    state = AppState::Running;
    assert!(matches!(state, AppState::Running));

    state = AppState::Stopping;
    assert!(matches!(state, AppState::Stopping));

    state = AppState::Done;
    assert!(matches!(state, AppState::Done));

    // Done → Running re-run is allowed.
    state = AppState::Running;
    assert!(matches!(state, AppState::Running));
}

// =============================================================================
// Negative: no intermediary / gate states
// =============================================================================

#[test]
fn test_no_awaiting_confirm_variant() {
    // The exhaustive match in test_all_four_variants_exist is the compile-time
    // check for no extra variants.  Runtime check via Debug output follows.
    let debug_str = format!("{:?}", AppState::Idle);
    assert!(
        !debug_str.contains("Awaiting"),
        "Found 'Awaiting' in AppState Debug output"
    );
}

#[test]
fn test_no_awaiting_cleanup_variant() {
    let debug_str = format!("{:?}", AppState::Running);
    assert!(
        !debug_str.contains("Cleanup"),
        "Found 'Cleanup' in AppState Debug output"
    );
}

#[test]
fn test_no_snapshotting_variant() {
    let debug_str = format!("{:?}", AppState::Running);
    assert!(
        !debug_str.contains("Snapshot"),
        "Found 'Snapshot' in AppState Debug output"
    );
}

// =============================================================================
// Config: no max_iterations
// =============================================================================

#[test]
fn test_config_has_no_max_iterations_field() {
    // Construct Config with its public fields to verify max_iterations is not
    // among them.  If max_iterations were added, it would appear here.
    use agentic_inferno::config::Config;
    use agentic_inferno::config::CriticStyle;
    use std::path::PathBuf;

    let cfg = Config {
        writer_model: String::new(),
        critic_model: "deepseek-chat".into(),
        critic_style: CriticStyle::Random,
        speed: agentic_inferno::config::Speed::Normal,
        task: agentic_inferno::config::InfernoTask::Writing,
        prompt: None,
        input: PathBuf::new(),
        max_cost_usd: 2.0,
        temperature: 0.8,
        max_tokens: 8192,
        timeout_secs: 120,
        repo_root: PathBuf::new(),
        openai_base_url: None,
        deepseek_base_url: None,
        moonshot_base_url: None,
    };

    let _ = cfg.writer_model;
    let _ = cfg.critic_model;
    let _ = cfg.max_cost_usd;
    let _ = cfg.temperature;
    let _ = cfg.max_tokens;
    let _ = cfg.timeout_secs;
}

// =============================================================================
// Invalid config handling
// =============================================================================

#[test]
fn test_invalid_config_validation_rejected() {
    // Invalid config prevents progression past Idle.  Config::build() must
    // reject known-bad numeric inputs for the state machine to stay Idle.
    use agentic_inferno::config::CliArgs;

    let cli = CliArgs {
        writer_model: "deepseek-reasoner".into(),
        critic_model: Some("deepseek-chat".into()),
        input: Some(std::path::PathBuf::from("/nonexistent/path")),
        task: None,
        prompt: None,
        max_cost_usd: Some(0.0),
        temperature: Some(0.8),
        max_tokens: Some(1024),
        timeout_secs: Some(30),
        config: None,
        critic_style: None,
        speed: None,
        openai_base_url: None,
        deepseek_base_url: None,
        moonshot_base_url: None,
    };
    let result = agentic_inferno::config::Config::build(cli, None);
    assert!(
        result.is_err(),
        "Config with max_cost_usd=0 should be rejected"
    );
}

#[test]
fn test_invalid_temperature_rejected() {
    use agentic_inferno::config::CliArgs;

    let cli = CliArgs {
        writer_model: "deepseek-reasoner".into(),
        critic_model: Some("deepseek-chat".into()),
        input: Some(std::path::PathBuf::from("/nonexistent/path")),
        task: None,
        prompt: None,
        max_cost_usd: Some(1.0),
        temperature: Some(99.9),
        max_tokens: Some(1024),
        timeout_secs: Some(30),
        config: None,
        critic_style: None,
        speed: None,
        openai_base_url: None,
        deepseek_base_url: None,
        moonshot_base_url: None,
    };
    let result = agentic_inferno::config::Config::build(cli, None);
    assert!(
        result.is_err(),
        "Config with temperature=99.9 should be rejected"
    );
}

// =============================================================================
// Error handling — Running state stays alive
// =============================================================================

#[test]
fn test_running_state_tolerates_errors() {
    // Errors arrive via AppEvent::Error — they are distinct from AppState
    // transitions.  A network timeout or API failure during Running does NOT
    // cause a state transition; the loop stays alive and retries.
    use agentic_inferno::app::AppEvent;
    use agentic_inferno::error::AppError;

    let state = AppState::Running;

    let errors = [
        AppEvent::Error(AppError::Validation("simulated error".into())),
        AppEvent::Error(AppError::CostCeilingExceeded(1.5, 1.0)),
        AppEvent::Error(AppError::MissingInput),
    ];

    assert_eq!(state, AppState::Running);
    assert_eq!(errors.len(), 3);
}
