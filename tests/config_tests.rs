use std::path::PathBuf;
use std::sync::Mutex;

use agentic_inferno::config::{CliArgs, Config, TomlConfig};
use agentic_inferno::error::AppError;

/// Serialize env-var access across tests running in parallel.
/// All `set_var`/`remove_var` calls and the subsequent `Config::build`
/// must be guarded by this lock to prevent data races.
static ENV_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a temp input file and set DEEPSEEK_API_KEY.
/// Caller MUST hold ENV_LOCK.
fn setup_deepseek(tmp: &tempfile::TempDir) -> PathBuf {
    let input = tmp.path().join("input.txt");
    std::fs::write(&input, "test content").expect("write test input");
    unsafe { std::env::set_var("DEEPSEEK_API_KEY", "sk-test-fake-key-for-testing"); }
    input
}

/// Remove an environment variable.
/// Caller MUST hold ENV_LOCK.
fn rm_env(key: &str) {
    unsafe { std::env::remove_var(key); }
}

/// Standard test CliArgs using DeepSeek models only.
fn cli_deepseek(input: PathBuf) -> CliArgs {
    CliArgs {
        writer_model: "deepseek-reasoner".into(),
        critic_model: Some("deepseek-chat".into()),
        input,
        max_cost_usd: Some(1.0),
        temperature: Some(0.8),
        max_tokens: Some(1024),
        timeout_secs: Some(30),
        config: None,
        critic_style: None,
        openai_base_url: None,
        deepseek_base_url: None,
        moonshot_base_url: None,
    }
}

/// Build an empty TomlConfig (all fields None) from an empty string.
fn empty_toml() -> TomlConfig {
    toml::from_str("").expect("empty toml parses to all-None TomlConfig")
}

/// Create a fake Cargo.toml so `find_repo_root` discovers it.
fn setup_repo_root(dir: &std::path::Path) {
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"test-repo\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo.toml");
}

/// Set CWD to a temporary repo directory, run the closure, restore CWD.
/// DEEPSEEK_API_KEY is set for the duration.  Acquires ENV_LOCK.
fn with_repo_root<F>(f: F)
where
    F: FnOnce(&std::path::Path),
{
    let _lock = ENV_LOCK.lock().unwrap();
    let repo = tempfile::TempDir::new().expect("create temp repo dir");
    setup_repo_root(repo.path());

    let orig_cwd = std::env::current_dir().expect("get original cwd");
    std::env::set_current_dir(repo.path()).expect("set cwd to repo");
    unsafe { std::env::set_var("DEEPSEEK_API_KEY", "sk-test-fake-key"); }

    f(repo.path());

    let _ = std::env::set_current_dir(&orig_cwd);
    rm_env("DEEPSEEK_API_KEY");
}

// ===========================================================================
// 1. Layered precedence: CLI > TOML > .env > defaults
// ===========================================================================

#[test]
fn test_precedence_cli_over_toml() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = setup_deepseek(&tmp);

    let toml = TomlConfig {
        writer_model: Some("deepseek-reasoner".into()),
        critic_model: Some("deepseek-reasoner".into()),
        critic_style: Some("academic-snob".into()),
        input: Some(input.clone()),
        max_cost_usd: Some(5.0),
        temperature: Some(0.5),
        max_tokens: Some(4096),
        timeout_secs: Some(60),
        openai_base_url: Some("https://toml.openai.example.com".into()),
        deepseek_base_url: Some("https://toml.deepseek.example.com".into()),
        moonshot_base_url: None,
    };

    let mut cli = cli_deepseek(input);
    cli.critic_model = Some("deepseek-chat".into());
    cli.max_cost_usd = Some(10.0);
    cli.temperature = Some(0.9);
    cli.max_tokens = Some(8192);
    cli.timeout_secs = Some(120);
    cli.openai_base_url = Some("https://cli.openai.example.com".into());

    let cfg = Config::build(cli, Some(toml)).expect("build should succeed");

    // CLI wins over TOML
    assert_eq!(cfg.critic_model, "deepseek-chat");
    assert_eq!(cfg.max_cost_usd, 10.0);
    assert_eq!(cfg.temperature, 0.9);
    assert_eq!(cfg.max_tokens, 8192);
    assert_eq!(cfg.timeout_secs, 120);
    assert_eq!(
        cfg.openai_base_url.as_deref(),
        Some("https://cli.openai.example.com")
    );

    // TOML fills in where CLI is None
    assert_eq!(cfg.critic_style.to_string(), "academic-snob");
    assert_eq!(
        cfg.deepseek_base_url.as_deref(),
        Some("https://toml.deepseek.example.com")
    );

    rm_env("DEEPSEEK_API_KEY");
}

#[test]
fn test_precedence_toml_over_defaults() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = setup_deepseek(&tmp);

    let toml = TomlConfig {
        writer_model: Some("deepseek-reasoner".into()),
        critic_model: Some("deepseek-reasoner".into()),
        critic_style: Some("disappointed".into()),
        input: Some(input.clone()),
        max_cost_usd: Some(7.5),
        temperature: Some(1.5),
        max_tokens: Some(2048),
        timeout_secs: Some(90),
        openai_base_url: None,
        deepseek_base_url: None,
        moonshot_base_url: None,
    };

    let mut cli = cli_deepseek(input);
    cli.critic_model = None;
    cli.critic_style = None;
    cli.max_cost_usd = None;
    cli.temperature = None;
    cli.max_tokens = None;
    cli.timeout_secs = None;

    let cfg = Config::build(cli, Some(toml)).expect("build should succeed");

    // TOML overrides defaults
    assert_eq!(cfg.critic_model, "deepseek-reasoner");
    assert_eq!(cfg.max_cost_usd, 7.5);
    assert_eq!(cfg.temperature, 1.5);
    assert_eq!(cfg.max_tokens, 2048);
    assert_eq!(cfg.timeout_secs, 90);
    assert_eq!(cfg.critic_style.to_string(), "disappointed");

    rm_env("DEEPSEEK_API_KEY");
}

#[test]
fn test_precedence_env_base_url_fills_gap() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = setup_deepseek(&tmp);

    unsafe { std::env::set_var("OPENAI_BASE_URL", "https://env.openai.example.com"); }

    let cli = cli_deepseek(input);
    let cfg = Config::build(cli, None).expect("build should succeed");

    assert_eq!(
        cfg.openai_base_url.as_deref(),
        Some("https://env.openai.example.com")
    );

    unsafe { std::env::remove_var("OPENAI_BASE_URL"); }
    rm_env("DEEPSEEK_API_KEY");
}

#[test]
fn test_precedence_toml_over_env_for_base_url() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = setup_deepseek(&tmp);

    unsafe { std::env::set_var("OPENAI_BASE_URL", "https://env.openai.example.com"); }

    let toml = TomlConfig {
        openai_base_url: Some("https://toml.openai.example.com".into()),
        ..empty_toml()
    };

    let mut cli = cli_deepseek(input);
    cli.openai_base_url = None;

    let cfg = Config::build(cli, Some(toml)).expect("build should succeed");

    // TOML wins over env var (layer 2 applied before layer 3)
    assert_eq!(
        cfg.openai_base_url.as_deref(),
        Some("https://toml.openai.example.com")
    );

    unsafe { std::env::remove_var("OPENAI_BASE_URL"); }
    rm_env("DEEPSEEK_API_KEY");
}

// ===========================================================================
// 2. Numerical bounds: max_cost_usd <= 0
// ===========================================================================

#[test]
fn test_validate_max_cost_usd_negative() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = setup_deepseek(&tmp);
    let mut cli = cli_deepseek(input);
    cli.max_cost_usd = Some(-1.0);
    let err = Config::build(cli, None).unwrap_err();
    assert!(matches!(&err, AppError::Validation(_)));
    assert!(err.to_string().contains("max-cost-usd"), "error: {err}");
    rm_env("DEEPSEEK_API_KEY");
}

#[test]
fn test_validate_max_cost_usd_zero() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = setup_deepseek(&tmp);
    let mut cli = cli_deepseek(input);
    cli.max_cost_usd = Some(0.0);
    let err = Config::build(cli, None).unwrap_err();
    assert!(matches!(&err, AppError::Validation(_)));
    assert!(err.to_string().contains("max-cost-usd"), "error: {err}");
    rm_env("DEEPSEEK_API_KEY");
}

// ===========================================================================
// 3. Numerical bounds: temperature out of [0.0, 2.0]
// ===========================================================================

#[test]
fn test_validate_temperature_below_range() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = setup_deepseek(&tmp);
    let mut cli = cli_deepseek(input);
    cli.temperature = Some(-0.1);
    let err = Config::build(cli, None).unwrap_err();
    assert!(matches!(&err, AppError::Validation(_)));
    assert!(err.to_string().contains("temperature"), "error: {err}");
    rm_env("DEEPSEEK_API_KEY");
}

#[test]
fn test_validate_temperature_above_range() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = setup_deepseek(&tmp);
    let mut cli = cli_deepseek(input);
    cli.temperature = Some(2.1);
    let err = Config::build(cli, None).unwrap_err();
    assert!(matches!(&err, AppError::Validation(_)));
    assert!(err.to_string().contains("temperature"), "error: {err}");
    rm_env("DEEPSEEK_API_KEY");
}

#[test]
fn test_validate_temperature_boundary_low() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = setup_deepseek(&tmp);
    let mut cli = cli_deepseek(input);
    cli.temperature = Some(0.0);
    assert!(Config::build(cli, None).is_ok());
    rm_env("DEEPSEEK_API_KEY");
}

#[test]
fn test_validate_temperature_boundary_high() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = setup_deepseek(&tmp);
    let mut cli = cli_deepseek(input);
    cli.temperature = Some(2.0);
    assert!(Config::build(cli, None).is_ok());
    rm_env("DEEPSEEK_API_KEY");
}

// ===========================================================================
// 4. Unknown model → AppError::UnknownModel with agent name
// ===========================================================================

#[test]
fn test_unknown_model_writer() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = setup_deepseek(&tmp);
    let mut cli = cli_deepseek(input);
    cli.writer_model = "nonexistent-model-v42".into();
    let err = Config::build(cli, None).unwrap_err();
    assert!(
        matches!(&err, AppError::UnknownModel(model, agent)
            if model == "nonexistent-model-v42" && agent == "Writer"
        ),
        "expected UnknownModel for Writer, got {err:?}"
    );
    rm_env("DEEPSEEK_API_KEY");
}

#[test]
fn test_unknown_model_critic() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = setup_deepseek(&tmp);
    let mut cli = cli_deepseek(input);
    cli.critic_model = Some("fake-model-9000".into());
    let err = Config::build(cli, None).unwrap_err();
    assert!(
        matches!(&err, AppError::UnknownModel(model, agent)
            if model == "fake-model-9000" && agent == "Critic"
        ),
        "expected UnknownModel for Critic, got {err:?}"
    );
    rm_env("DEEPSEEK_API_KEY");
}

// ===========================================================================
// 5. Missing API key → AppError::MissingKey
// ===========================================================================

#[test]
fn test_missing_api_key_deepseek() {
    // Use lock to ensure no other test can interfere with env state.
    let _lock = ENV_LOCK.lock().unwrap();
    // Explicitly clear the key — a previous test may have left it set.
    rm_env("DEEPSEEK_API_KEY");

    let tmp = tempfile::TempDir::new().unwrap();
    let input = tmp.path().join("input.txt");
    std::fs::write(&input, "content").expect("write input");

    let cli = cli_deepseek(input);
    let err = Config::build(cli, None).unwrap_err();
    assert!(
        matches!(&err, AppError::MissingKey(key) if key == "DEEPSEEK_API_KEY"),
        "expected MissingKey for DEEPSEEK_API_KEY, got {err:?}"
    );
}

#[test]
fn test_missing_api_key_openai() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = tmp.path().join("input.txt");
    std::fs::write(&input, "content").expect("write input");

    let mut cli = cli_deepseek(input);
    cli.writer_model = "gpt-4o".into();
    cli.critic_model = Some("deepseek-chat".into());
    unsafe { std::env::set_var("DEEPSEEK_API_KEY", "sk-test-fake-key"); }

    let err = Config::build(cli, None).unwrap_err();
    assert!(
        matches!(&err, AppError::MissingKey(key) if key == "OPENAI_API_KEY"),
        "expected MissingKey for OPENAI_API_KEY, got {err:?}"
    );
    rm_env("DEEPSEEK_API_KEY");
}

#[test]
fn test_missing_api_key_placeholder_rejected() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = tmp.path().join("input.txt");
    std::fs::write(&input, "content").expect("write input");

    let mut cli = cli_deepseek(input);
    cli.writer_model = "gpt-4o".into();
    cli.critic_model = Some("deepseek-chat".into());
    unsafe {
        std::env::set_var("DEEPSEEK_API_KEY", "sk-test-fake");
        std::env::set_var("OPENAI_API_KEY", "sk-...");
    }

    let err = Config::build(cli, None).unwrap_err();
    assert!(
        matches!(&err, AppError::MissingKey(key) if key == "OPENAI_API_KEY"),
        "expected MissingKey for placeholder key, got {err:?}"
    );
    rm_env("DEEPSEEK_API_KEY");
    unsafe { std::env::remove_var("OPENAI_API_KEY"); }
}

// ===========================================================================
// 6. claude not on PATH → AppError::ClaudeNotFound
// ===========================================================================

#[test]
fn test_claude_not_found() {
    let _lock = ENV_LOCK.lock().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let input = tmp.path().join("input.txt");
    std::fs::write(&input, "content").expect("write input");

    unsafe {
        std::env::set_var("DEEPSEEK_API_KEY", "sk-test-fake-key");
        std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-test-key");
    }

    let mut cli = cli_deepseek(input);
    cli.writer_model = "claude-sonnet-4-20250514".into();
    cli.critic_model = Some("deepseek-chat".into());

    let result = Config::build(cli, None);

    let claude_available = std::process::Command::new("claude")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if claude_available {
        if let Err(err) = &result {
            assert!(
                !matches!(err, AppError::ClaudeNotFound),
                "claude is on PATH but still got ClaudeNotFound"
            );
        }
    } else {
        assert!(
            matches!(&result, Err(AppError::ClaudeNotFound)),
            "expected ClaudeNotFound, got {result:?}"
        );
    }

    rm_env("DEEPSEEK_API_KEY");
    unsafe { std::env::remove_var("ANTHROPIC_API_KEY"); }
}

// ===========================================================================
// 7–9. Leak guard
// ===========================================================================

#[test]
fn test_leak_guard_inside_repo_not_inputs() {
    with_repo_root(|repo| {
        let src_dir = repo.join("src");
        std::fs::create_dir_all(&src_dir).expect("create src dir");
        let file = src_dir.join("lib.rs");
        std::fs::write(&file, "content").expect("write file");

        let cli = cli_deepseek(file);
        let err = Config::build(cli, None).unwrap_err();
        assert!(
            matches!(&err, AppError::LeakGuard(_)),
            "expected LeakGuard, got {err:?}"
        );
    });
}

#[test]
fn test_leak_guard_outside_repo() {
    with_repo_root(|repo| {
        let _ = repo;
        let outside = tempfile::TempDir::new().expect("create outside dir");
        let file = outside.path().join("data.txt");
        std::fs::write(&file, "content").expect("write file");

        let cli = cli_deepseek(file);
        let cfg = Config::build(cli, None).expect("build should succeed");
        assert!(cfg.input.exists(), "input file should exist after build");
    });
}

#[test]
fn test_leak_guard_in_inputs() {
    with_repo_root(|repo| {
        let inputs_dir = repo.join("inputs");
        std::fs::create_dir_all(&inputs_dir).expect("create inputs dir");
        let file = inputs_dir.join("data.txt");
        std::fs::write(&file, "content").expect("write file");

        let cli = cli_deepseek(file);
        let cfg = Config::build(cli, None).expect("build should succeed");
        assert!(cfg.input.exists(), "input file should exist");
        assert!(
            cfg.input.display().to_string().contains("inputs"),
            "input path should contain 'inputs', got: {}",
            cfg.input.display()
        );
    });
}
