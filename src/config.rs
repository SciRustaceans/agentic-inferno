use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;
use serde::Deserialize;

use crate::error::AppError;
use crate::providers::{detect_provider, resolve_claude_bin, Provider};

// ---------------------------------------------------------------------------
// CriticStyle
// ---------------------------------------------------------------------------

/// The personality/attitude the critic agent adopts for its commentary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CriticStyle {
    Aggressive,
    PassiveAggressive,
    Theatrical,
    AcademicSnob,
    Disappointed,
    Random,
}

impl fmt::Display for CriticStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CriticStyle::Aggressive => write!(f, "aggressive"),
            CriticStyle::PassiveAggressive => write!(f, "passive-aggressive"),
            CriticStyle::Theatrical => write!(f, "theatrical"),
            CriticStyle::AcademicSnob => write!(f, "academic-snob"),
            CriticStyle::Disappointed => write!(f, "disappointed"),
            CriticStyle::Random => write!(f, "random"),
        }
    }
}

impl FromStr for CriticStyle {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Normalise: lowercase, replace hyphens/underscores with underscores,
        // then strip all underscores for comparison.
        let normalised: String = s
            .to_lowercase()
            .chars()
            .map(|c| if c == '-' || c == '_' { '_' } else { c })
            .collect();
        match normalised.as_str() {
            "aggressive" => Ok(CriticStyle::Aggressive),
            "passive_aggressive" | "passiveaggressive" => Ok(CriticStyle::PassiveAggressive),
            "theatrical" => Ok(CriticStyle::Theatrical),
            "academic_snob" | "academicsnob" => Ok(CriticStyle::AcademicSnob),
            "disappointed" => Ok(CriticStyle::Disappointed),
            "random" => Ok(CriticStyle::Random),
            _ => Err(format!(
                "Unknown critic style '{s}'. Valid: aggressive, passive-aggressive, theatrical, academic-snob, disappointed, random"
            )),
        }
    }
}

// clap::ValueEnum for CLI parsing — uses kebab-case by default.
impl clap::ValueEnum for CriticStyle {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            CriticStyle::Aggressive,
            CriticStyle::PassiveAggressive,
            CriticStyle::Theatrical,
            CriticStyle::AcademicSnob,
            CriticStyle::Disappointed,
            CriticStyle::Random,
        ]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(clap::builder::PossibleValue::new(match self {
            CriticStyle::Aggressive => "aggressive",
            CriticStyle::PassiveAggressive => "passive-aggressive",
            CriticStyle::Theatrical => "theatrical",
            CriticStyle::AcademicSnob => "academic-snob",
            CriticStyle::Disappointed => "disappointed",
            CriticStyle::Random => "random",
        }))
    }
}

// ---------------------------------------------------------------------------
// Speed
// ---------------------------------------------------------------------------

/// Typewriter reveal speed for the Writer/Critic panes.
///
/// Maps to a characters-per-second rate via [`Speed::cps`] that drives both the
/// TUI reveal animation and how long each agent loop waits for its reply to
/// finish typing out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Speed {
    /// Slow reveal — 20 chars/sec.
    Slow,
    /// Normal reveal — 40 chars/sec. The default.
    #[default]
    Normal,
    /// Fast reveal — 80 chars/sec.
    Fast,
}

impl Speed {
    /// Characters-per-second reveal rate for this speed.
    pub fn cps(&self) -> u32 {
        match self {
            Speed::Slow => 20,
            Speed::Normal => 40,
            Speed::Fast => 80,
        }
    }
}

impl fmt::Display for Speed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Speed::Slow => write!(f, "slow"),
            Speed::Normal => write!(f, "normal"),
            Speed::Fast => write!(f, "fast"),
        }
    }
}

// clap::ValueEnum for CLI parsing — uses kebab-case (single-word) values.
impl clap::ValueEnum for Speed {
    fn value_variants<'a>() -> &'a [Self] {
        &[Speed::Slow, Speed::Normal, Speed::Fast]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(clap::builder::PossibleValue::new(match self {
            Speed::Slow => "slow",
            Speed::Normal => "normal",
            Speed::Fast => "fast",
        }))
    }
}

// ---------------------------------------------------------------------------
// InfernoTask
// ---------------------------------------------------------------------------

/// The kind of work the Writer agent is set loose on.
///
/// `Prompt` is the "guided" mode: the Writer is handed a free-form task it can
/// never fully complete and keeps attempting it forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InfernoTask {
    /// Revise a piece of prose. The default.
    #[default]
    Writing,
    /// Revise or rewrite a code file.
    Code,
    /// Expand a research write-up.
    Research,
    /// Re-analyse material and draw conclusions.
    Analysis,
    /// Attempt a free-form prompt that can never be fully completed.
    Prompt,
}

impl fmt::Display for InfernoTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InfernoTask::Writing => write!(f, "writing"),
            InfernoTask::Code => write!(f, "code"),
            InfernoTask::Research => write!(f, "research"),
            InfernoTask::Analysis => write!(f, "analysis"),
            InfernoTask::Prompt => write!(f, "prompt"),
        }
    }
}

// clap::ValueEnum for CLI parsing — uses kebab-case (single-word) values.
impl clap::ValueEnum for InfernoTask {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            InfernoTask::Writing,
            InfernoTask::Code,
            InfernoTask::Research,
            InfernoTask::Analysis,
            InfernoTask::Prompt,
        ]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(clap::builder::PossibleValue::new(match self {
            InfernoTask::Writing => "writing",
            InfernoTask::Code => "code",
            InfernoTask::Research => "research",
            InfernoTask::Analysis => "analysis",
            InfernoTask::Prompt => "prompt",
        }))
    }
}

// ---------------------------------------------------------------------------
// CLI args (clap derive)
// ---------------------------------------------------------------------------

/// Agentic Inferno — watch a Writer LLM revise a document while a Critic LLM heckles it.
#[derive(Parser, Debug)]
#[command(name = "agentic-inferno", version, about)]
pub struct CliArgs {
    /// Model to use for the writer agent (required).
    #[arg(long = "writer-model", required = true)]
    pub writer_model: String,

    /// Model to use for the critic agent (default: deepseek-chat).
    #[arg(long = "critic-model")]
    pub critic_model: Option<String>,

    /// Input file or directory path for the writer to work on.
    ///
    /// Required for every task except `prompt`; ignored in prompt mode.
    #[arg(long = "input", required = false, value_hint = clap::ValueHint::FilePath)]
    pub input: Option<PathBuf>,

    /// What kind of work the writer agent attempts (default: writing).
    #[arg(long = "task")]
    pub task: Option<InfernoTask>,

    /// A free-form prompt for the writer to attempt forever (implies --task prompt).
    #[arg(long = "prompt")]
    pub prompt: Option<String>,

    /// Maximum total cost in USD before stopping (default: 2.0).
    #[arg(long = "max-cost-usd")]
    pub max_cost_usd: Option<f64>,

    /// Temperature for model sampling (0.0–2.0, default: 0.8).
    #[arg(long = "temperature")]
    pub temperature: Option<f64>,

    /// Maximum tokens per model response (default: 8192).
    #[arg(long = "max-tokens")]
    pub max_tokens: Option<u32>,

    /// Request timeout in seconds (default: 120).
    #[arg(long = "timeout-secs")]
    pub timeout_secs: Option<u64>,

    /// Path to an optional TOML config file.
    #[arg(long = "config", value_hint = clap::ValueHint::FilePath)]
    pub config: Option<PathBuf>,

    /// Critic personality style (default: random).
    #[arg(long = "critic-style")]
    pub critic_style: Option<CriticStyle>,

    /// Typewriter reveal speed for the panes (default: normal).
    #[arg(long = "speed")]
    pub speed: Option<Speed>,

    /// OpenAI-compatible API base URL (overrides default).
    #[arg(long = "openai-base-url")]
    pub openai_base_url: Option<String>,

    /// DeepSeek API base URL (overrides default).
    #[arg(long = "deepseek-base-url")]
    pub deepseek_base_url: Option<String>,

    /// Moonshot API base URL (overrides default).
    #[arg(long = "moonshot-base-url")]
    pub moonshot_base_url: Option<String>,
}

// ---------------------------------------------------------------------------
// TOML config (serde deserialise)
// ---------------------------------------------------------------------------

/// Optional TOML config file structure.
/// All fields are optional — partial config files are allowed.
#[derive(Debug, Deserialize)]
pub struct TomlConfig {
    pub writer_model: Option<String>,
    pub critic_model: Option<String>,
    pub critic_style: Option<String>,
    pub input: Option<PathBuf>,
    pub max_cost_usd: Option<f64>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub timeout_secs: Option<u64>,
    pub openai_base_url: Option<String>,
    pub deepseek_base_url: Option<String>,
    pub moonshot_base_url: Option<String>,
}

impl TomlConfig {
    /// Load and parse a TOML config file from disk.
    pub fn from_file(path: &std::path::Path) -> Result<Self, AppError> {
        let contents = std::fs::read_to_string(path)?;
        toml::from_str(&contents).map_err(|e| {
            AppError::Validation(format!(
                "Failed to parse config file '{}': {e}",
                path.display()
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// Main Config
// ---------------------------------------------------------------------------

/// Fully resolved and validated application configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Model name for the writer agent.
    pub writer_model: String,
    /// Model name for the critic agent.
    pub critic_model: String,
    /// Critic personality style.
    pub critic_style: CriticStyle,
    /// Typewriter reveal speed for the panes.
    pub speed: Speed,
    /// The kind of work the writer agent attempts.
    pub task: InfernoTask,
    /// Free-form prompt for `InfernoTask::Prompt` mode (None otherwise).
    pub prompt: Option<String>,
    /// Canonicalised input file path (exists after build).
    ///
    /// In `InfernoTask::Prompt` mode no input file is used and this stays
    /// empty (`PathBuf::new()`).
    pub input: PathBuf,
    /// Maximum total cost in USD.
    pub max_cost_usd: f64,
    /// Sampling temperature.
    pub temperature: f64,
    /// Max tokens per response.
    pub max_tokens: u32,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Repository root directory (discovered during build).
    pub repo_root: PathBuf,
    /// Override base URL for OpenAI-compatible endpoints.
    pub openai_base_url: Option<String>,
    /// Override base URL for DeepSeek.
    pub deepseek_base_url: Option<String>,
    /// Override base URL for Moonshot.
    pub moonshot_base_url: Option<String>,
}

impl Config {
    /// Default values used when nothing else provides a value.
    fn defaults() -> Self {
        Self {
            writer_model: String::new(), // required from CLI
            critic_model: "deepseek-chat".into(),
            critic_style: CriticStyle::Random,
            speed: Speed::Normal,
            task: InfernoTask::Writing,
            prompt: None,
            input: PathBuf::new(), // required from CLI (except prompt mode)
            max_cost_usd: 2.0,
            temperature: 0.8,
            max_tokens: 8192,
            timeout_secs: 120,
            repo_root: PathBuf::new(),
            openai_base_url: None,
            deepseek_base_url: None,
            moonshot_base_url: None,
        }
    }

    /// Characters-per-second reveal rate derived from the configured [`Speed`].
    ///
    /// Drives both the TUI typewriter animation and how long each agent loop
    /// waits for its reply to finish typing out.
    pub fn reveal_cps(&self) -> u32 {
        self.speed.cps()
    }

    /// Build a validated `Config` using layered precedence:
    ///
    /// 1. Hardcoded defaults
    /// 2. TOML config file values
    /// 3. `.env` / environment variable overrides (base URLs)
    /// 4. CLI arguments (highest priority)
    pub fn build(cli: CliArgs, toml: Option<TomlConfig>) -> Result<Self, AppError> {
        let mut config = Self::defaults();

        // --- Layer 2: TOML overrides defaults ---
        if let Some(t) = toml {
            if let Some(v) = t.writer_model {
                config.writer_model = v;
            }
            if let Some(v) = t.critic_model {
                config.critic_model = v;
            }
            if let Some(v) = t.critic_style {
                config.critic_style = v.parse::<CriticStyle>().map_err(|e| {
                    AppError::Validation(format!("critic_style in config file: {e}"))
                })?;
            }
            if let Some(v) = t.input {
                config.input = v;
            }
            if let Some(v) = t.max_cost_usd {
                config.max_cost_usd = v;
            }
            if let Some(v) = t.temperature {
                config.temperature = v;
            }
            if let Some(v) = t.max_tokens {
                config.max_tokens = v;
            }
            if let Some(v) = t.timeout_secs {
                config.timeout_secs = v;
            }
            if let Some(v) = t.openai_base_url {
                config.openai_base_url = Some(v);
            }
            if let Some(v) = t.deepseek_base_url {
                config.deepseek_base_url = Some(v);
            }
            if let Some(v) = t.moonshot_base_url {
                config.moonshot_base_url = Some(v);
            }
        }

        // --- Layer 3: .env / env var base URL overrides (if not already set) ---
        if config.openai_base_url.is_none() {
            config.openai_base_url = std::env::var("OPENAI_BASE_URL").ok();
        }
        if config.deepseek_base_url.is_none() {
            config.deepseek_base_url = std::env::var("DEEPSEEK_BASE_URL").ok();
        }
        if config.moonshot_base_url.is_none() {
            config.moonshot_base_url = std::env::var("MOONSHOT_BASE_URL").ok();
        }

        // --- Layer 4: CLI overrides everything ---
        // writer_model is required by clap — always present.
        config.writer_model = cli.writer_model;

        // Resolve the task. Supplying --prompt implies --task prompt unless an
        // explicit --task was given (the explicit flag wins).
        config.task = cli.task.unwrap_or(if cli.prompt.is_some() {
            InfernoTask::Prompt
        } else {
            InfernoTask::Writing
        });
        config.prompt = cli.prompt;

        // --input is conditionally required (enforced in validate()). Only
        // carry it over when actually supplied.
        if let Some(input) = cli.input {
            config.input = input;
        }
        if let Some(v) = cli.critic_model {
            config.critic_model = v;
        }
        if let Some(v) = cli.critic_style {
            config.critic_style = v;
        }
        if let Some(v) = cli.speed {
            config.speed = v;
        }
        if let Some(v) = cli.max_cost_usd {
            config.max_cost_usd = v;
        }
        if let Some(v) = cli.temperature {
            config.temperature = v;
        }
        if let Some(v) = cli.max_tokens {
            config.max_tokens = v;
        }
        if let Some(v) = cli.timeout_secs {
            config.timeout_secs = v;
        }
        if let Some(v) = cli.openai_base_url {
            config.openai_base_url = Some(v);
        }
        if let Some(v) = cli.deepseek_base_url {
            config.deepseek_base_url = Some(v);
        }
        if let Some(v) = cli.moonshot_base_url {
            config.moonshot_base_url = Some(v);
        }

        // --- Validate ---
        config.validate()?;

        Ok(config)
    }

    /// Validate the resolved configuration.
    fn validate(&mut self) -> Result<(), AppError> {
        // Numerical bounds
        if self.max_cost_usd <= 0.0 {
            return Err(AppError::Validation(
                "max-cost-usd must be greater than 0".into(),
            ));
        }
        if !(0.0..=2.0).contains(&self.temperature) {
            return Err(AppError::Validation(
                "temperature must be in the range [0.0, 2.0]".into(),
            ));
        }
        if self.max_tokens == 0 {
            return Err(AppError::Validation(
                "max-tokens must be greater than 0".into(),
            ));
        }
        if self.timeout_secs == 0 {
            return Err(AppError::Validation(
                "timeout-secs must be greater than 0".into(),
            ));
        }

        // Detect providers for both models
        let writer_provider = detect_provider(&self.writer_model, "Writer")?;
        let critic_provider = detect_provider(&self.critic_model, "Critic")?;

        // Check API keys for all used providers
        for provider in [writer_provider.0, critic_provider.0] {
            let env_var = provider.api_key_env_var();
            let key = std::env::var(env_var).unwrap_or_default();
            if key.is_empty() || key.starts_with("sk-...") {
                // "sk-..." is the placeholder from .env.example — treat as missing
                return Err(AppError::MissingKey(env_var.into()));
            }
        }

        // Check `claude` CLI on PATH if Anthropic model selected
        if (writer_provider.0 == Provider::Anthropic || critic_provider.0 == Provider::Anthropic)
            && !claude_cli_on_path()
        {
            return Err(AppError::ClaudeNotFound);
        }

        // Input requirements depend on the task.
        if self.task == InfernoTask::Prompt {
            // Prompt mode: --prompt is required; --input is ignored, so the
            // leak guard and canonicalisation are skipped entirely.
            let has_prompt = self
                .prompt
                .as_deref()
                .map(|p| !p.trim().is_empty())
                .unwrap_or(false);
            if !has_prompt {
                return Err(AppError::Validation(
                    "--prompt <text> is required when --task is 'prompt' (or when relying on --prompt to select the task)".into(),
                ));
            }
        } else {
            // Non-prompt tasks: --input is required. An empty path means it was
            // never supplied (clap no longer enforces it).
            if self.input.as_os_str().is_empty() {
                return Err(AppError::MissingInput);
            }
            // Leak guard — canonicalise input, verify it's either outside repo
            // or inside repo/inputs/.
            self.apply_leak_guard()?;
        }

        Ok(())
    }

    /// Canonicalise `--input`, discover repo root, and enforce the leak guard.
    ///
    /// Rules:
    /// - Input OUTSIDE repo root → allowed.
    /// - Input INSIDE repo root AND in `repo_root/inputs/` → allowed.
    /// - Input INSIDE repo root but NOT in `inputs/` → rejected with `LeakGuard`.
    fn apply_leak_guard(&mut self) -> Result<(), AppError> {
        // Canonicalise input (serves as existence check + symlink resolution).
        self.input = self
            .input
            .canonicalize()
            .map_err(|_| AppError::MissingInput)?;

        // Find repo root by walking up from the current working directory.
        let cwd = std::env::current_dir()?;
        let repo_root = find_repo_root(&cwd).ok_or_else(|| {
            AppError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not find repository root — no Cargo.toml or .git/ found",
            ))
        })?;

        // Canonicalise repo root for consistent path comparison.
        let repo_root = repo_root.canonicalize()?;
        self.repo_root = repo_root.clone();

        // Derive inputs/ path WITHOUT canonicalising (may not exist yet).
        let inputs_dir = repo_root.join("inputs");

        // Check: input INSIDE repo → must be in inputs/.
        if self.input.starts_with(&repo_root) && !self.input.starts_with(&inputs_dir) {
            return Err(AppError::LeakGuard(self.input.display().to_string()));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Walk upward from `start` looking for a `Cargo.toml` or `.git` directory.
fn find_repo_root(start: &std::path::Path) -> Option<PathBuf> {
    let mut current = Some(start.to_path_buf());
    while let Some(dir) = current {
        if dir.join("Cargo.toml").exists() || dir.join(".git").exists() {
            return Some(dir);
        }
        current = dir.parent().map(|p| p.to_path_buf());
    }
    None
}

/// Check whether the `claude` CLI binary is available on PATH.
///
/// Uses [`resolve_claude_bin`] so the lookup honors the Windows `.cmd`/`.exe`
/// shim names that Rust's `Command` does not resolve via `PATHEXT`.
fn claude_cli_on_path() -> bool {
    std::process::Command::new(resolve_claude_bin())
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ValueEnum;
    use std::sync::Mutex;

    /// Serialise the env-var set/remove window inside `build_test_config` so
    /// parallel tests don't clobber each other's `DEEPSEEK_API_KEY`.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // -- CriticStyle --

    #[test]
    fn test_critic_style_display() {
        assert_eq!(CriticStyle::Aggressive.to_string(), "aggressive");
        assert_eq!(
            CriticStyle::PassiveAggressive.to_string(),
            "passive-aggressive"
        );
        assert_eq!(CriticStyle::Theatrical.to_string(), "theatrical");
        assert_eq!(CriticStyle::AcademicSnob.to_string(), "academic-snob");
        assert_eq!(CriticStyle::Disappointed.to_string(), "disappointed");
        assert_eq!(CriticStyle::Random.to_string(), "random");
    }

    #[test]
    fn test_critic_style_from_str() {
        assert_eq!(
            "aggressive".parse::<CriticStyle>().unwrap(),
            CriticStyle::Aggressive
        );
        assert_eq!(
            "passive-aggressive".parse::<CriticStyle>().unwrap(),
            CriticStyle::PassiveAggressive
        );
        assert_eq!(
            "PassiveAggressive".parse::<CriticStyle>().unwrap(),
            CriticStyle::PassiveAggressive
        );
        assert_eq!(
            "theatrical".parse::<CriticStyle>().unwrap(),
            CriticStyle::Theatrical
        );
        assert_eq!(
            "academic-snob".parse::<CriticStyle>().unwrap(),
            CriticStyle::AcademicSnob
        );
        assert_eq!(
            "disappointed".parse::<CriticStyle>().unwrap(),
            CriticStyle::Disappointed
        );
        assert_eq!(
            "random".parse::<CriticStyle>().unwrap(),
            CriticStyle::Random
        );
    }

    #[test]
    fn test_critic_style_from_str_invalid() {
        assert!("banana".parse::<CriticStyle>().is_err());
    }

    #[test]
    fn test_critic_style_value_variants() {
        let variants = CriticStyle::value_variants();
        assert_eq!(variants.len(), 6);
        assert!(variants.contains(&CriticStyle::Aggressive));
        assert!(variants.contains(&CriticStyle::Random));
    }

    // -- InfernoTask --

    #[test]
    fn test_inferno_task_default_is_writing() {
        assert_eq!(InfernoTask::default(), InfernoTask::Writing);
    }

    #[test]
    fn test_inferno_task_display() {
        assert_eq!(InfernoTask::Writing.to_string(), "writing");
        assert_eq!(InfernoTask::Code.to_string(), "code");
        assert_eq!(InfernoTask::Research.to_string(), "research");
        assert_eq!(InfernoTask::Analysis.to_string(), "analysis");
        assert_eq!(InfernoTask::Prompt.to_string(), "prompt");
    }

    #[test]
    fn test_inferno_task_value_variants() {
        let variants = InfernoTask::value_variants();
        assert_eq!(variants.len(), 5);
        assert!(variants.contains(&InfernoTask::Writing));
        assert!(variants.contains(&InfernoTask::Prompt));
    }

    #[test]
    fn test_inferno_task_value_enum_each_variant_parses() {
        for (name, expected) in [
            ("writing", InfernoTask::Writing),
            ("code", InfernoTask::Code),
            ("research", InfernoTask::Research),
            ("analysis", InfernoTask::Analysis),
            ("prompt", InfernoTask::Prompt),
        ] {
            let parsed = InfernoTask::from_str(name, true).expect("variant should parse");
            assert_eq!(parsed, expected, "{name} should parse to {expected:?}");
        }
    }

    // -- Speed --

    #[test]
    fn test_speed_default_is_normal() {
        assert_eq!(Speed::default(), Speed::Normal);
    }

    #[test]
    fn test_speed_cps_mapping() {
        assert_eq!(Speed::Slow.cps(), 20);
        assert_eq!(Speed::Normal.cps(), 40);
        assert_eq!(Speed::Fast.cps(), 80);
    }

    #[test]
    fn test_speed_display() {
        assert_eq!(Speed::Slow.to_string(), "slow");
        assert_eq!(Speed::Normal.to_string(), "normal");
        assert_eq!(Speed::Fast.to_string(), "fast");
    }

    #[test]
    fn test_speed_value_variants() {
        let variants = Speed::value_variants();
        assert_eq!(variants.len(), 3);
        assert!(variants.contains(&Speed::Slow));
        assert!(variants.contains(&Speed::Normal));
        assert!(variants.contains(&Speed::Fast));
    }

    #[test]
    fn test_speed_value_enum_each_variant_parses() {
        for (name, expected) in [
            ("slow", Speed::Slow),
            ("normal", Speed::Normal),
            ("fast", Speed::Fast),
        ] {
            let parsed = Speed::from_str(name, true).expect("variant should parse");
            assert_eq!(parsed, expected, "{name} should parse to {expected:?}");
        }
    }

    #[test]
    fn test_build_default_speed_is_normal() {
        let cfg = build_test_config(|_| {}).expect("build should succeed");
        assert_eq!(cfg.speed, Speed::Normal);
        assert_eq!(cfg.reveal_cps(), 40);
    }

    #[test]
    fn test_build_speed_override() {
        let cfg = build_test_config(|c| c.speed = Some(Speed::Fast)).expect("build should succeed");
        assert_eq!(cfg.speed, Speed::Fast);
        assert_eq!(cfg.reveal_cps(), 80);
    }

    #[test]
    fn test_cli_parses_each_speed_variant() {
        use clap::Parser;
        for (flag, expected) in [
            ("slow", Speed::Slow),
            ("normal", Speed::Normal),
            ("fast", Speed::Fast),
        ] {
            let cli = CliArgs::try_parse_from([
                "agentic-inferno",
                "--writer-model",
                "deepseek-reasoner",
                "--speed",
                flag,
            ])
            .expect("--speed should parse");
            assert_eq!(cli.speed, Some(expected), "--speed {flag}");
        }
    }

    #[test]
    fn test_cli_speed_default_is_none_when_omitted() {
        use clap::Parser;
        let cli =
            CliArgs::try_parse_from(["agentic-inferno", "--writer-model", "deepseek-reasoner"])
                .expect("should parse without --speed");
        // Omitted on the CLI → None; Config::build() then applies Speed::Normal.
        assert_eq!(cli.speed, None);
    }

    #[test]
    fn test_cli_rejects_invalid_speed() {
        use clap::Parser;
        let result = CliArgs::try_parse_from([
            "agentic-inferno",
            "--writer-model",
            "deepseek-reasoner",
            "--speed",
            "ludicrous",
        ]);
        assert!(result.is_err(), "invalid --speed value should be rejected");
    }

    // -- TomlConfig --

    #[test]
    fn test_toml_config_parse_full() {
        let toml_str = r#"
writer_model = "deepseek-reasoner"
critic_model = "deepseek-chat"
critic_style = "AcademicSnob"
input = "/some/path"
max_cost_usd = 5.0
temperature = 0.9
max_tokens = 4096
timeout_secs = 60
openai_base_url = "https://custom.openai.com"
"#;
        let tc: TomlConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(tc.writer_model.as_deref(), Some("deepseek-reasoner"));
        assert_eq!(tc.critic_model.as_deref(), Some("deepseek-chat"));
        assert_eq!(tc.critic_style.as_deref(), Some("AcademicSnob"));
        assert_eq!(tc.max_cost_usd, Some(5.0));
        assert_eq!(tc.temperature, Some(0.9));
        assert_eq!(tc.max_tokens, Some(4096));
        assert_eq!(tc.timeout_secs, Some(60));
        assert_eq!(
            tc.openai_base_url.as_deref(),
            Some("https://custom.openai.com")
        );
        assert!(tc.deepseek_base_url.is_none());
        assert!(tc.moonshot_base_url.is_none());
    }

    #[test]
    fn test_toml_config_parse_empty() {
        let tc: TomlConfig = toml::from_str("").unwrap();
        assert!(tc.writer_model.is_none());
        assert!(tc.critic_model.is_none());
    }

    // -- Config::build() validation --

    #[test]
    fn test_validate_max_cost_usd_zero() {
        let cfg = build_test_config(|c| c.max_cost_usd = Some(-1.0));
        assert!(cfg.is_err());
        assert!(cfg.unwrap_err().to_string().contains("max-cost-usd"));
    }

    #[test]
    fn test_validate_temperature_out_of_range() {
        let cfg = build_test_config(|c| c.temperature = Some(2.5));
        assert!(cfg.is_err());
        assert!(cfg.unwrap_err().to_string().contains("temperature"));
    }

    #[test]
    fn test_validate_temperature_negative() {
        let cfg = build_test_config(|c| c.temperature = Some(-0.1));
        assert!(cfg.is_err());
    }

    #[test]
    fn test_validate_max_tokens_zero() {
        let cfg = build_test_config(|c| c.max_tokens = Some(0));
        assert!(cfg.is_err());
    }

    #[test]
    fn test_validate_timeout_secs_zero() {
        let cfg = build_test_config(|c| c.timeout_secs = Some(0));
        assert!(cfg.is_err());
    }

    // -- Task / prompt mode build behaviour --

    #[test]
    fn test_build_default_task_is_writing() {
        let cfg = build_test_config(|_| {}).expect("build should succeed");
        assert_eq!(cfg.task, InfernoTask::Writing);
        assert!(cfg.prompt.is_none());
    }

    #[test]
    fn test_build_explicit_task_analysis() {
        let cfg = build_test_config(|c| c.task = Some(InfernoTask::Analysis))
            .expect("build should succeed");
        assert_eq!(cfg.task, InfernoTask::Analysis);
    }

    #[test]
    fn test_build_prompt_implies_prompt_task_without_input() {
        // No --input, no --task, just --prompt: should select Prompt task and
        // build successfully despite the missing input file.
        let cfg = build_test_config(|c| {
            c.input = None;
            c.prompt = Some("prove that 1 equals 2".into());
        })
        .expect("prompt mode should build without --input");
        assert_eq!(cfg.task, InfernoTask::Prompt);
        assert_eq!(cfg.prompt.as_deref(), Some("prove that 1 equals 2"));
    }

    #[test]
    fn test_build_explicit_task_wins_over_prompt_implication() {
        // Explicit --task code with a --prompt present: explicit task wins.
        let cfg = build_test_config(|c| {
            c.task = Some(InfernoTask::Code);
            c.prompt = Some("ignored implication".into());
        })
        .expect("build should succeed (input still present)");
        assert_eq!(cfg.task, InfernoTask::Code);
    }

    #[test]
    fn test_build_prompt_task_without_prompt_text_errors() {
        let err = build_test_config(|c| {
            c.input = None;
            c.task = Some(InfernoTask::Prompt);
            c.prompt = None;
        })
        .unwrap_err();
        assert!(matches!(&err, AppError::Validation(_)), "got {err:?}");
        assert!(err.to_string().contains("--prompt"), "error: {err}");
    }

    #[test]
    fn test_build_non_prompt_task_without_input_errors() {
        let err = build_test_config(|c| {
            c.input = None;
            c.task = Some(InfernoTask::Analysis);
        })
        .unwrap_err();
        assert!(matches!(&err, AppError::MissingInput), "got {err:?}");
    }

    #[test]
    fn test_build_default_task_without_input_errors() {
        // Default (Writing) task with no input and no prompt must require input.
        let err = build_test_config(|c| c.input = None).unwrap_err();
        assert!(matches!(&err, AppError::MissingInput), "got {err:?}");
    }

    #[test]
    fn test_find_repo_root() {
        // The test is running inside the project — should find it.
        let cwd = std::env::current_dir().unwrap();
        let root = find_repo_root(&cwd);
        assert!(root.is_some(), "Should find repo root from cwd");
        let root = root.unwrap();
        assert!(root.join("Cargo.toml").exists() || root.join(".git").exists());
    }

    // Build a minimal Config with just enough to pass numerical validation.
    // Creates a temp file outside any repo so the leak guard passes.
    // Sets dummy env vars for API keys to satisfy the key check.
    fn build_test_config<F>(modify_cli: F) -> Result<Config, AppError>
    where
        F: FnOnce(&mut CliArgs),
    {
        // Hold the lock across the whole set→build→remove window so parallel
        // tests can't remove the key mid-validation. Recover from poisoning so
        // a panic in one test doesn't cascade into spurious failures elsewhere.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let tmp_dir = tempfile::TempDir::new().expect("failed to create temp dir");
        let input_path = tmp_dir.path().join("test-input.txt");
        std::fs::write(&input_path, "test content").expect("failed to write test input");

        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY", "sk-test-fake-key-for-testing");
        }

        let mut cli = CliArgs {
            writer_model: "deepseek-reasoner".into(),
            critic_model: Some("deepseek-chat".into()),
            input: Some(input_path),
            task: None,
            prompt: None,
            max_cost_usd: Some(1.0),
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
        modify_cli(&mut cli);

        let result = Config::build(cli, None);

        unsafe {
            std::env::remove_var("DEEPSEEK_API_KEY");
        }

        // Keep tmp_dir alive until after Config::build() canonicalises the path.
        let _ = tmp_dir;

        result
    }
}
