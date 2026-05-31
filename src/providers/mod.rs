pub mod openai_compat;

pub mod anthropic_cli;

use crate::error::AppError;
use std::fmt;

/// Supported LLM providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAi,
    DeepSeek,
    Moonshot,
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Provider::Anthropic => write!(f, "Anthropic"),
            Provider::OpenAi => write!(f, "OpenAI"),
            Provider::DeepSeek => write!(f, "DeepSeek"),
            Provider::Moonshot => write!(f, "Moonshot"),
        }
    }
}

impl Provider {
    /// The environment variable that holds the API key for this provider.
    pub fn api_key_env_var(&self) -> &'static str {
        match self {
            Provider::Anthropic => "ANTHROPIC_API_KEY",
            Provider::OpenAi => "OPENAI_API_KEY",
            Provider::DeepSeek => "DEEPSEEK_API_KEY",
            Provider::Moonshot => "MOONSHOT_API_KEY",
        }
    }

    /// The default base URL for this provider's API.
    /// Returns `None` for Anthropic since it uses the `claude` CLI.
    pub fn default_base_url(&self) -> Option<&'static str> {
        match self {
            Provider::Anthropic => None,
            Provider::OpenAi => Some("https://api.openai.com/v1"),
            Provider::DeepSeek => Some("https://api.deepseek.com/v1"),
            Provider::Moonshot => Some("https://api.moonshot.ai/v1"),
        }
    }

    /// The environment variable that can override this provider's base URL.
    /// Returns `None` for Anthropic (routing handled by `claude` CLI).
    pub fn base_url_env_var(&self) -> Option<&'static str> {
        match self {
            Provider::Anthropic => None,
            Provider::OpenAi => Some("OPENAI_BASE_URL"),
            Provider::DeepSeek => Some("DEEPSEEK_BASE_URL"),
            Provider::Moonshot => Some("MOONSHOT_BASE_URL"),
        }
    }
}

/// Detect which provider a model name belongs to.
///
/// Matching is order-sensitive — Anthropic patterns are checked first
/// so that `opus` is correctly identified as Anthropic rather than
/// being mis-caught by the OpenAI `o<digit>` rule.
///
/// Returns `Ok((Provider, normalized_model_name))` on success, or
/// `AppError::UnknownModel` if no provider matches.
pub fn detect_provider(model: &str, agent_name: &str) -> Result<(Provider, String), AppError> {
    let lowered = model.to_ascii_lowercase();

    // --- Anthropic patterns (checked first — `opus` is Anthropic) ---
    if lowered.starts_with("claude-")
        || lowered == "opus"
        || lowered == "sonnet"
        || lowered == "haiku"
    {
        return Ok((Provider::Anthropic, lowered));
    }

    // --- OpenAI patterns: `gpt-*` or `o` followed by a digit ---
    if lowered.starts_with("gpt-")
        || (lowered.starts_with('o')
            && lowered.len() > 1
            && lowered.as_bytes()[1].is_ascii_digit())
    {
        return Ok((Provider::OpenAi, lowered));
    }

    // --- DeepSeek patterns ---
    if lowered.starts_with("deepseek-") {
        return Ok((Provider::DeepSeek, lowered));
    }

    // --- Moonshot patterns ---
    if lowered.starts_with("kimi-") || lowered.starts_with("moonshot-") {
        return Ok((Provider::Moonshot, lowered));
    }

    // --- No match: hard error, never a silent default ---
    Err(AppError::UnknownModel(
        model.to_string(),
        agent_name.to_string(),
    ))
}

/// A request to chat with an LLM provider.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub system: String,
    pub user: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
}

/// A response from an LLM provider.
#[derive(Debug, Clone)]
pub struct ChatReply {
    pub text: String,
    pub cost_usd: Option<f64>,
}

/// Represents a client connected to a specific LLM provider.
///
/// This enum abstracts over the transport mechanism:
/// - `OpenAiCompat`: REST API via `reqwest` (used by OpenAI, DeepSeek, Moonshot)
/// - `AnthropicCli`: The `claude` CLI binary on the local machine
///
/// The `complete()` method (Tasks 6/7) will be implemented later.
#[derive(Debug)]
pub enum LlmClient {
    /// OpenAI-compatible REST API client.
    OpenAiCompat {
        base_url: reqwest::Url,
        api_key: String,
        model: String,
        http: reqwest::Client,
    },
    /// Local `claude` CLI binary client.
    AnthropicCli {
        model: String,
        claude_bin: String,
    },
}

impl LlmClient {
    /// Returns a human-readable provider name for display and logging.
    pub fn provider_name(&self) -> &str {
        match self {
            LlmClient::OpenAiCompat { .. } => "OpenAI-compatible",
            LlmClient::AnthropicCli { .. } => "Anthropic CLI",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Anthropic ──────────────────────────────────────────────

    #[test]
    fn test_anthropic_claude_prefix() {
        let (provider, name) = detect_provider("claude-sonnet-4-20250514", "Writer").unwrap();
        assert_eq!(provider, Provider::Anthropic);
        assert_eq!(name, "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_anthropic_opus() {
        let (provider, _) = detect_provider("opus", "Writer").unwrap();
        assert_eq!(provider, Provider::Anthropic);
    }

    #[test]
    fn test_anthropic_sonnet() {
        let (provider, _) = detect_provider("sonnet", "Writer").unwrap();
        assert_eq!(provider, Provider::Anthropic);
    }

    #[test]
    fn test_anthropic_haiku() {
        let (provider, _) = detect_provider("haiku", "Writer").unwrap();
        assert_eq!(provider, Provider::Anthropic);
    }

    // ── OpenAI ─────────────────────────────────────────────────

    #[test]
    fn test_openai_gpt_prefix() {
        let (provider, name) = detect_provider("gpt-4o", "Critic").unwrap();
        assert_eq!(provider, Provider::OpenAi);
        assert_eq!(name, "gpt-4o");
    }

    #[test]
    fn test_openai_o1() {
        let (provider, _) = detect_provider("o1", "Writer").unwrap();
        assert_eq!(provider, Provider::OpenAi);
    }

    #[test]
    fn test_openai_o3_mini() {
        let (provider, _) = detect_provider("o3-mini", "Writer").unwrap();
        assert_eq!(provider, Provider::OpenAi);
    }

    #[test]
    fn test_openai_o4_mini() {
        let (provider, _) = detect_provider("o4-mini", "Writer").unwrap();
        assert_eq!(provider, Provider::OpenAi);
    }

    // ── DeepSeek ───────────────────────────────────────────────

    #[test]
    fn test_deepseek_chat() {
        let (provider, _) = detect_provider("deepseek-chat", "Writer").unwrap();
        assert_eq!(provider, Provider::DeepSeek);
    }

    #[test]
    fn test_deepseek_reasoner() {
        let (provider, _) = detect_provider("deepseek-reasoner", "Writer").unwrap();
        assert_eq!(provider, Provider::DeepSeek);
    }

    // ── Moonshot ───────────────────────────────────────────────

    #[test]
    fn test_moonshot_v1() {
        let (provider, _) = detect_provider("moonshot-v1", "Writer").unwrap();
        assert_eq!(provider, Provider::Moonshot);
    }

    #[test]
    fn test_kimi_v1() {
        let (provider, _) = detect_provider("kimi-v1", "Writer").unwrap();
        assert_eq!(provider, Provider::Moonshot);
    }

    // ── Edge cases ─────────────────────────────────────────────

    #[test]
    fn test_unknown_model_returns_error() {
        let err = detect_provider("unknown-model", "Writer").unwrap_err();
        assert!(err.to_string().contains("unknown-model"));
        assert!(err.to_string().contains("Writer"));
    }

    #[test]
    fn test_unknown_model_never_silent() {
        let err = detect_provider("not-a-model", "Critic").unwrap_err();
        assert!(matches!(err, AppError::UnknownModel(..)));
    }

    #[test]
    fn test_case_insensitive_matching() {
        let (provider, name) = detect_provider("CLAUDE-SONNET-4", "Writer").unwrap();
        assert_eq!(provider, Provider::Anthropic);
        assert_eq!(name, "claude-sonnet-4");
    }

    #[test]
    fn test_opus_is_anthropic_not_openai() {
        // 'opus' starts with 'o' but 'p' is not a digit —
        // must be Anthropic, not OpenAI.
        let (provider, _) = detect_provider("opus", "Writer").unwrap();
        assert_eq!(provider, Provider::Anthropic);
    }

    #[test]
    fn test_bare_o_is_unknown() {
        // bare 'o' has no digit after it — must be unknown.
        let err = detect_provider("o", "Writer").unwrap_err();
        assert!(matches!(err, AppError::UnknownModel(..)));
    }

    #[test]
    fn test_openai_o_with_extra_chars_after_digit() {
        // 'o4-mini' starts with 'o' and second char is '4' (digit) → OpenAI
        let (provider, _) = detect_provider("o4-mini", "Writer").unwrap();
        assert_eq!(provider, Provider::OpenAi);
    }

    // ── LlmClient provider_name ────────────────────────────────

    #[test]
    fn test_llm_client_provider_name_openai() {
        let client = LlmClient::OpenAiCompat {
            base_url: "https://api.openai.com/v1".parse().unwrap(),
            api_key: "sk-test".into(),
            model: "gpt-4o".into(),
            http: reqwest::Client::new(),
        };
        assert_eq!(client.provider_name(), "OpenAI-compatible");
    }

    #[test]
    fn test_llm_client_provider_name_anthropic() {
        let client = LlmClient::AnthropicCli {
            model: "claude-sonnet-4-20250514".into(),
            claude_bin: "claude".into(),
        };
        assert_eq!(client.provider_name(), "Anthropic CLI");
    }
}
