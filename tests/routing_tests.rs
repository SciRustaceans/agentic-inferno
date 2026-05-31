//! Table-driven tests for `detect_provider` — provider routing from model names.
//!
//! Covers:
//! - All known prefix patterns → correct `Provider` variant
//! - Bare aliases (`opus`, `sonnet`, `haiku`) → `Provider::Anthropic`
//! - `o1` / `o3` / `o4-mini` → `Provider::OpenAi` (ensures `opus` not caught)
//! - Unknown models → `Err(AppError::UnknownModel)`

use agentic_inferno::providers::{detect_provider, Provider};

#[test]
fn test_routing_table() {
    /// A single test-case entry: `(model_name, agent_name, expected_result)`.
    struct Case {
        model: &'static str,
        agent: &'static str,
        expected: Result<(Provider, &'static str), &'static str>,
    }

    let cases = &[
        // ── Anthropic: claude- prefix ────────────────────────────
        Case {
            model: "claude-sonnet-4-20250514",
            agent: "Writer",
            expected: Ok((Provider::Anthropic, "claude-sonnet-4-20250514")),
        },
        Case {
            model: "claude-3-5-sonnet-20241022",
            agent: "Critic",
            expected: Ok((Provider::Anthropic, "claude-3-5-sonnet-20241022")),
        },
        Case {
            model: "claude-opus-4-20250514",
            agent: "Writer",
            expected: Ok((Provider::Anthropic, "claude-opus-4-20250514")),
        },
        Case {
            model: "claude-haiku-3-5-20241022",
            agent: "Writer",
            expected: Ok((Provider::Anthropic, "claude-haiku-3-5-20241022")),
        },
        // ── Anthropic: bare aliases ──────────────────────────────
        Case {
            model: "opus",
            agent: "Writer",
            expected: Ok((Provider::Anthropic, "opus")),
        },
        Case {
            model: "sonnet",
            agent: "Writer",
            expected: Ok((Provider::Anthropic, "sonnet")),
        },
        Case {
            model: "haiku",
            agent: "Writer",
            expected: Ok((Provider::Anthropic, "haiku")),
        },
        // ── Case-insensitive Anthropic ───────────────────────────
        Case {
            model: "CLAUDE-SONNET-4",
            agent: "Writer",
            expected: Ok((Provider::Anthropic, "claude-sonnet-4")),
        },
        Case {
            model: "Opus",
            agent: "Writer",
            expected: Ok((Provider::Anthropic, "opus")),
        },
        Case {
            model: "SoNnEt",
            agent: "Writer",
            expected: Ok((Provider::Anthropic, "sonnet")),
        },
        // ── OpenAI: gpt- prefix ──────────────────────────────────
        Case {
            model: "gpt-4o",
            agent: "Critic",
            expected: Ok((Provider::OpenAi, "gpt-4o")),
        },
        Case {
            model: "gpt-4-turbo",
            agent: "Writer",
            expected: Ok((Provider::OpenAi, "gpt-4-turbo")),
        },
        Case {
            model: "gpt-3.5-turbo",
            agent: "Writer",
            expected: Ok((Provider::OpenAi, "gpt-3.5-turbo")),
        },
        Case {
            model: "gpt-4.1",
            agent: "Writer",
            expected: Ok((Provider::OpenAi, "gpt-4.1")),
        },
        // ── OpenAI: o<digit>* models ─────────────────────────────
        Case {
            model: "o1",
            agent: "Writer",
            expected: Ok((Provider::OpenAi, "o1")),
        },
        Case {
            model: "o3",
            agent: "Writer",
            expected: Ok((Provider::OpenAi, "o3")),
        },
        Case {
            model: "o4-mini",
            agent: "Writer",
            expected: Ok((Provider::OpenAi, "o4-mini")),
        },
        Case {
            model: "o3-mini",
            agent: "Writer",
            expected: Ok((Provider::OpenAi, "o3-mini")),
        },
        Case {
            model: "o1-pro",
            agent: "Critic",
            expected: Ok((Provider::OpenAi, "o1-pro")),
        },
        // ── DeepSeek ─────────────────────────────────────────────
        Case {
            model: "deepseek-chat",
            agent: "Writer",
            expected: Ok((Provider::DeepSeek, "deepseek-chat")),
        },
        Case {
            model: "deepseek-reasoner",
            agent: "Writer",
            expected: Ok((Provider::DeepSeek, "deepseek-reasoner")),
        },
        Case {
            model: "deepseek-v3",
            agent: "Writer",
            expected: Ok((Provider::DeepSeek, "deepseek-v3")),
        },
        // ── Moonshot ─────────────────────────────────────────────
        Case {
            model: "moonshot-v1-8k",
            agent: "Writer",
            expected: Ok((Provider::Moonshot, "moonshot-v1-8k")),
        },
        Case {
            model: "kimi-v1",
            agent: "Writer",
            expected: Ok((Provider::Moonshot, "kimi-v1")),
        },
        Case {
            model: "moonshot-v1",
            agent: "Writer",
            expected: Ok((Provider::Moonshot, "moonshot-v1")),
        },
        // ── Edge: bare 'o' is unknown (no digit follows) ────────
        Case {
            model: "o",
            agent: "Writer",
            expected: Err("should fail — bare 'o' has no digit"),
        },
        // ── Edge: 'opus' is Anthropic NOT OpenAI ─────────────────
        Case {
            model: "opus",
            agent: "Writer",
            expected: Ok((Provider::Anthropic, "opus")),
        },
        // ── Unknown models → hard error ─────────────────────────
        Case {
            model: "unknown-model",
            agent: "Writer",
            expected: Err("unknown model"),
        },
        Case {
            model: "not-a-model",
            agent: "Critic",
            expected: Err("unknown model"),
        },
        Case {
            model: "llama-3-70b",
            agent: "Writer",
            expected: Err("unknown model"),
        },
        Case {
            model: "gemini-1.5-pro",
            agent: "Writer",
            expected: Err("unknown model"),
        },
        Case {
            model: "",
            agent: "Writer",
            expected: Err("empty model name"),
        },
    ];

    for (i, case) in cases.iter().enumerate() {
        let result = detect_provider(case.model, case.agent);

        match (&case.expected, &result) {
            (Ok((expected_provider, expected_name)), Ok((actual_provider, actual_name))) => {
                assert_eq!(
                    *actual_provider, *expected_provider,
                    "[case {i}] model='{}', agent='{}': provider mismatch",
                    case.model, case.agent,
                );
                assert_eq!(
                    actual_name, expected_name,
                    "[case {i}] model='{}', agent='{}': normalized name mismatch",
                    case.model, case.agent,
                );
            }
            (Err(_), Err(_)) => {
                // Both expected and actual are errors — that's the correct outcome.
            }
            (Ok(_), Err(e)) => {
                panic!(
                    "[case {i}] model='{}', agent='{}': expected Ok but got Err: {e}",
                    case.model, case.agent,
                );
            }
            (Err(_reason), Ok((provider, name))) => {
                panic!(
                    "[case {i}] model='{}', agent='{}': expected Err but got Ok(({provider:?}, \"{name}\"))",
                    case.model, case.agent,
                );
            }
        }
    }
}

#[test]
fn test_opus_is_anthropic_in_table() {
    // Redundant with table but explicit assertion per task requirement.
    let (provider, _) = detect_provider("opus", "Writer").unwrap();
    assert_eq!(
        provider,
        Provider::Anthropic,
        "'opus' must be Anthropic, not OpenAI"
    );
}

#[test]
fn test_o1_mini_is_openai() {
    let (provider, name) = detect_provider("o1-mini", "Writer").unwrap();
    assert_eq!(provider, Provider::OpenAi);
    assert_eq!(name, "o1-mini");
}

#[test]
fn test_unknown_model_includes_agent_in_error() {
    let err = detect_provider("no-such-model", "Critic").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("no-such-model"),
        "error must include model name"
    );
    assert!(msg.contains("Critic"), "error must include agent name");
}
