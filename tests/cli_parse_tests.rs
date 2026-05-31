use agentic_inferno::error::AppError;
use agentic_inferno::providers::anthropic_cli::{validate_claude_response, ClaudeCliResponse};
use agentic_inferno::providers::ChatReply;

fn parse_and_validate(json_bytes: &[u8]) -> Result<ChatReply, AppError> {
    let response: ClaudeCliResponse =
        serde_json::from_slice(json_bytes).map_err(|e| AppError::ClaudeCli {
            subtype: "parse".into(),
            message: format!("Failed to parse claude JSON output ({e})."),
        })?;
    validate_claude_response(response, "")
}

#[test]
fn test_valid_success_json_returns_chat_reply() {
    let json = br#"{
        "type": "result",
        "subtype": "success",
        "is_error": false,
        "result": "Hello from claude!",
        "total_cost_usd": 0.015
    }"#;

    let reply = parse_and_validate(json).unwrap();
    assert_eq!(reply.text, "Hello from claude!");
    assert_eq!(reply.cost_usd, Some(0.015));
}

#[test]
fn test_valid_success_no_is_error_field() {
    let json = br#"{
        "type": "result",
        "subtype": "success",
        "result": "works without is_error field",
        "total_cost_usd": 0.0
    }"#;

    let reply = parse_and_validate(json).unwrap();
    assert_eq!(reply.text, "works without is_error field");
    assert_eq!(reply.cost_usd, Some(0.0));
}

#[test]
fn test_is_error_true_returns_claude_cli_error() {
    let json = br#"{
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": true,
        "result": "Something went wrong"
    }"#;

    let err = parse_and_validate(json).unwrap_err();
    match err {
        AppError::ClaudeCli { subtype, message } => {
            assert_eq!(subtype, "error_during_execution");
            assert!(message.contains("Something went wrong"));
        }
        other => panic!("expected ClaudeCli error, got {other:?}"),
    }
}

#[test]
fn test_subtype_not_success_returns_claude_cli_error() {
    let json = br#"{
        "type": "result",
        "subtype": "rate_limited",
        "is_error": false,
        "result": "Try again later"
    }"#;

    let err = parse_and_validate(json).unwrap_err();
    match err {
        AppError::ClaudeCli { subtype, message } => {
            assert_eq!(subtype, "rate_limited");
            assert!(message.contains("Try again later"));
        }
        other => panic!("expected ClaudeCli error, got {other:?}"),
    }
}

#[test]
fn test_is_error_false_but_subtype_not_success_still_errors() {
    let json = br#"{
        "type": "result",
        "subtype": "internal_error",
        "is_error": false,
        "result": "Internal failure"
    }"#;

    let err = parse_and_validate(json).unwrap_err();
    assert!(matches!(err, AppError::ClaudeCli { .. }));
    match err {
        AppError::ClaudeCli { subtype, .. } => {
            assert_eq!(subtype, "internal_error");
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_unexpected_response_type_errors() {
    let json = br#"{
        "type": "error",
        "subtype": "success",
        "is_error": false,
        "result": "looks ok but wrong type"
    }"#;

    let err = parse_and_validate(json).unwrap_err();
    match err {
        AppError::ClaudeCli { subtype, message } => {
            assert_eq!(subtype, "unexpected_type");
            assert!(message.contains("error"));
            assert!(message.contains("result"));
        }
        other => panic!("expected ClaudeCli with unexpected_type, got {other:?}"),
    }
}

#[test]
fn test_non_json_stdout_returns_claude_cli_error() {
    let garbage = b"this is not json at all, just raw text";

    let err = parse_and_validate(garbage).unwrap_err();
    match err {
        AppError::ClaudeCli { subtype, message } => {
            assert_eq!(subtype, "parse");
            assert!(message.contains("Failed to parse"));
        }
        other => panic!("expected ClaudeCli parse error, got {other:?}"),
    }
}

#[test]
fn test_non_json_stdout_empty_bytes() {
    let err = parse_and_validate(b"").unwrap_err();
    assert!(matches!(err, AppError::ClaudeCli { .. }));
}

#[test]
fn test_non_json_stdout_partial_json() {
    let partial = br#"{"type": "result", "subtype": "success""#;

    let err = parse_and_validate(partial).unwrap_err();
    assert!(matches!(err, AppError::ClaudeCli { .. }));
}

#[test]
fn test_non_zero_exit_maps_to_claude_cli() {
    let err = AppError::ClaudeCli {
        subtype: "exit".into(),
        message: "claude CLI exited with exit status: 1. stderr: some error".into(),
    };

    let msg = err.to_string();
    assert!(
        msg.contains("exit"),
        "error message should mention exit: {msg}"
    );
    assert!(
        msg.contains("some error"),
        "error message should include stderr: {msg}"
    );
}

#[test]
fn test_missing_result_text_errors() {
    let json = br#"{
        "type": "result",
        "subtype": "success",
        "is_error": false
    }"#;

    let err = parse_and_validate(json).unwrap_err();
    match err {
        AppError::ClaudeCli { subtype, message } => {
            assert_eq!(subtype, "empty");
            assert!(message.contains("no result text"));
        }
        other => panic!("expected ClaudeCli empty error, got {other:?}"),
    }
}

#[test]
fn test_error_with_no_result_message() {
    let json = br#"{
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": true
    }"#;

    let err = parse_and_validate(json).unwrap_err();
    match err {
        AppError::ClaudeCli { subtype, message } => {
            assert_eq!(subtype, "error_during_execution");
            assert!(message.contains("no message"));
        }
        other => panic!("expected ClaudeCli error, got {other:?}"),
    }
}
