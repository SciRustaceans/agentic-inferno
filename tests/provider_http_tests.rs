use std::time::Duration;

use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use agentic_inferno::error::AppError;
use agentic_inferno::providers::{ChatReply, ChatRequest, LlmClient};

fn make_request() -> ChatRequest {
    ChatRequest {
        system: "You are a helpful assistant.".into(),
        user: "Say hello.".into(),
        model: "gpt-4o".into(),
        temperature: 0.7_f32,
        max_tokens: 1024,
    }
}

fn make_client(mock_server: &MockServer, timeout_secs: u64) -> LlmClient {
    LlmClient::OpenAiCompat {
        base_url: mock_server.uri().parse().expect("invalid mock server URI"),
        api_key: "sk-test-key".into(),
        model: "gpt-4o".into(),
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .expect("failed to build reqwest client"),
    }
}

// ── 200 success → ChatReply ──────────────────────────────────────────

#[tokio::test]
async fn test_200_success_returns_chat_reply() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"content": "Hello, world!"}}],
            })),
        )
        .mount(&mock_server)
        .await;

    let client = make_client(&mock_server, 30);
    let reply = client.complete(make_request(), 30).await.unwrap();

    assert_eq!(reply.text, "Hello, world!");
    assert!(reply.cost_usd.is_none());
}

// ── 200 with cost_usd present ────────────────────────────────────────

#[tokio::test]
async fn test_200_with_cost_usd() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"content": "Costly response"}}],
                "total_cost_usd": 0.042,
            })),
        )
        .mount(&mock_server)
        .await;

    let client = make_client(&mock_server, 30);
    let reply: ChatReply = client.complete(make_request(), 30).await.unwrap();

    assert_eq!(reply.text, "Costly response");
    assert_eq!(reply.cost_usd, Some(0.042));
}

// ── 401 → AppError::MissingKey (never retried) ───────────────────────

#[tokio::test]
async fn test_401_returns_missing_key() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid key"))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = make_client(&mock_server, 30);
    let err = client.complete(make_request(), 30).await.unwrap_err();

    match err {
        AppError::MissingKey(msg) => {
            assert!(msg.contains("401"), "MissingKey message should mention 401: {msg}");
            assert!(msg.contains("invalid key"), "MissingKey message should include body: {msg}");
        }
        other => panic!("expected MissingKey, got {other:?}"),
    }
}

// ── 429 → retried twice then AppError::Http ──────────────────────────

#[tokio::test]
async fn test_429_retried_twice_then_http_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .expect(3)
        .mount(&mock_server)
        .await;

    let client = make_client(&mock_server, 30);
    let err = client.complete(make_request(), 30).await.unwrap_err();

    match err {
        AppError::Http { status, body } => {
            assert_eq!(status, 429);
            assert!(body.contains("rate limited"));
        }
        other => panic!("expected Http(429), got {other:?}"),
    }
}

// ── 5xx → AppError::Http (retried twice then fails) ──────────────────

#[tokio::test]
async fn test_500_retried_then_http_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
        .expect(3)
        .mount(&mock_server)
        .await;

    let client = make_client(&mock_server, 30);
    let err = client.complete(make_request(), 30).await.unwrap_err();

    match err {
        AppError::Http { status, body } => {
            assert_eq!(status, 500);
            assert!(body.contains("internal error"));
        }
        other => panic!("expected Http(500), got {other:?}"),
    }
}

// ── 503 (another 5xx) → retried then Http ────────────────────────────

#[tokio::test]
async fn test_503_retried_then_http_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
        .expect(3)
        .mount(&mock_server)
        .await;

    let client = make_client(&mock_server, 30);
    let err = client.complete(make_request(), 30).await.unwrap_err();

    match err {
        AppError::Http { status, .. } => assert_eq!(status, 503),
        other => panic!("expected Http(503), got {other:?}"),
    }
}

// ── Malformed JSON (200 status, non-JSON body) → AppError::Http ──────

#[tokio::test]
async fn test_malformed_json_returns_http_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json at all"))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = make_client(&mock_server, 30);
    let err = client.complete(make_request(), 30).await.unwrap_err();

    match err {
        AppError::Http { status, body } => {
            assert_eq!(status, 200);
            assert!(body.contains("not json"));
        }
        other => panic!("expected Http(200), got {other:?}"),
    }
}

// ── Malformed JSON: valid JSON but missing choices ───────────────────

#[tokio::test]
async fn test_json_missing_content_returns_http_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {}}],
            })),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = make_client(&mock_server, 30);
    let err = client.complete(make_request(), 30).await.unwrap_err();

    assert!(matches!(err, AppError::Http { status: 200, .. }));
}

// ── Timeout → AppError::Timeout ──────────────────────────────────────

#[tokio::test]
async fn test_timeout_returns_timeout_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(10))
                .set_body_json(json!({
                    "choices": [{"message": {"content": "too late"}}],
                })),
        )
        .mount(&mock_server)
        .await;

    let http = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
        .expect("failed to build reqwest client");

    let client = LlmClient::OpenAiCompat {
        base_url: mock_server.uri().parse().expect("invalid mock server URI"),
        api_key: "sk-test-key".into(),
        model: "gpt-4o".into(),
        http,
    };

    let err = client.complete(make_request(), 30).await.unwrap_err();
    assert!(matches!(err, AppError::Timeout), "expected Timeout, got {err:?}");
}

// ── Request body shape validation ────────────────────────────────────

#[tokio::test]
async fn test_request_body_has_expected_shape() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello."},
            ],
            "temperature": 1.0,
            "max_tokens": 1024,
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"content": "shape matches"}}],
            })),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = make_client(&mock_server, 30);
    let request = ChatRequest {
        system: "You are a helpful assistant.".into(),
        user: "Say hello.".into(),
        model: "gpt-4o".into(),
        temperature: 1.0_f32,
        max_tokens: 1024,
    };
    let reply = client.complete(request, 30).await.unwrap();
    assert_eq!(reply.text, "shape matches");
}

#[tokio::test]
async fn test_request_body_shape_different_request() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(json!({
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Tell me a joke."},
            ],
            "temperature": 0.5,
            "max_tokens": 512,
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"content": "different request matched"}}],
            })),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = LlmClient::OpenAiCompat {
        base_url: mock_server.uri().parse().expect("invalid mock server URI"),
        api_key: "sk-test-key".into(),
        model: "deepseek-chat".into(),
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build reqwest client"),
    };

    let request = ChatRequest {
        system: "Be concise.".into(),
        user: "Tell me a joke.".into(),
        model: "deepseek-chat".into(),
        temperature: 0.5_f32,
        max_tokens: 512,
    };

    let reply = client.complete(request, 30).await.unwrap();
    assert_eq!(reply.text, "different request matched");
}

// ── Bearer header present ────────────────────────────────────────────

#[tokio::test]
async fn test_bearer_header_present() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", "Bearer sk-test-key"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"content": "header ok"}}],
            })),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = make_client(&mock_server, 30);
    let reply = client.complete(make_request(), 30).await.unwrap();
    assert_eq!(reply.text, "header ok");
}

#[tokio::test]
async fn test_wrong_bearer_header_not_accepted() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", "Bearer wrong-key"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"content": "should not match"}}],
            })),
        )
        .expect(0)
        .mount(&mock_server)
        .await;

    let client = make_client(&mock_server, 30);

    let result = client.complete(make_request(), 30).await;
    assert!(result.is_err(), "wrong bearer header should not match any mock");
}
