use crate::error::AppError;
use crate::providers::{ChatReply, ChatRequest};
use std::time::Duration;

/// Maximum number of retries for transient server-side errors (429 / 5xx).
const MAX_RETRIES: u32 = 2;

/// Core HTTP call for OpenAI-compatible providers.
///
/// POSTs to `{base_url}/chat/completions` with Bearer auth.  Handles retry
/// logic for 429 (rate limit) and 5xx (server error) with exponential backoff
/// (1 s, 2 s; max 3 total attempts).  Never retries auth failures (401/403),
/// client errors (4xx), network errors, or timeouts.
///
/// If the response JSON contains a `total_cost_usd` field, its value is
/// extracted into `ChatReply.cost_usd`.  `NaN` values are treated as absent.
pub(crate) async fn do_complete(
    base_url: &reqwest::Url,
    api_key: &str,
    model: &str,
    http: &reqwest::Client,
    req: &ChatRequest,
) -> Result<ChatReply, AppError> {
    let url = base_url
        .join("chat/completions")
        .expect("base_url must be a valid base that supports path joins");

    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.user},
        ],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    });

    let mut attempt: u32 = 0;

    loop {
        let response_result = http
            .post(url.clone())
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await;

        match response_result {
            Ok(response) => {
                let status = response.status();
                let status_code = status.as_u16();

                // ── Success ──────────────────────────────────────
                if status.is_success() {
                    return parse_success_response(response).await;
                }

                // ── 401 / 403: auth failure — never retry ────────
                if status_code == 401 || status_code == 403 {
                    let body_text = response.text().await.unwrap_or_default();
                    return Err(AppError::MissingKey(format!(
                        "API key rejected (HTTP {}): {}",
                        status_code, body_text,
                    )));
                }

                // ── 429 / 5xx: transient — retry with backoff ────
                if (status_code == 429 || status_code >= 500) && attempt < MAX_RETRIES {
                    attempt += 1;
                    let delay_secs = 1u64 << (attempt - 1); // 1 s, 2 s
                    tokio::time::sleep(Duration::from_secs(delay_secs)).await;
                    continue;
                }

                // ── Everything else: hard error ──────────────────
                let body_text = response.text().await.unwrap_or_default();
                return Err(AppError::Http {
                    status: status_code,
                    body: body_text,
                });
            }

            // ── Transport / network error ──────────────────────────
            Err(e) => {
                // Timeout is a distinct error variant — do NOT retry.
                if e.is_timeout() {
                    return Err(AppError::Timeout);
                }
                return Err(AppError::Network(e));
            }
        }
    }
}

/// Parse a successful (2xx) JSON response body into a `ChatReply`.
///
/// Extracts `choices[0].message.content` as the reply text.
/// Extracts `total_cost_usd` (if present and non-NaN) as the cost.
async fn parse_success_response(response: reqwest::Response) -> Result<ChatReply, AppError> {
    let status_code = response.status().as_u16();

    // Buffer the full body so we can return it verbatim on parse failure.
    let body_bytes = response.bytes().await.map_err(AppError::Network)?;

    let full_body: serde_json::Value =
        serde_json::from_slice(&body_bytes).map_err(|_| AppError::Http {
            status: status_code,
            body: String::from_utf8_lossy(&body_bytes).into_owned(),
        })?;

    // Extract the assistant message content.
    let content = full_body["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| AppError::Http {
            status: status_code,
            body: String::from_utf8_lossy(&body_bytes).into_owned(),
        })?
        .to_string();

    // Extract cost (if present and non-NaN).
    let cost_usd = full_body
        .get("total_cost_usd")
        .and_then(|v| v.as_f64())
        .filter(|c| !c.is_nan());

    // Extract total token usage if the provider reported it.
    let tokens = full_body["usage"]["total_tokens"].as_u64();

    Ok(ChatReply {
        text: content,
        cost_usd,
        tokens,
    })
}

// ── Tests ──────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // We test the parse helper directly to avoid needing a live HTTP server.
    // The retry / timeout / auth paths are integration-tested separately.

    fn fake_response(body: serde_json::Value) -> reqwest::Response {
        let body_str = serde_json::to_string(&body).unwrap();
        reqwest::Response::from(
            http::Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(body_str)
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn parse_success_simple() {
        let resp = fake_response(json!({
            "choices": [{"message": {"content": "Hello, world!"}}],
        }));
        let reply = parse_success_response(resp).await.unwrap();
        assert_eq!(reply.text, "Hello, world!");
        assert!(reply.cost_usd.is_none());
    }

    #[tokio::test]
    async fn parse_success_with_cost() {
        let resp = fake_response(json!({
            "choices": [{"message": {"content": "Hi"}}],
            "total_cost_usd": 0.0042,
        }));
        let reply = parse_success_response(resp).await.unwrap();
        assert_eq!(reply.text, "Hi");
        assert_eq!(reply.cost_usd, Some(0.0042));
    }

    #[tokio::test]
    async fn parse_success_with_usage_tokens() {
        let resp = fake_response(json!({
            "choices": [{"message": {"content": "Hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }));
        let reply = parse_success_response(resp).await.unwrap();
        assert_eq!(reply.text, "Hi");
        assert_eq!(reply.tokens, Some(15));
    }

    #[tokio::test]
    async fn parse_success_absent_usage_tokens_is_none() {
        let resp = fake_response(json!({
            "choices": [{"message": {"content": "Hi"}}],
        }));
        let reply = parse_success_response(resp).await.unwrap();
        assert!(reply.tokens.is_none());
    }

    #[tokio::test]
    async fn parse_nan_cost_becomes_none() {
        let resp = fake_response(json!({
            "choices": [{"message": {"content": "Hi"}}],
            "total_cost_usd": f64::NAN,
        }));
        let reply = parse_success_response(resp).await.unwrap();
        assert_eq!(reply.text, "Hi");
        assert!(
            reply.cost_usd.is_none(),
            "NaN cost must be filtered to None"
        );
    }

    #[tokio::test]
    async fn parse_cost_null_becomes_none() {
        let resp = fake_response(json!({
            "choices": [{"message": {"content": "Hi"}}],
            "total_cost_usd": null,
        }));
        let reply = parse_success_response(resp).await.unwrap();
        assert_eq!(reply.text, "Hi");
        assert!(reply.cost_usd.is_none());
    }

    #[tokio::test]
    async fn parse_missing_content_is_error() {
        let resp = fake_response(json!({
            "choices": [{"message": {}}],
        }));
        let err = parse_success_response(resp).await.unwrap_err();
        assert!(matches!(err, AppError::Http { status: 200, .. }));
    }

    #[tokio::test]
    async fn parse_empty_choices_is_error() {
        let resp = fake_response(json!({
            "choices": [],
        }));
        let err = parse_success_response(resp).await.unwrap_err();
        assert!(matches!(err, AppError::Http { status: 200, .. }));
    }

    #[tokio::test]
    async fn parse_malformed_json_is_http_error() {
        // Build a raw response with invalid JSON.
        let resp = reqwest::Response::from(
            http::Response::builder()
                .status(200)
                .body("not json at all")
                .unwrap(),
        );
        let err = parse_success_response(resp).await.unwrap_err();
        assert!(matches!(err, AppError::Http { status: 200, .. }));
    }
}
