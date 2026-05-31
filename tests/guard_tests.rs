use agentic_inferno::error::AppError;
use agentic_inferno::guards::{semantic_hash, ContextWindow, CostCeiling, LoopDetection};

// ── CostCeiling ───────────────────────────────────────────────────

#[test]
fn test_cost_ceiling_spend_within_limit() {
    let c = CostCeiling::new(10.0);
    // Spend exactly at limit → ok
    assert!(c.record(10.0).is_ok());
    assert_eq!(c.spent(), 10.0);
}

#[test]
fn test_cost_ceiling_spend_over_limit() {
    let c = CostCeiling::new(5.0);
    let err = c.record(5.01).unwrap_err();
    assert!(
        matches!(err, AppError::CostCeilingExceeded(5.01, 5.0)),
        "expected CostCeilingExceeded(5.01, 5.0), got {err:?}"
    );
}

#[test]
fn test_cost_ceiling_only_successful_calls_count() {
    let c = CostCeiling::new(5.0);
    // First call succeeds
    assert!(c.record(3.0).is_ok());
    assert_eq!(c.spent(), 3.0);

    // Second call would exceed → not recorded
    let err = c.record(3.0).unwrap_err();
    assert!(matches!(err, AppError::CostCeilingExceeded(6.0, 5.0)));
    // Spend unchanged
    assert_eq!(c.spent(), 3.0);
}

// ── LoopDetection ────────────────────────────────────────────────

#[test]
fn test_loop_detection_three_identical_in_window_five() {
    let ld = LoopDetection::new(5, 3);
    // Push 3 identical hashes → should trigger
    assert!(ld.check(42).is_ok());
    assert!(ld.check(42).is_ok());
    let err = ld.check(42).unwrap_err();
    assert!(matches!(err, AppError::LoopExhausted(_)));
}

#[test]
fn test_loop_detection_two_identical_no_trigger() {
    let ld = LoopDetection::new(5, 3);
    // Only 2 identical hashes in window → less than min_repeats=3
    assert!(ld.check(42).is_ok());
    assert!(ld.check(42).is_ok());
    // Fill the rest of the window with different hashes
    assert!(ld.check(99).is_ok());
    assert!(ld.check(88).is_ok());
    assert!(ld.check(77).is_ok());
    // Still no trigger — the two 42s are within the window but count < 3
    assert!(ld.check(66).is_ok());
}

#[test]
fn test_loop_detection_rephrased_content_triggers() {
    let ld = LoopDetection::new(5, 3);
    // Different phrasings produce the same semantic hash
    let h1 = semantic_hash("I cannot complete this task");
    let h2 = semantic_hash("  i   CANNOT  complete  THIS  task  ");
    assert_eq!(h1, h2, "semantic hashes should match after normalization");

    // Three rephrased variants → loop detection should catch it
    assert!(ld.check(h1).is_ok());
    assert!(ld.check(h2).is_ok());
    assert!(ld.check(h1).is_err()); // 3rd hit → LoopExhausted
}

// ── Semantic Hash ─────────────────────────────────────────────────

#[test]
fn test_semantic_hash_whitespace_and_case_invariant() {
    let inputs = vec![
        "Hello World",
        "  hello   world  ",
        "HELLO WORLD",
        "hello\nworld\n",
        "\tHello\tWorld\t",
    ];
    let base = semantic_hash("Hello World");
    for input in &inputs {
        assert_eq!(
            semantic_hash(input),
            base,
            "semantic_hash({input:?}) should equal semantic_hash(\"Hello World\")"
        );
    }
}

// ── ContextWindow ─────────────────────────────────────────────────

#[test]
fn test_context_window_warn_at_80_percent() {
    let cw = ContextWindow::new(1000);
    // 799/1000 = 79.9% → no warning
    cw.record_usage(799);
    assert!(cw.would_exceed(0).is_none());

    // 800/1000 = 80% → warning
    cw.record_usage(1);
    let ratio = cw.would_exceed(0).unwrap();
    assert!(
        (ratio - 0.8).abs() < 1e-10,
        "expected ratio 0.8, got {ratio}"
    );

    // 850/1000 = 85% → warning still
    cw.record_usage(50);
    let ratio = cw.would_exceed(0).unwrap();
    assert!(
        (ratio - 0.85).abs() < 1e-10,
        "expected ratio 0.85, got {ratio}"
    );
}

#[test]
fn test_context_window_prune_at_90_percent() {
    let cw = ContextWindow::new(1000);
    cw.record_usage(900);
    let ratio = cw.would_exceed(0).unwrap();
    // 90% → caller should force prune
    assert!(
        (ratio - 0.9).abs() < 1e-10,
        "expected ratio 0.9, got {ratio}"
    );

    // After pruning, warning should go away
    cw.prune(500);
    assert!(
        cw.would_exceed(0).is_none(),
        "after pruning, projected ratio should be below 80%"
    );
}

#[test]
fn test_context_window_different_model_limits() {
    // Model A: 4096 token limit
    let small = ContextWindow::new(4096);
    small.record_usage(3276); // 3276/4096 ≈ 79.98%
    assert!(
        small.would_exceed(0).is_none(),
        "should not warn below 80% (model A)"
    );
    small.record_usage(5); // 3281/4096 ≈ 80.1%
    assert!(
        small.would_exceed(0).is_some(),
        "should warn at 80% (model A)"
    );

    // Model B: 128000 token limit (Claude Opus)
    let large = ContextWindow::new(128000);
    large.record_usage(102_399); // 102399/128000 ≈ 79.999%
    assert!(
        large.would_exceed(0).is_none(),
        "should not warn below 80% (model B)"
    );
    large.record_usage(1); // 102400/128000 = 80%
    assert!(
        large.would_exceed(0).is_some(),
        "should warn at 80% (model B)"
    );

    // Verify the ratio is correct
    let ratio = large.would_exceed(0).unwrap();
    assert!(
        (ratio - 0.8).abs() < 1e-10,
        "model B ratio should be 0.8, got {ratio}"
    );
}
