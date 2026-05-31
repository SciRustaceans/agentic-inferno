//! Integration tests for apology detection and cooldown logic.
//!
//! Tests cover:
//! - `[APOLOGY]` marker parsing (case-insensitive)
//! - No marker / partial marker edge cases
//! - Harsh keyword counting and threshold detection
//! - Apology cooldown state machine (time + cycle conditions)
//! - Display/formatting of cooldown state
//!
//! These tests use pure functions and mock cooldown state — no LLM calls.

use std::time::{Duration, Instant};

use agentic_inferno::orchestrator::{
    count_harsh_keywords, cooldown_remaining_secs, find_apology_marker,
};
use agentic_inferno::state::ApologyCooldown;

// ---------------------------------------------------------------------------
// find_apology_marker
// ---------------------------------------------------------------------------

/// `[APOLOGY]` marker at the start of a line → detected at the opening `[`.
#[test]
fn test_marker_detected_in_text() {
    let text = "Here is the document text.\n[APOLOGY] I apologize for the confusion.";
    let idx = find_apology_marker(text);
    assert!(idx.is_some(), "expected [APOLOGY] marker to be found");
    let found = &text[idx.unwrap()..idx.unwrap() + 9];
    assert_eq!(found, "[APOLOGY]");
}

/// No marker present → returns None.
#[test]
fn test_no_marker_returns_none() {
    let text = "This is a perfectly ordinary text with no apology.";
    assert!(find_apology_marker(text).is_none());
}

/// Partial marker `[APOL` without closing `]` → no false trigger.
#[test]
fn test_partial_marker_no_false_trigger() {
    let text = "[APOL without a closing bracket";
    assert!(find_apology_marker(text).is_none());
}

/// Case-insensitive matching: all casing variants find the marker.
#[test]
fn test_marker_case_insensitive() {
    assert!(find_apology_marker("[apology]").is_some(), "lowercase");
    assert!(find_apology_marker("[APOLOGY]").is_some(), "uppercase");
    assert!(find_apology_marker("[Apology]").is_some(), "capitalized");
    assert!(find_apology_marker("[APOLOGy]").is_some(), "mixed 1");
    assert!(find_apology_marker("[ApOlOgY]").is_some(), "mixed 2");
}

/// Marker in the middle of text → still found at the correct index.
#[test]
fn test_marker_in_middle_of_text() {
    let text = "prefix text [APOLOGY] suffix text";
    let idx = find_apology_marker(text).expect("marker should be found");
    assert_eq!(&text[idx..idx + 9], "[APOLOGY]");
}

/// Marker with extra whitespace before `]` → NOT matched (exact token required).
#[test]
fn test_marker_with_extra_whitespace_not_matched() {
    let text = "[APOLOGY ]";
    assert!(find_apology_marker(text).is_none(), "space before ] should not match");
}

/// Only the opening bracket without the closing one → not matched.
#[test]
fn test_marker_without_closing_bracket() {
    let text = "some text [APOLOGY here";
    assert!(find_apology_marker(text).is_none());
}

/// Text that contains "apology" without bracket syntax → not matched.
#[test]
fn test_plain_word_apology_not_matched() {
    let text = "I offer my sincere apology.";
    assert!(find_apology_marker(text).is_none());
}

// ---------------------------------------------------------------------------
// count_harsh_keywords
// ---------------------------------------------------------------------------

/// No harsh keywords → count is 0.
#[test]
fn test_zero_harsh_keywords() {
    assert_eq!(count_harsh_keywords("This is a kind and gentle critique."), 0);
}

/// Single keyword → count is 1.
#[test]
fn test_one_harsh_keyword() {
    assert_eq!(count_harsh_keywords("This is incompetent work."), 1);
}

/// Two keywords → below the 3-word threshold.
#[test]
fn test_two_harsh_keywords_below_threshold() {
    assert_eq!(
        count_harsh_keywords("This is incompetent and worthless work."),
        2
    );
}

/// Three keywords → at the threshold (≥3 fires the trigger).
#[test]
fn test_three_harsh_keywords_at_threshold() {
    assert_eq!(
        count_harsh_keywords("incompetent, worthless, pathetic"),
        3
    );
}

/// Four keywords → still counted correctly.
#[test]
fn test_four_harsh_keywords() {
    assert_eq!(
        count_harsh_keywords("incompetent worthless pathetic garbage"),
        4
    );
}

/// All eight harsh keywords → counted correctly.
#[test]
fn test_all_eight_harsh_keywords() {
    let text = "incompetent worthless pathetic garbage useless hopeless embarrassing disgrace";
    assert_eq!(count_harsh_keywords(text), 8);
}

/// Keywords are detected case-insensitively.
#[test]
fn test_keywords_case_insensitive() {
    assert_eq!(count_harsh_keywords("INCOMPETENT Worthless pathetic"), 3);
}

/// Keyword as a substring of a larger word IS counted (uses `str::contains`).
/// This is a known characteristic of the current implementation.
#[test]
fn test_keyword_substring_matches() {
    assert_eq!(count_harsh_keywords("uselessly"), 1, "\"uselessly\" contains \"useless\"");
    assert_eq!(count_harsh_keywords("disgraceful"), 1, "\"disgraceful\" contains \"disgrace\"");
}

/// Empty string → no keywords.
#[test]
fn test_empty_string_no_keywords() {
    assert_eq!(count_harsh_keywords(""), 0);
}

/// Only whitespace → no keywords.
#[test]
fn test_whitespace_only_no_keywords() {
    assert_eq!(count_harsh_keywords("   \n  \t  "), 0);
}

// ---------------------------------------------------------------------------
// cooldown_remaining_secs
// ---------------------------------------------------------------------------

/// No apology has ever fired → cooldown is None (no cooldown active).
#[test]
fn test_no_apology_no_cooldown() {
    let cd = ApologyCooldown::default();
    assert!(cd.last_apology_time.is_none());
    assert_eq!(cd.cycles_since_apology, 0);
    assert!(cooldown_remaining_secs(&cd).is_none());
}

/// Apology just fired (elapsed ≈ 0s) → cooldown returns Some(30).
#[test]
fn test_cooldown_immediately_after_apology() {
    let cd = ApologyCooldown {
        last_apology_time: Some(Instant::now()),
        cycles_since_apology: 0,
    };
    let remaining = cooldown_remaining_secs(&cd);
    assert!(remaining.is_some(), "cooldown should be active immediately after apology");
    let secs = remaining.unwrap();
    assert!((1..=30).contains(&secs), "remaining should be between 1 and 30, got {secs}");
}

/// Apology 15 seconds ago → cooldown returns ~15s remaining.
#[test]
fn test_cooldown_fifteen_seconds_ago() {
    let cd = ApologyCooldown {
        last_apology_time: Some(Instant::now() - Duration::from_secs(15)),
        cycles_since_apology: 0,
    };
    let remaining = cooldown_remaining_secs(&cd);
    assert!(remaining.is_some(), "cooldown should be active at t=15s");
    let secs = remaining.unwrap();
    // Elapsed could be 15-16s depending on timing, so remaining is 14-15s.
    assert!((13..=16).contains(&secs), "remaining should be ~15s, got {secs}");
}

/// Apology 30+ seconds ago but only 2 cycles elapsed → cooldown returns Some(0)
/// (waiting for cycle condition).
#[test]
fn test_cooldown_time_met_cycles_not_met() {
    let cd = ApologyCooldown {
        last_apology_time: Some(Instant::now() - Duration::from_secs(35)),
        cycles_since_apology: 2,
    };
    assert_eq!(
        cooldown_remaining_secs(&cd),
        Some(0),
        "time condition met but cycles=2 (<3) should return Some(0)"
    );
}

/// Apology 30+ seconds ago and 3+ cycles elapsed → no cooldown (returns None).
#[test]
fn test_cooldown_time_and_cycles_satisfied() {
    let cd = ApologyCooldown {
        last_apology_time: Some(Instant::now() - Duration::from_secs(35)),
        cycles_since_apology: 3,
    };
    assert!(
        cooldown_remaining_secs(&cd).is_none(),
        "both time and cycle conditions met → no cooldown"
    );
}

/// Apology 30+ seconds ago and many cycles → no cooldown (returns None).
#[test]
fn test_cooldown_satisfied_with_extra_cycles() {
    let cd = ApologyCooldown {
        last_apology_time: Some(Instant::now() - Duration::from_secs(35)),
        cycles_since_apology: 10,
    };
    assert!(cooldown_remaining_secs(&cd).is_none());
}

/// Apology 29 seconds ago with 3 cycles → NOT satisfied (time condition not met).
#[test]
fn test_cooldown_cycles_met_time_not_met() {
    let cd = ApologyCooldown {
        last_apology_time: Some(Instant::now() - Duration::from_secs(29)),
        cycles_since_apology: 3,
    };
    let remaining = cooldown_remaining_secs(&cd);
    assert!(remaining.is_some(), "time condition not met, cooldown should be active");
    // elapsed ≈ 29s (could be 29-30), so remaining ≈ 1-2s.
    let secs = remaining.unwrap();
    assert!(secs <= 2, "remaining should be ~1s, got {secs}");
}

/// A full cycle: apology → cooldown active → conditions met → cooldown clears.
#[test]
fn test_cooldown_full_lifecycle() {
    // Phase 1: immediately after apology — cooldown active
    let cd = ApologyCooldown {
        last_apology_time: Some(Instant::now()),
        cycles_since_apology: 0,
    };
    assert!(cooldown_remaining_secs(&cd).is_some());

    // Phase 2: after 30s with 2 cycles — time condition met, cycles not met
    let cd = ApologyCooldown {
        last_apology_time: Some(Instant::now() - Duration::from_secs(31)),
        cycles_since_apology: 2,
    };
    assert_eq!(cooldown_remaining_secs(&cd), Some(0));

    // Phase 3: after 30s with 3+ cycles — cooldown cleared
    let cd = ApologyCooldown {
        last_apology_time: Some(Instant::now() - Duration::from_secs(31)),
        cycles_since_apology: 3,
    };
    assert!(cooldown_remaining_secs(&cd).is_none());
}

// ---------------------------------------------------------------------------
// ApologyCooldown display / debug formatting
// ---------------------------------------------------------------------------

/// `ApologyCooldown` implements `Debug` for diagnostic display.
#[test]
fn test_apology_cooldown_debug_format() {
    let cd = ApologyCooldown {
        last_apology_time: Some(Instant::now() - Duration::from_secs(10)),
        cycles_since_apology: 1,
    };
    let debug_str = format!("{cd:?}");
    assert!(debug_str.contains("last_apology_time"), "Debug output should include field names");
    assert!(debug_str.contains("cycles_since_apology"), "Debug output should include field names");
}

/// Default cooldown displays properly.
#[test]
fn test_default_cooldown_debug_format() {
    let cd = ApologyCooldown::default();
    let dbg = format!("{cd:?}");
    assert!(dbg.contains("None"), "default cooldown has no last_apology_time");
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// Keyword detection with mixed punctuation and spacing works correctly.
#[test]
fn test_keywords_with_punctuation() {
    assert_eq!(
        count_harsh_keywords("Incompetent! Worthless? Pathetic."),
        3
    );
}

/// Multiple markers in the same text — returns index of first occurrence.
#[test]
fn test_multiple_markers_finds_first() {
    let text = "[APOLOGY] first [APOLOGY] second";
    let idx = find_apology_marker(text).expect("first marker should be found");
    assert_eq!(idx, 0, "first marker is at position 0");
    assert_eq!(&text[idx..idx + 9], "[APOLOGY]");
}
