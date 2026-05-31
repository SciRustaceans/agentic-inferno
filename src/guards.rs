use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::error::AppError;

/// Cost ceiling guard — tracks cumulative LLM spend and rejects once the limit is hit.
///
/// Only successful calls count. The limit is the maximum total spend in USD.
///
/// The limit is stored as an [`AtomicU64`] holding the `f64::to_bits`
/// representation, so it can be updated live (e.g. from a TUI settings menu)
/// without a second mutex while the hot-path `record()` keeps a single lock.
pub struct CostCeiling {
    spent: Mutex<f64>,
    limit: AtomicU64,
}

impl CostCeiling {
    /// Create a new cost ceiling with the given limit in USD.
    ///
    /// # Panics
    ///
    /// Panics if `limit <= 0.0`.
    pub fn new(limit: f64) -> Self {
        assert!(
            limit > 0.0,
            "CostCeiling limit must be positive, got {limit}"
        );
        Self {
            spent: Mutex::new(0.0),
            limit: AtomicU64::new(limit.to_bits()),
        }
    }

    /// Record a successful LLM call cost.
    ///
    /// Returns `Err(AppError::CostCeilingExceeded)` if the addition would exceed the limit.
    /// The cost is not recorded on error — only successful calls count.
    pub fn record(&self, cost: f64) -> Result<(), AppError> {
        let limit = self.limit();
        let mut spent = self.spent.lock().expect("CostCeiling mutex poisoned");
        let new_total = *spent + cost;
        if new_total > limit {
            return Err(AppError::CostCeilingExceeded(new_total, limit));
        }
        *spent = new_total;
        Ok(())
    }

    /// Return the total spend so far.
    pub fn spent(&self) -> f64 {
        *self.spent.lock().expect("CostCeiling mutex poisoned")
    }

    /// Return the spend limit.
    pub fn limit(&self) -> f64 {
        f64::from_bits(self.limit.load(Ordering::Relaxed))
    }

    /// Update the spend limit live (e.g. from a settings menu).
    ///
    /// Lock-free — stores the new limit's bit pattern atomically. Raising the
    /// limit allows spends that previously errored; lowering it tightens the
    /// threshold `record()` enforces on the next call.
    pub fn set_limit(&self, new: f64) {
        self.limit.store(new.to_bits(), Ordering::Relaxed);
    }
}

/// Loop detection guard — detects repetitive output via semantic hashing.
///
/// Maintains a sliding window of semantic hashes. If the same hash appears
/// `min_repeats` times within the window, the output is considered a loop.
pub struct LoopDetection {
    window: usize,
    min_repeats: usize,
    history: Mutex<VecDeque<u64>>,
}

impl LoopDetection {
    /// Create a new loop detector.
    ///
    /// `window` is the number of recent iterations to consider (default 5).
    /// `min_repeats` is the threshold for declaring a loop (default 3).
    pub fn new(window: usize, min_repeats: usize) -> Self {
        Self {
            window,
            min_repeats,
            history: Mutex::new(VecDeque::with_capacity(window)),
        }
    }

    /// Check if the given semantic hash indicates a loop.
    ///
    /// Pushes the hash into the history window, then counts how many times
    /// it appears. Returns `Err(AppError::LoopExhausted)` if the repetition
    /// count meets or exceeds `min_repeats`.
    pub fn check(&self, text_hash: u64) -> Result<(), AppError> {
        let mut history = self.history.lock().expect("LoopDetection mutex poisoned");
        history.push_back(text_hash);
        while history.len() > self.window {
            history.pop_front();
        }
        let count = history.iter().filter(|&&h| h == text_hash).count();
        if count >= self.min_repeats {
            return Err(AppError::LoopExhausted(format!(
                "Semantic hash {:#x} appeared {} times in the last {} iterations (threshold: {})",
                text_hash, count, self.window, self.min_repeats,
            )));
        }
        Ok(())
    }
}

/// Context window guard — tracks token usage and warns when approaching the limit.
///
/// Uses a fast atomic counter for the current token count. The limit is the
/// maximum number of tokens (input + output) for the context window.
pub struct ContextWindow {
    max_tokens: usize,
    current: AtomicUsize,
}

impl ContextWindow {
    /// Create a new context window guard with the given maximum token count.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            current: AtomicUsize::new(0),
        }
    }

    /// Check if adding `additional` tokens would exceed warning thresholds.
    ///
    /// Returns `None` if the projected usage is below 80% of the limit.
    /// Returns `Some(ratio)` if the projected usage is at or above 80%.
    /// Callers should interpret the ratio:
    /// - 80–90%: advisory — consider pruning.
    /// - Above 90%: force prune before proceeding.
    pub fn would_exceed(&self, additional: usize) -> Option<f64> {
        let current = self.current.load(Ordering::Relaxed);
        let projected = current + additional;
        let ratio = projected as f64 / self.max_tokens as f64;
        if ratio < 0.8 {
            None
        } else {
            Some(ratio)
        }
    }

    /// Record token usage. Adds `tokens` to the current count.
    pub fn record_usage(&self, tokens: usize) {
        self.current.fetch_add(tokens, Ordering::Relaxed);
    }

    /// Prune the oldest tokens (e.g., dropping the oldest critique).
    /// Subtracts `oldest_tokens` from the current count.
    pub fn prune(&self, oldest_tokens: usize) {
        self.current.fetch_sub(oldest_tokens, Ordering::Relaxed);
    }

    /// Return the current token count.
    pub fn current(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    /// Return the maximum token count.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

/// Compute a semantic hash of text for loop detection.
///
/// Strips all whitespace, lowercases ASCII characters, then hashes with
/// `std::hash::DefaultHasher`. This catches rephrased dead ends that
/// differ only in whitespace, capitalization, or minor wording changes.
pub fn semantic_hash(text: &str) -> u64 {
    let normalized: String = text
        .chars()
        .filter(|c| !c.is_ascii_whitespace())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    normalized.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CostCeiling ──────────────────────────────────────────────

    #[test]
    fn test_cost_ceiling_new() {
        let c = CostCeiling::new(2.0);
        assert_eq!(c.limit(), 2.0);
        assert_eq!(c.spent(), 0.0);
    }

    #[test]
    #[should_panic(expected = "limit must be positive")]
    fn test_cost_ceiling_zero_limit() {
        CostCeiling::new(0.0);
    }

    #[test]
    #[should_panic(expected = "limit must be positive")]
    fn test_cost_ceiling_negative_limit() {
        CostCeiling::new(-1.0);
    }

    #[test]
    fn test_cost_ceiling_record_below_limit() {
        let c = CostCeiling::new(10.0);
        assert!(c.record(2.50).is_ok());
        assert_eq!(c.spent(), 2.50);
    }

    #[test]
    fn test_cost_ceiling_record_multiple_calls() {
        let c = CostCeiling::new(5.0);
        assert!(c.record(1.0).is_ok());
        assert!(c.record(2.0).is_ok());
        assert!(c.record(1.50).is_ok());
        assert_eq!(c.spent(), 4.50);
    }

    #[test]
    fn test_cost_ceiling_exceeded() {
        let c = CostCeiling::new(5.0);
        assert!(c.record(3.0).is_ok());
        let err = c.record(3.0).unwrap_err();
        assert!(matches!(err, AppError::CostCeilingExceeded(6.0, 5.0)));
        // The failed call should NOT have been recorded:
        assert_eq!(c.spent(), 3.0);
    }

    #[test]
    fn test_cost_ceiling_set_limit_raise_and_lower() {
        let c = CostCeiling::new(5.0);
        assert_eq!(c.limit(), 5.0);

        // A spend over the original cap errors.
        assert!(c.record(6.0).is_err());
        assert_eq!(c.spent(), 0.0);

        // Raise the cap — the same spend now succeeds.
        c.set_limit(10.0);
        assert_eq!(c.limit(), 10.0);
        assert!(c.record(6.0).is_ok());
        assert_eq!(c.spent(), 6.0);

        // Lower the cap below the next would-be total — it tightens immediately.
        c.set_limit(7.0);
        assert_eq!(c.limit(), 7.0);
        let err = c.record(2.0).unwrap_err();
        assert!(matches!(err, AppError::CostCeilingExceeded(8.0, 7.0)));
        // Already-spent total is unchanged on the rejected call.
        assert_eq!(c.spent(), 6.0);
    }

    #[test]
    fn test_cost_ceiling_exact_limit() {
        let c = CostCeiling::new(5.0);
        assert!(c.record(5.0).is_ok());
        assert_eq!(c.spent(), 5.0);
        // One more cent exceeds
        let err = c.record(0.01).unwrap_err();
        assert!(matches!(err, AppError::CostCeilingExceeded(5.01, 5.0)));
    }

    // ── LoopDetection ────────────────────────────────────────────

    #[test]
    fn test_loop_detection_new() {
        let ld = LoopDetection::new(5, 3);
        assert!(ld.check(1).is_ok());
        assert!(ld.check(2).is_ok());
    }

    #[test]
    fn test_loop_detection_triggers() {
        let ld = LoopDetection::new(5, 3);
        assert!(ld.check(42).is_ok());
        assert!(ld.check(42).is_ok());
        // Third time should trigger
        let err = ld.check(42).unwrap_err();
        assert!(matches!(err, AppError::LoopExhausted(_)));
    }

    #[test]
    fn test_loop_detection_no_false_positive_different_hashes() {
        let ld = LoopDetection::new(5, 3);
        for h in 0..5 {
            assert!(ld.check(h).is_ok());
        }
    }

    #[test]
    fn test_loop_detection_window_slides() {
        let ld = LoopDetection::new(3, 2);
        assert!(ld.check(1).is_ok()); // [1]
        assert!(ld.check(2).is_ok()); // [1, 2]
        assert!(ld.check(3).is_ok()); // [1, 2, 3]
        assert!(ld.check(1).is_ok()); // [2, 3, 1] — 1 fell out, only one 1
        assert!(ld.check(1).is_err()); // [3, 1, 1] — count=2 >= 2 → loop
    }

    // ── ContextWindow ────────────────────────────────────────────

    #[test]
    fn test_context_window_new() {
        let cw = ContextWindow::new(100);
        assert_eq!(cw.max_tokens(), 100);
        assert_eq!(cw.current(), 0);
    }

    #[test]
    fn test_context_window_record_and_current() {
        let cw = ContextWindow::new(1000);
        cw.record_usage(200);
        assert_eq!(cw.current(), 200);
        cw.record_usage(300);
        assert_eq!(cw.current(), 500);
    }

    #[test]
    fn test_context_window_prune() {
        let cw = ContextWindow::new(1000);
        cw.record_usage(500);
        cw.prune(200);
        assert_eq!(cw.current(), 300);
    }

    #[test]
    fn test_context_window_would_exceed_below_80() {
        let cw = ContextWindow::new(1000);
        cw.record_usage(100);
        assert!(cw.would_exceed(100).is_none()); // 200/1000 = 20%
        assert!(cw.would_exceed(699).is_none()); // 799/1000 = 79.9%
    }

    #[test]
    fn test_context_window_would_exceed_at_80() {
        let cw = ContextWindow::new(1000);
        cw.record_usage(800);
        let ratio = cw.would_exceed(0).unwrap();
        assert!((ratio - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_context_window_would_exceed_above_80() {
        let cw = ContextWindow::new(1000);
        cw.record_usage(500);
        let ratio = cw.would_exceed(350).unwrap(); // 850/1000 = 85%
        assert!((ratio - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_context_window_would_exceed_above_90() {
        let cw = ContextWindow::new(1000);
        cw.record_usage(500);
        let ratio = cw.would_exceed(450).unwrap(); // 950/1000 = 95%
        assert!((ratio - 0.95).abs() < 1e-10);
    }

    // ── semantic_hash ────────────────────────────────────────────

    #[test]
    fn test_semantic_hash_ignores_whitespace_and_case() {
        let h1 = semantic_hash("Hello World");
        let h2 = semantic_hash("  hello   world  ");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_semantic_hash_different_content() {
        let h1 = semantic_hash("Hello World");
        let h2 = semantic_hash("Hello World!");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_semantic_hash_normalizes_newlines() {
        let h1 = semantic_hash("foo\nbar\nbaz");
        let h2 = semantic_hash("foo bar baz");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_semantic_hash_full_normalize() {
        let h1 = semantic_hash("The quick brown fox");
        let h2 = semantic_hash("thequickbrownfox");
        assert_eq!(h1, h2);
    }
}
