use std::collections::VecDeque;

/// A bounded ring buffer for pane content with scroll position tracking.
///
/// # Scroll model
///
/// `scroll_position = 0` means the viewport is at the **bottom** (showing the
/// latest content). `scroll_position` increases as the user scrolls **up**.
///
/// # Thread safety
///
/// `PaneBuffer` itself is **not** thread-safe. Callers wrap it in
/// `Arc<RwLock<PaneBuffer>>` for shared access between the TUI render loop
/// and the content-producing tasks.
#[derive(Debug, Clone)]
pub struct PaneBuffer {
    /// Ring buffer of content lines — newest at the back.
    lines: VecDeque<String>,
    /// How many lines the user has scrolled up from the bottom.
    /// 0 = bottom (latest), positive = scrolled up.
    scroll_position: usize,
    /// Maximum number of lines held in the buffer.
    max_lines: usize,
}

impl PaneBuffer {
    /// Create a new `PaneBuffer` with the default capacity (1000 lines).
    pub fn new() -> Self {
        Self {
            lines: VecDeque::with_capacity(1000),
            scroll_position: 0,
            max_lines: 1000,
        }
    }

    /// Create a new `PaneBuffer` with a custom maximum line count.
    ///
    /// `max_lines` is clamped to a minimum of 1 — zero or empty buffers are
    /// not useful.
    pub fn with_max_lines(max_lines: usize) -> Self {
        let max = max_lines.max(1);
        Self {
            lines: VecDeque::with_capacity(max),
            scroll_position: 0,
            max_lines: max,
        }
    }

    /// Append a line to the buffer.
    ///
    /// If the buffer is at capacity, the oldest line is evicted first-in,
    /// first-out.
    pub fn push(&mut self, line: &str) {
        if self.lines.len() == self.max_lines {
            self.lines.pop_front();
        }
        self.lines.push_back(line.to_string());
    }

    /// Return a slice of lines visible within the given `height` viewport,
    /// accounting for the current scroll position.
    ///
    /// `scroll_position = 0` shows the last `height` lines. Higher
    /// scroll positions shift the window upward.
    pub fn visible_lines(&self, height: usize) -> Vec<&str> {
        if self.lines.is_empty() || height == 0 {
            return Vec::new();
        }

        let total = self.lines.len();

        // Buffer is shorter than viewport — show everything.
        if total <= height {
            return self.lines.iter().map(String::as_str).collect();
        }

        // scroll_position=0 shows last `height` lines.
        // scroll_position increases → look further up the buffer.
        let start = total.saturating_sub(height + self.scroll_position);
        let end = (start + height).min(total);

        self.lines.range(start..end).map(String::as_str).collect()
    }

    /// Scroll the viewport up by `n` lines.
    ///
    /// `scroll_position` increases, meaning we're looking at older content.
    /// It saturates — it cannot grow past `usize::MAX`.
    pub fn scroll_up(&mut self, n: usize) {
        self.scroll_position = self.scroll_position.saturating_add(n);
    }

    /// Scroll the viewport down by `n` lines (toward the latest content).
    ///
    /// `scroll_position` decreases. It cannot go below 0 — at 0 the viewport
    /// shows the bottom (most recent) content.
    pub fn scroll_down(&mut self, n: usize) {
        self.scroll_position = self.scroll_position.saturating_sub(n);
    }

    /// Reset scroll position to 0 — showing the bottom (latest) content.
    pub fn scroll_to_bottom(&mut self) {
        self.scroll_position = 0;
    }

    /// Scroll to the top of the buffer, showing the earliest content.
    ///
    /// Sets scroll position to `usize::MAX` so that `visible_lines()` always
    /// starts at index 0 of the buffer.
    pub fn scroll_to_top(&mut self) {
        self.scroll_position = usize::MAX;
    }

    /// Clear all content and reset scroll position.
    pub fn clear(&mut self) {
        self.lines.clear();
        self.scroll_position = 0;
    }

    /// Full buffer content joined by newlines (for prompt construction).
    ///
    /// Does **not** include a trailing newline.
    pub fn content(&self) -> String {
        self.lines.iter().map(String::as_str).collect::<Vec<_>>().join("\n")
    }

    // ── Accessors (for testing / inspection) ──────────────────────

    /// Current number of buffered lines.
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Current scroll position.
    pub fn scroll_position(&self) -> usize {
        self.scroll_position
    }

    /// Maximum line capacity.
    pub fn max_lines(&self) -> usize {
        self.max_lines
    }
}

impl Default for PaneBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ──────────────────────────────────────────────

    #[test]
    fn test_new_creates_empty_buffer() {
        let buf = PaneBuffer::new();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.max_lines(), 1000);
        assert_eq!(buf.scroll_position(), 0);
    }

    #[test]
    fn test_with_max_lines_clamps_to_one() {
        let buf = PaneBuffer::with_max_lines(0);
        assert_eq!(buf.max_lines(), 1);
        let buf = PaneBuffer::with_max_lines(1);
        assert_eq!(buf.max_lines(), 1);
        let buf = PaneBuffer::with_max_lines(500);
        assert_eq!(buf.max_lines(), 500);
    }

    // ── Push and capping ──────────────────────────────────────────

    #[test]
    fn test_push_appends() {
        let mut buf = PaneBuffer::new();
        buf.push("hello");
        buf.push("world");
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.content(), "hello\nworld");
    }

    #[test]
    fn test_push_caps_at_max_lines() {
        let mut buf = PaneBuffer::with_max_lines(3);
        buf.push("a");
        buf.push("b");
        buf.push("c");
        assert_eq!(buf.len(), 3);
        // Fourth push evicts "a"
        buf.push("d");
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.content(), "b\nc\nd");
    }

    #[test]
    fn test_push_evicts_oldest_first() {
        let mut buf = PaneBuffer::with_max_lines(2);
        buf.push("line1");
        buf.push("line2");
        buf.push("line3");
        assert_eq!(buf.content(), "line2\nline3");
        buf.push("line4");
        assert_eq!(buf.content(), "line3\nline4");
    }

    // ── visible_lines ─────────────────────────────────────────────

    #[test]
    fn test_visible_lines_empty_buffer() {
        let buf = PaneBuffer::new();
        assert!(buf.visible_lines(10).is_empty());
    }

    #[test]
    fn test_visible_lines_zero_height() {
        let mut buf = PaneBuffer::new();
        buf.push("a");
        assert!(buf.visible_lines(0).is_empty());
    }

    #[test]
    fn test_visible_lines_partial_fill() {
        let mut buf = PaneBuffer::new();
        buf.push("a");
        buf.push("b");
        // height=10, but only 2 lines → show both
        let visible = buf.visible_lines(10);
        assert_eq!(visible, vec!["a", "b"]);
    }

    #[test]
    fn test_visible_lines_at_bottom() {
        let mut buf = PaneBuffer::with_max_lines(100);
        for i in 0..50 {
            buf.push(&format!("line_{i:02}"));
        }
        // height=10, scroll=0 → last 10 lines
        let visible = buf.visible_lines(10);
        assert_eq!(visible.len(), 10);
        assert_eq!(visible[0], "line_40");
        assert_eq!(visible[9], "line_49");
    }

    #[test]
    fn test_visible_lines_scrolled_up() {
        let mut buf = PaneBuffer::with_max_lines(100);
        for i in 0..50 {
            buf.push(&format!("line_{i:02}"));
        }
        // scroll up by 5
        buf.scroll_up(5);
        let visible = buf.visible_lines(10);
        assert_eq!(visible.len(), 10);
        // Without scroll: lines 40..49
        // With scroll=5:  lines 35..44
        assert_eq!(visible[0], "line_35");
        assert_eq!(visible[9], "line_44");
    }

    #[test]
    fn test_visible_lines_scrolled_past_start() {
        let mut buf = PaneBuffer::with_max_lines(100);
        for i in 0..20 {
            buf.push(&format!("line_{i:02}"));
        }
        // 20 lines total, height=10, scroll up 50 → would go past beginning
        buf.scroll_up(50);
        let visible = buf.visible_lines(10);
        // Clamped to show first 10 lines
        assert_eq!(visible.len(), 10);
        assert_eq!(visible[0], "line_00");
        assert_eq!(visible[9], "line_09");
    }

    // ── Scroll ────────────────────────────────────────────────────

    #[test]
    fn test_scroll_down_clamps_at_zero() {
        let mut buf = PaneBuffer::new();
        buf.scroll_up(10);
        assert_eq!(buf.scroll_position(), 10);
        buf.scroll_down(5);
        assert_eq!(buf.scroll_position(), 5);
        buf.scroll_down(10); // would go negative → clamped to 0
        assert_eq!(buf.scroll_position(), 0);
    }

    #[test]
    fn test_scroll_up_saturates() {
        let mut buf = PaneBuffer::new();
        buf.scroll_up(usize::MAX - 1);
        buf.scroll_up(5);
        // Should saturate at usize::MAX, not overflow
        assert_eq!(buf.scroll_position(), usize::MAX);
    }

    #[test]
    fn test_scroll_to_bottom() {
        let mut buf = PaneBuffer::new();
        buf.scroll_up(42);
        assert_eq!(buf.scroll_position(), 42);
        buf.scroll_to_bottom();
        assert_eq!(buf.scroll_position(), 0);
    }

    // ── Clear ─────────────────────────────────────────────────────

    #[test]
    fn test_clear() {
        let mut buf = PaneBuffer::new();
        buf.push("a");
        buf.push("b");
        buf.scroll_up(10);
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.scroll_position(), 0);
        assert_eq!(buf.content(), "");
    }

    // ── Content ───────────────────────────────────────────────────

    #[test]
    fn test_content_empty() {
        let buf = PaneBuffer::new();
        assert_eq!(buf.content(), "");
    }

    #[test]
    fn test_content_single_line() {
        let mut buf = PaneBuffer::new();
        buf.push("only line");
        assert_eq!(buf.content(), "only line");
    }

    #[test]
    fn test_content_multiple_lines() {
        let mut buf = PaneBuffer::new();
        buf.push("first");
        buf.push("second");
        buf.push("third");
        assert_eq!(buf.content(), "first\nsecond\nthird");
    }

    #[test]
    fn test_content_no_trailing_newline() {
        let mut buf = PaneBuffer::new();
        buf.push("a");
        buf.push("b");
        let s = buf.content();
        assert!(!s.ends_with('\n'));
        assert_eq!(s, "a\nb");
    }

    // ── Default ───────────────────────────────────────────────────

    #[test]
    fn test_default_equals_new() {
        assert_eq!(
            PaneBuffer::default().max_lines(),
            PaneBuffer::new().max_lines()
        );
    }
}
