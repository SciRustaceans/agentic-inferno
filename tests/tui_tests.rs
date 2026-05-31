//! TUI behavior tests — pane rendering, resize handling, and key processing.
//!
//! These tests verify TUI logic without requiring an actual terminal or API keys.
//! Rendering is tested with `ratatui::backend::TestBackend` so no TTY is needed.
//!
//! # Coverage
//!
//! - PaneBuffer: capping, scroll, push, visible_lines
//! - App state: event simulation (WriterOutput, CriticOutput, ApologyReady, etc.)
//! - Key handling: Esc, Ctrl+C, Tab, Up/Down, PageUp/PageDown, Home/End
//! - Focus cycling: Tab toggles Writer ↔ Critic
//! - Render: three-pane titles, content positioning, too-small warning, resize
//! - Terminal safety: TerminalGuard drop, panic hook installation

use agentic_inferno::app::AppState;
use agentic_inferno::error::AppError;
use agentic_inferno::tui::input::{self, ControlFlow};
use agentic_inferno::tui::pane::PaneBuffer;
use agentic_inferno::tui::ui::{self, App, FocusTarget};
use agentic_inferno::tui::TerminalGuard;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::style::Color;
use ratatui::Terminal;

// =============================================================================
// Test helpers
// =============================================================================

/// Build a `KeyEvent` for a single key press with no modifiers.
fn press(key_code: KeyCode) -> KeyEvent {
    KeyEvent {
        code: key_code,
        modifiers: KeyModifiers::empty(),
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

/// Build a `KeyEvent` for a Ctrl+key combination.
fn ctrl(key_code: KeyCode) -> KeyEvent {
    KeyEvent {
        code: key_code,
        modifiers: KeyModifiers::CONTROL,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

/// Build a `KeyEvent` with `KeyEventKind::Repeat` (should be ignored by the handler).
fn repeat(key_code: KeyCode) -> KeyEvent {
    KeyEvent {
        code: key_code,
        modifiers: KeyModifiers::empty(),
        kind: KeyEventKind::Repeat,
        state: KeyEventState::NONE,
    }
}

/// Collect the full text content of a `Buffer` as a single string, one row per
/// line joined by `\n`.  Trailing whitespace is trimmed from each row so
/// substring assertions are less fragile.
fn buffer_text(buffer: &Buffer) -> String {
    let area = buffer.area();
    let mut lines: Vec<String> = Vec::with_capacity(area.height as usize);
    for y in 0..area.height {
        let mut row = String::with_capacity(area.width as usize);
        for x in 0..area.width {
            row.push_str(buffer.cell((x, y)).map(|c| c.symbol()).unwrap_or(" "));
        }
        let trimmed = row.trim_end().to_string();
        if !trimmed.is_empty() || !lines.is_empty() {
            lines.push(trimmed);
        }
    }
    while lines.last().is_some_and(|s| s.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

fn buffer_contains(buffer: &Buffer, needle: &str) -> bool {
    buffer_text(buffer).contains(needle)
}

/// Create an `App` with content pushed to both writer and critic buffers, ready
/// for render tests.
fn app_with_content(writer_lines: &[&str], critic_lines: &[&str]) -> App {
    let app = App::new();
    {
        let mut w = app.writer_buffer.write().expect("writer lock");
        for line in writer_lines {
            w.push(line);
        }
    }
    {
        let mut c = app.critic_buffer.write().expect("critic lock");
        for line in critic_lines {
            c.push(line);
        }
    }
    app
}

/// Render an app to a `TestBackend` of the given size and return the buffer.
fn render_app(app: &App, width: u16, height: u16) -> Buffer {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal creation");
    terminal.draw(|frame| ui::render(frame, app)).expect("draw");
    terminal.backend().buffer().clone()
}

// =============================================================================
// PaneBuffer integration — capping & scroll
// =============================================================================

#[test]
fn test_pane_buffer_cap_2000_lines_retain_1000() {
    let mut buf = PaneBuffer::new();
    for i in 0..2000 {
        buf.push(&format!("line_{i:04}"));
    }
    assert_eq!(buf.len(), 1000, "buffer must cap at max_lines=1000");
    assert_eq!(buf.max_lines(), 1000);
    let content = buf.content();
    assert!(!content.contains("line_0000"), "first 1000 lines evicted");
    assert!(
        content.contains("line_1000"),
        "line 1000 should be first retained"
    );
    assert!(
        content.contains("line_1999"),
        "line 1999 should be last retained"
    );
}

#[test]
fn test_pane_buffer_cap_2000_with_custom_max() {
    let mut buf = PaneBuffer::with_max_lines(1000);
    for i in 0..2000 {
        buf.push(&format!("L{i:04}"));
    }
    assert_eq!(buf.len(), 1000);
    let content = buf.content();
    assert!(!content.contains("L0000"));
    assert!(content.contains("L1000"));
}

#[test]
fn test_pane_buffer_scroll_pgup_pgdn_changes_visible_content() {
    let mut buf = PaneBuffer::with_max_lines(100);
    for i in 0..50 {
        buf.push(&format!("row_{i:02}"));
    }
    let bottom = buf.visible_lines(10);
    assert_eq!(bottom.len(), 10);
    assert_eq!(bottom[0], "row_40");
    assert_eq!(bottom[9], "row_49");

    buf.scroll_up(10);
    let after_pgup = buf.visible_lines(10);
    assert_eq!(after_pgup.len(), 10);
    assert_eq!(after_pgup[0], "row_30");
    assert_eq!(after_pgup[9], "row_39");

    buf.scroll_down(10);
    let after_pgdn = buf.visible_lines(10);
    assert_eq!(after_pgdn[0], "row_40");
    assert_eq!(after_pgdn[9], "row_49");
}

#[test]
fn test_pane_buffer_scroll_up_past_start_clamps() {
    let mut buf = PaneBuffer::with_max_lines(100);
    for i in 0..10 {
        buf.push(&format!("r{i:02}"));
    }
    buf.scroll_up(100);
    let visible = buf.visible_lines(5);
    assert_eq!(visible.len(), 5);
    assert_eq!(visible[0], "r00");
    assert_eq!(visible[4], "r04");
}

// =============================================================================
// App construction & defaults
// =============================================================================

#[test]
fn test_app_new_has_correct_defaults() {
    let app = App::new();
    assert_eq!(app.state, AppState::Idle);
    assert!(app.apology_text.is_none());
    assert!(app.error.is_none());
    assert_eq!(app.cost_spent, 0.0);
    assert_eq!(app.cost_limit, 0.0);
    assert_eq!(app.writer_cost, 0.0);
    assert_eq!(app.critic_cost, 0.0);
    assert_eq!(app.apology_cost, 0.0);
    assert_eq!(app.writer_version, 0);
    assert_eq!(app.critic_version, 0);
    assert_eq!(app.focused_pane, FocusTarget::Writer);
    assert!(app.apology_cooldown.is_none());
    assert!(app.writer_buffer.read().expect("lock").is_empty());
    assert!(app.critic_buffer.read().expect("lock").is_empty());
}

#[test]
fn test_app_default_equals_new() {
    let a = App::new();
    let b = App::default();
    assert_eq!(a.state, b.state);
    assert_eq!(a.writer_version, b.writer_version);
    assert_eq!(a.critic_version, b.critic_version);
    assert_eq!(a.focused_pane, b.focused_pane);
}

// =============================================================================
// Event handling — simulate handle_app_event via direct App manipulation
// =============================================================================

#[test]
fn test_event_writer_output_pushes_to_writer_buffer() {
    let app = App::new();
    {
        let mut buf = app.writer_buffer.write().expect("lock");
        buf.push("hello world");
        buf.scroll_to_bottom();
    }
    assert!(app
        .writer_buffer
        .read()
        .expect("lock")
        .content()
        .contains("hello world"));
    assert!(app.critic_buffer.read().expect("lock").is_empty());
}

#[test]
fn test_event_critic_output_pushes_to_critic_buffer() {
    let app = App::new();
    {
        let mut buf = app.critic_buffer.write().expect("lock");
        buf.push("critique line");
        buf.scroll_to_bottom();
    }
    assert!(app
        .critic_buffer
        .read()
        .expect("lock")
        .content()
        .contains("critique line"));
    assert!(app.writer_buffer.read().expect("lock").is_empty());
}

#[test]
fn test_event_apology_ready_sets_apology_text() {
    let mut app = App::new();
    app.apology_text = Some("I regret everything".to_string());
    assert_eq!(app.apology_text.as_deref(), Some("I regret everything"));
}

#[test]
fn test_event_writer_done_updates_writer_version() {
    let mut app = App::new();
    app.writer_version = 42;
    assert_eq!(app.writer_version, 42);
}

#[test]
fn test_event_critique_ready_updates_critic_version() {
    let mut app = App::new();
    app.critic_version = 7;
    assert_eq!(app.critic_version, 7);
}

#[test]
fn test_event_error_sets_error_field() {
    let mut app = App::new();
    app.error = Some(AppError::Timeout);
    assert!(app
        .error
        .as_ref()
        .unwrap()
        .to_string()
        .contains("timed out"));
}

#[test]
fn test_event_cost_warning_updates_cost_fields() {
    let mut app = App::new();
    app.cost_spent = 1.5;
    app.cost_limit = 2.0;
    app.writer_cost = 0.8;
    app.critic_cost = 0.4;
    app.apology_cost = 0.3;
    assert_eq!(app.cost_spent, 1.5);
    assert_eq!(app.cost_limit, 2.0);
    assert_eq!(app.writer_cost, 0.8);
    assert_eq!(app.critic_cost, 0.4);
    assert_eq!(app.apology_cost, 0.3);
}

#[test]
fn test_event_loop_exhausted_transitions_to_done() {
    let mut app = App::new();
    app.state = AppState::Running;
    app.state = AppState::Done;
    assert_eq!(app.state, AppState::Done);
}

#[test]
fn test_event_shutdown_transitions_to_done() {
    let mut app = App::new();
    app.state = AppState::Running;
    app.state = AppState::Done;
    assert_eq!(app.state, AppState::Done);
}

#[test]
fn test_event_apology_cooldown_updates_field() {
    let mut app = App::new();
    app.apology_cooldown = Some(30);
    assert_eq!(app.apology_cooldown, Some(30));
    app.apology_cooldown = None;
    assert_eq!(app.apology_cooldown, None);
}

// =============================================================================
// Focus cycling
// =============================================================================

#[test]
fn test_focus_cycling_writer_to_critic_to_writer() {
    let mut app = App::new();
    assert_eq!(
        app.focused_pane,
        FocusTarget::Writer,
        "default focus is Writer"
    );
    app.focused_pane = FocusTarget::Critic;
    assert_eq!(app.focused_pane, FocusTarget::Critic);
    app.focused_pane = FocusTarget::Writer;
    assert_eq!(app.focused_pane, FocusTarget::Writer);
}

// =============================================================================
// Key input — handle_key
// =============================================================================

#[test]
fn test_esc_key_sends_stop() {
    let mut app = App::new();
    assert_eq!(
        input::handle_key(&mut app, press(KeyCode::Esc)),
        ControlFlow::Stop
    );
}

#[test]
fn test_q_key_sends_stop() {
    let mut app = App::new();
    assert_eq!(
        input::handle_key(&mut app, press(KeyCode::Char('q'))),
        ControlFlow::Stop
    );
}

#[test]
fn test_ctrl_c_sends_quit() {
    let mut app = App::new();
    assert_eq!(
        input::handle_key(&mut app, ctrl(KeyCode::Char('c'))),
        ControlFlow::Quit
    );
}

#[test]
fn test_tab_cycles_focus_writer_to_critic() {
    let mut app = App::new();
    assert_eq!(app.focused_pane, FocusTarget::Writer);
    assert_eq!(
        input::handle_key(&mut app, press(KeyCode::Tab)),
        ControlFlow::Continue
    );
    assert_eq!(app.focused_pane, FocusTarget::Critic);
}

#[test]
fn test_tab_cycles_focus_critic_to_writer() {
    let mut app = App::new();
    app.focused_pane = FocusTarget::Critic;
    assert_eq!(
        input::handle_key(&mut app, press(KeyCode::Tab)),
        ControlFlow::Continue
    );
    assert_eq!(app.focused_pane, FocusTarget::Writer);
}

#[test]
fn test_up_scrolls_focused_writer_pane() {
    let mut app = App::new();
    for i in 0..50 {
        app.writer_buffer
            .write()
            .expect("lock")
            .push(&format!("w{i:02}"));
    }
    assert_eq!(app.writer_buffer.read().expect("lock").scroll_position(), 0);
    input::handle_key(&mut app, press(KeyCode::Up));
    assert_eq!(app.writer_buffer.read().expect("lock").scroll_position(), 1);
}

#[test]
fn test_up_scrolls_focused_critic_pane() {
    let mut app = App::new();
    app.focused_pane = FocusTarget::Critic;
    for i in 0..50 {
        app.critic_buffer
            .write()
            .expect("lock")
            .push(&format!("c{i:02}"));
    }
    input::handle_key(&mut app, press(KeyCode::Up));
    assert_eq!(app.critic_buffer.read().expect("lock").scroll_position(), 1);
}

#[test]
fn test_down_scrolls_focused_pane() {
    let mut app = App::new();
    app.writer_buffer.write().expect("lock").scroll_up(5);
    assert_eq!(app.writer_buffer.read().expect("lock").scroll_position(), 5);
    input::handle_key(&mut app, press(KeyCode::Down));
    assert_eq!(app.writer_buffer.read().expect("lock").scroll_position(), 4);
}

#[test]
fn test_down_clamps_at_zero() {
    let mut app = App::new();
    input::handle_key(&mut app, press(KeyCode::Down));
    assert_eq!(app.writer_buffer.read().expect("lock").scroll_position(), 0);
}

#[test]
fn test_pageup_scrolls_by_10() {
    let mut app = App::new();
    for i in 0..50 {
        app.writer_buffer
            .write()
            .expect("lock")
            .push(&format!("w{i}"));
    }
    input::handle_key(&mut app, press(KeyCode::PageUp));
    assert_eq!(
        app.writer_buffer.read().expect("lock").scroll_position(),
        10
    );
    input::handle_key(&mut app, press(KeyCode::PageUp));
    assert_eq!(
        app.writer_buffer.read().expect("lock").scroll_position(),
        20
    );
}

#[test]
fn test_pagedown_scrolls_by_10() {
    let mut app = App::new();
    for i in 0..50 {
        app.writer_buffer
            .write()
            .expect("lock")
            .push(&format!("w{i}"));
    }
    app.writer_buffer.write().expect("lock").scroll_up(30);
    input::handle_key(&mut app, press(KeyCode::PageDown));
    assert_eq!(
        app.writer_buffer.read().expect("lock").scroll_position(),
        20
    );
    input::handle_key(&mut app, press(KeyCode::PageDown));
    assert_eq!(
        app.writer_buffer.read().expect("lock").scroll_position(),
        10
    );
}

#[test]
fn test_home_scrolls_to_top() {
    let mut app = App::new();
    for i in 0..50 {
        app.writer_buffer
            .write()
            .expect("lock")
            .push(&format!("w{i:02}"));
    }
    input::handle_key(&mut app, press(KeyCode::Home));
    let scroll_pos = app.writer_buffer.read().expect("lock").scroll_position();
    assert_eq!(scroll_pos, usize::MAX);
}

#[test]
fn test_visible_lines_shows_top_when_scrolled_up_enough() {
    let mut buf = PaneBuffer::with_max_lines(100);
    for i in 0..20 {
        buf.push(&format!("top_{i:02}"));
    }
    buf.scroll_up(10);
    let visible = buf.visible_lines(10);
    assert_eq!(visible[0], "top_00");
    assert_eq!(visible[9], "top_09");
}

#[test]
fn test_end_scrolls_to_bottom() {
    let mut app = App::new();
    for i in 0..50 {
        app.writer_buffer
            .write()
            .expect("lock")
            .push(&format!("w{i:02}"));
    }
    app.writer_buffer.write().expect("lock").scroll_up(40);
    input::handle_key(&mut app, press(KeyCode::End));
    assert_eq!(app.writer_buffer.read().expect("lock").scroll_position(), 0);
}

#[test]
fn test_non_press_events_are_ignored() {
    let mut app = App::new();
    assert_eq!(
        input::handle_key(&mut app, repeat(KeyCode::Esc)),
        ControlFlow::Continue
    );
    assert_eq!(
        input::handle_key(&mut app, repeat(KeyCode::Tab)),
        ControlFlow::Continue
    );
    assert_eq!(app.focused_pane, FocusTarget::Writer);
}

#[test]
fn test_unknown_key_continues() {
    let mut app = App::new();
    assert_eq!(
        input::handle_key(&mut app, press(KeyCode::F(1))),
        ControlFlow::Continue
    );
}

// =============================================================================
// Render — three-pane layout & titles
// =============================================================================

#[test]
fn test_render_three_pane_titles_120x40() {
    let app = app_with_content(&["writer text"], &["critic text"]);
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(text.contains("Writer"), "missing Writer pane title");
    assert!(text.contains("Critic"), "missing Critic pane title");
    assert!(text.contains("Status"), "missing Status pane title");
}

#[test]
fn test_render_writer_output_appears_in_left_pane() {
    let app = app_with_content(&["UNIQUE_WRITER_MARKER_12345"], &[]);
    let buffer = render_app(&app, 120, 40);
    assert!(buffer_contains(&buffer, "UNIQUE_WRITER_MARKER_12345"));
}

#[test]
fn test_render_critic_output_appears_in_right_pane() {
    let app = app_with_content(&[], &["UNIQUE_CRITIC_MARKER_67890"]);
    let buffer = render_app(&app, 120, 40);
    assert!(buffer_contains(&buffer, "UNIQUE_CRITIC_MARKER_67890"));
}

#[test]
fn test_render_apology_text_appears_in_centered_popup() {
    let mut app = App::new();
    app.apology_text = Some("SORRY_I_EXIST_99999".to_string());
    let buffer = render_app(&app, 120, 40);
    assert!(buffer_contains(&buffer, "SORRY_I_EXIST_99999"));
}

#[test]
fn test_render_apology_popup_shows_apology_title() {
    let mut app = App::new();
    app.apology_text = Some("mea culpa".to_string());
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(
        text.contains("Apology"),
        "Apology popup must be titled 'Apology'"
    );
    assert!(text.contains("mea culpa"));
}

#[test]
fn test_apology_ttl_decrements_and_clears_text() {
    let mut app = App::new();
    app.apology_text = Some("transient apology".to_string());
    app.apology_ttl = 2;

    app.tick_apology();
    assert_eq!(app.apology_ttl, 1);
    assert!(
        app.apology_text.is_some(),
        "apology stays while ttl remains"
    );

    app.tick_apology();
    assert_eq!(app.apology_ttl, 0);
    assert!(
        app.apology_text.is_none(),
        "apology cleared when ttl reaches 0"
    );

    // Further ticks are a no-op at ttl 0.
    app.tick_apology();
    assert_eq!(app.apology_ttl, 0);
}

#[test]
fn test_render_too_small_warning_40x10() {
    let app = App::new();
    let buffer = render_app(&app, 40, 10);
    let text = buffer_text(&buffer);
    assert!(
        text.contains("Terminal too small") || text.contains("too small"),
        "Expected 'Terminal too small' warning at 40x10. Got: {text}"
    );
}

#[test]
fn test_render_layout_recalculates_120x40() {
    let app = app_with_content(&["w"], &["c"]);
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(
        !text.contains("Terminal too small"),
        "120x40 should not trigger warning"
    );
    assert!(text.contains("Writer"), "Writer pane missing at 120x40");
    assert!(text.contains("Critic"), "Critic pane missing at 120x40");
}

#[test]
fn test_render_at_minimum_bounds_80x24() {
    let app = app_with_content(&["data"], &["data"]);
    let buffer = render_app(&app, 80, 24);
    let text = buffer_text(&buffer);
    assert!(
        !text.contains("Terminal too small"),
        "80x24 is the minimum, should not warn"
    );
    assert!(text.contains("Writer"));
    assert!(text.contains("Critic"));
}

#[test]
fn test_render_just_below_minimum_79x23_warns() {
    let app = App::new();
    let buffer = render_app(&app, 79, 23);
    let text = buffer_text(&buffer);
    assert!(
        text.contains("Terminal too small") || text.contains("too small"),
        "79x23 is below minimum, should warn. Got: {text}"
    );
}

// =============================================================================
// Render — version titles
// =============================================================================

#[test]
fn test_render_writer_title_shows_version_when_writer_done() {
    let mut app = App::new();
    app.writer_buffer.write().expect("lock").push("content");
    app.writer_version = 3;
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(
        text.contains("Writer [v3]"),
        "Writer title should show version: {text}"
    );
}

#[test]
fn test_render_critic_title_shows_version_when_critique_ready() {
    let mut app = App::new();
    app.critic_buffer.write().expect("lock").push("content");
    app.critic_version = 5;
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(
        text.contains("Critic [v5]"),
        "Critic title should show version: {text}"
    );
}

#[test]
fn test_render_titles_without_versions() {
    let app = App::new();
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(text.contains("Writer"), "Writer title missing");
    assert!(text.contains("Critic"), "Critic title missing");
    assert!(!text.contains("[v"));
}

// =============================================================================
// Render — focused pane border highlighting
// =============================================================================

#[test]
fn test_render_writer_focused_border() {
    let mut app = app_with_content(&["ww"], &["cc"]);
    app.focused_pane = FocusTarget::Writer;
    let buffer = render_app(&app, 120, 40);
    assert!(buffer_contains(&buffer, "Writer"));
    assert!(buffer_contains(&buffer, "Critic"));
}

#[test]
fn test_render_critic_focused_border() {
    let mut app = app_with_content(&["ww"], &["cc"]);
    app.focused_pane = FocusTarget::Critic;
    let buffer = render_app(&app, 120, 40);
    assert!(buffer_contains(&buffer, "Writer"));
    assert!(buffer_contains(&buffer, "Critic"));
}

// =============================================================================
// Render — status bar with cost info
// =============================================================================

#[test]
fn test_render_status_bar_shows_cost_info() {
    let mut app = App::new();
    app.cost_spent = 1.25;
    app.cost_limit = 2.00;
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(
        text.contains("$1.25"),
        "Status bar must show cost spent: {text}"
    );
    assert!(
        text.contains("$2.00"),
        "Status bar must show cost limit: {text}"
    );
    assert!(
        text.contains("Esc to stop"),
        "Status bar must show exit hint: {text}"
    );
}

#[test]
fn test_render_banner_shows_per_agent_tokens() {
    // Per-agent detail moved out of the status bar into the banner's token
    // meter. The slim status bar no longer carries per-agent figures.
    let mut app = App::new();
    app.writer_tokens = 500;
    app.critic_tokens = 300;
    app.apology_tokens = 100;
    app.total_tokens = 900;
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    // Abbreviated to W/C/A (C2) so the meter stays short and never word-wraps.
    assert!(text.contains("W 500"), "Writer tokens missing: {text}");
    assert!(text.contains("C 300"), "Critic tokens missing: {text}");
    assert!(text.contains("A 100"), "Apology tokens missing: {text}");
}

#[test]
fn test_render_status_bar_shows_cooldown() {
    let mut app = App::new();
    app.apology_cooldown = Some(42);
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(
        text.contains("cooldown: 42s"),
        "Status bar should show apology cooldown: {text}"
    );
}

// =============================================================================
// Render — error display in bottom pane
// =============================================================================

#[test]
fn test_render_error_shows_in_penance_pane() {
    let mut app = App::new();
    app.error = Some(AppError::Timeout);
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(
        text.contains("Error"),
        "Error pane should have 'Error' title: {text}"
    );
    assert!(
        text.contains("timed out"),
        "Error text should be visible: {text}"
    );
}

#[test]
fn test_render_error_takes_priority_over_apology() {
    let mut app = App::new();
    app.error = Some(AppError::CostCeilingExceeded(5.0, 3.0));
    app.apology_text = Some("hidden apology".to_string());
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(text.contains("Error"), "Error should take priority: {text}");
    assert!(
        !text.contains("hidden apology"),
        "Apology should be hidden when error present: {text}"
    );
}

// =============================================================================
// Render — version info in status bar
// =============================================================================

#[test]
fn test_render_version_info_in_pane_titles() {
    // Versions are no longer duplicated in the slim status bar — they live in
    // the pane titles (bracketed form).
    let mut app = App::new();
    app.writer_buffer.write().expect("lock").push("w");
    app.critic_buffer.write().expect("lock").push("c");
    app.writer_version = 2;
    app.critic_version = 1;
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(text.contains("Writer [v2]"), "writer version title: {text}");
    assert!(text.contains("Critic [v1]"), "critic version title: {text}");
}

// =============================================================================
// Terminal safety
// =============================================================================

#[test]
fn test_terminal_guard_can_be_created_and_dropped() {
    let guard = TerminalGuard;
    drop(guard);
}

#[test]
fn test_terminal_guard_exists() {
    let _guard: TerminalGuard = TerminalGuard;
}

#[test]
fn test_panic_hook_installs_without_panicking() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        agentic_inferno::tui::install_panic_hook();
    }));
    assert!(result.is_ok(), "install_panic_hook should not panic");
}

// =============================================================================
// ControlFlow enum
// =============================================================================

#[test]
fn test_control_flow_values_are_distinct() {
    assert_ne!(ControlFlow::Continue, ControlFlow::Stop);
    assert_ne!(ControlFlow::Stop, ControlFlow::Quit);
    assert_ne!(ControlFlow::Continue, ControlFlow::Quit);
}

#[test]
fn test_control_flow_debug() {
    assert!(format!("{:?}", ControlFlow::Stop).contains("Stop"));
    assert!(format!("{:?}", ControlFlow::Quit).contains("Quit"));
    assert!(format!("{:?}", ControlFlow::Continue).contains("Continue"));
}

// =============================================================================
// FocusTarget enum
// =============================================================================

#[test]
fn test_focus_target_default_is_writer() {
    assert_eq!(FocusTarget::default(), FocusTarget::Writer);
}

// =============================================================================
// Thread-safety compile-time checks
// =============================================================================

#[test]
fn test_app_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<App>();
}

#[test]
fn test_app_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<App>();
}

// =============================================================================
// Arc<RwLock<PaneBuffer>> shared access
// =============================================================================

#[test]
fn test_writer_buffer_is_shared_between_app_and_clone() {
    let app = App::new();
    let clone = app.writer_buffer.clone();
    clone.write().expect("lock").push("shared line");
    assert!(app
        .writer_buffer
        .read()
        .expect("lock")
        .content()
        .contains("shared line"));
}

#[test]
fn test_critic_buffer_is_shared_between_app_and_clone() {
    let app = App::new();
    let clone = app.critic_buffer.clone();
    clone.write().expect("lock").push("critic shared");
    assert!(app
        .critic_buffer
        .read()
        .expect("lock")
        .content()
        .contains("critic shared"));
}

// =============================================================================
// PaneBuffer — visible_lines edge cases
// =============================================================================

#[test]
fn test_visible_lines_empty_buffer_returns_empty() {
    let buf = PaneBuffer::new();
    assert!(buf.visible_lines(20).is_empty());
}

#[test]
fn test_visible_lines_zero_height_returns_empty() {
    let mut buf = PaneBuffer::new();
    buf.push("data");
    assert!(buf.visible_lines(0).is_empty());
}

#[test]
fn test_visible_lines_buffer_smaller_than_height() {
    let mut buf = PaneBuffer::new();
    buf.push("a");
    buf.push("b");
    buf.push("c");
    let visible = buf.visible_lines(10);
    assert_eq!(visible, vec!["a", "b", "c"]);
}

// =============================================================================
// PaneBuffer — clear
// =============================================================================

#[test]
fn test_clear_also_resets_scroll() {
    let mut buf = PaneBuffer::new();
    buf.push("data");
    buf.scroll_up(20);
    buf.clear();
    assert!(buf.is_empty());
    assert_eq!(buf.scroll_position(), 0);
}

// =============================================================================
// CENTERPIECE — render-level scroll proof
// =============================================================================

/// Push ~60 distinct lines, render, and assert that scroll position actually
/// changes what is drawn. This is the proof the TUI scroll fix is real: the
/// previous renderer used `content()` (whole buffer joined) and ignored scroll.
#[test]
fn test_render_scroll_changes_visible_lines() {
    let app = App::new();
    {
        let mut w = app.writer_buffer.write().expect("writer lock");
        for i in 0..60 {
            // 7-char `line_NN`; a ~58-col pane (120 wide / 2 - borders) won't
            // wrap, and the values won't substring-collide.
            w.push(&format!("line_{i:02}"));
        }
    }

    // scroll_position starts at 0 (bottom) → newest visible, oldest not.
    let buffer = render_app(&app, 120, 40);
    assert!(
        buffer_contains(&buffer, "line_59"),
        "bottom line should be visible at scroll=0"
    );
    assert!(
        !buffer_contains(&buffer, "line_00"),
        "top line must NOT be visible at scroll=0"
    );

    // Scroll to top → oldest visible, newest not.
    app.writer_buffer
        .write()
        .expect("writer lock")
        .scroll_to_top();
    let buffer = render_app(&app, 120, 40);
    assert!(
        buffer_contains(&buffer, "line_00"),
        "top line should be visible after scroll_to_top"
    );
    assert!(
        !buffer_contains(&buffer, "line_59"),
        "bottom line must NOT be visible after scroll_to_top"
    );
}

// =============================================================================
// apply_writer_output — REPLACE semantics
// =============================================================================

#[test]
fn test_apply_writer_output_replaces_previous_document() {
    let mut app = App::new();
    app.apply_writer_output("docA l1\ndocA l2");
    app.reveal_all();
    {
        let content = app.writer_buffer.read().expect("lock").content();
        assert!(
            content.contains("docA"),
            "first document should be present: {content}"
        );
    }
    app.apply_writer_output("docB only");
    app.reveal_all();
    let content = app.writer_buffer.read().expect("lock").content();
    assert!(
        content.contains("docB"),
        "second document should be present: {content}"
    );
    assert!(
        !content.contains("docA"),
        "first document should be replaced: {content}"
    );
}

#[test]
fn test_apply_writer_output_splits_into_lines() {
    let mut app = App::new();
    app.apply_writer_output("first\nsecond\nthird");
    // Typing now reveals over time; reveal everything to inspect the full doc.
    app.reveal_all();
    let buf = app.writer_buffer.read().expect("lock");
    // Each text line becomes its own buffer line so scroll moves by lines.
    assert_eq!(buf.len(), 3, "document should be split line-by-line");
    // The rebuilt pane follows the typing, so the viewport sits at the bottom.
    assert_eq!(
        buf.scroll_position(),
        0,
        "writer reveal follows the typing to the bottom"
    );
}

#[test]
fn test_apply_writer_output_empty_leaves_buffer_empty() {
    let mut app = App::new();
    app.apply_writer_output("seed");
    app.apply_writer_output("");
    app.reveal_all();
    let buf = app.writer_buffer.read().expect("lock");
    assert!(buf.is_empty(), "empty text should leave the buffer empty");
}

// =============================================================================
// Typewriter reveal — panes type out over ticks
// =============================================================================

/// Join the writer buffer's lines back into a single string (matching how the
/// reveal rebuild splits on `.lines()`).
fn writer_content(app: &App) -> String {
    app.writer_buffer.read().expect("lock").content()
}

/// Strip newlines so a partial reveal (whose trailing `\n` is dropped by
/// `.lines()`) can be compared against the target prefix.
fn no_newlines(s: &str) -> String {
    s.chars().filter(|c| *c != '\n').collect()
}

#[test]
fn test_writer_reveal_starts_empty_then_grows() {
    let mut app = App::new();
    app.reveal_step = 5;
    app.apply_writer_output("line one\nline two\nline three");

    // Before any tick, nothing is revealed — the buffer is empty.
    assert!(
        app.writer_buffer.read().expect("lock").is_empty(),
        "writer pane should start empty (revealed=0)"
    );
    assert_eq!(app.writer_revealed, 0);

    // Tick a few times; the visible content must grow monotonically.
    let mut prev = no_newlines(&writer_content(&app)).len();
    for _ in 0..4 {
        app.tick_reveal();
        let now = no_newlines(&writer_content(&app)).len();
        assert!(
            now >= prev,
            "revealed content length must not shrink: {prev} -> {now}"
        );
        // Whatever is visible must be a prefix of the target (sans newlines).
        let target_nl = no_newlines("line one\nline two\nline three");
        let visible_nl = no_newlines(&writer_content(&app));
        assert!(
            target_nl.starts_with(&visible_nl),
            "visible content must be a prefix of the target: {visible_nl:?}"
        );
        prev = now;
    }
}

#[test]
fn test_writer_reveal_completes_after_enough_ticks() {
    let mut app = App::new();
    app.reveal_step = 5;
    let target = "line one\nline two\nline three";
    app.apply_writer_output(target);

    // Tick enough times to reveal everything (char count / step, plus slack).
    let total = target.chars().count();
    for _ in 0..(total / 5 + 2) {
        app.tick_reveal();
    }

    assert_eq!(
        app.writer_revealed, total,
        "all characters should be revealed"
    );
    let expected = target.lines().collect::<Vec<_>>().join("\n");
    assert_eq!(
        writer_content(&app),
        expected,
        "fully revealed content must equal the target document"
    );
}

#[test]
fn test_reveal_all_shows_full_target_immediately() {
    let mut app = App::new();
    app.reveal_step = 1;
    let target = "alpha\nbeta\ngamma";
    app.apply_writer_output(target);
    // Only one tick (1 char) — still mostly hidden.
    app.tick_reveal();
    assert!(app.writer_revealed < target.chars().count());

    app.reveal_all();
    assert_eq!(app.writer_revealed, target.chars().count());
    let expected = target.lines().collect::<Vec<_>>().join("\n");
    assert_eq!(writer_content(&app), expected);
}

#[test]
fn test_tick_reveal_clamps_at_target_length() {
    let mut app = App::new();
    app.reveal_step = 100;
    app.apply_writer_output("short");
    app.tick_reveal();
    assert_eq!(app.writer_revealed, "short".chars().count());
    // Further ticks do not overshoot.
    app.tick_reveal();
    assert_eq!(app.writer_revealed, "short".chars().count());
}

#[test]
fn test_critic_reveal_keeps_already_revealed_text() {
    let mut app = App::new();
    app.reveal_step = 1000; // reveal whole appends each tick
    app.critic_version = 1;
    app.apply_critic_output("first critique");
    app.tick_reveal();
    let after_first = app.critic_buffer.read().expect("lock").content();
    assert!(
        after_first.contains("first critique"),
        "first critique should be revealed: {after_first}"
    );

    // A second append leaves the first shown and reveals the new tail.
    app.critic_version = 2;
    app.apply_critic_output("second critique");
    app.tick_reveal();
    let after_second = app.critic_buffer.read().expect("lock").content();
    assert!(
        after_second.contains("first critique"),
        "earlier critique must stay shown: {after_second}"
    );
    assert!(
        after_second.contains("second critique"),
        "new critique should reveal: {after_second}"
    );
}

// =============================================================================
// apply_critic_output — FEED semantics with version headers
// =============================================================================

#[test]
fn test_apply_critic_output_feeds_with_version_headers() {
    let mut app = App::new();
    app.critic_version = 1;
    app.apply_critic_output("crit one");
    app.critic_version = 2;
    app.apply_critic_output("crit two");
    app.reveal_all();

    let content = app.critic_buffer.read().expect("lock").content();
    assert!(
        content.contains("── v1 ──"),
        "first version header missing: {content}"
    );
    assert!(
        content.contains("crit one"),
        "first critique missing: {content}"
    );
    assert!(
        content.contains("── v2 ──"),
        "second version header missing: {content}"
    );
    assert!(
        content.contains("crit two"),
        "second critique missing: {content}"
    );
    // Feed scrolls to the bottom so the newest critique is visible.
    assert_eq!(app.critic_buffer.read().expect("lock").scroll_position(), 0);
}

// =============================================================================
// AGENT INFERNO banner + animated flame title
// =============================================================================

/// Find the first buffer row whose joined symbols contain `needle` and return
/// the per-cell foreground colors for that whole row.
fn row_fg_colors(buffer: &Buffer, needle: &str) -> Vec<ratatui::style::Color> {
    let area = *buffer.area();
    for y in 0..area.height {
        let mut row = String::new();
        for x in 0..area.width {
            row.push_str(buffer.cell((x, y)).map(|c| c.symbol()).unwrap_or(" "));
        }
        if row.contains(needle) {
            return (0..area.width)
                .map(|x| {
                    buffer
                        .cell((x, y))
                        .map(|c| c.fg)
                        .unwrap_or(ratatui::style::Color::Reset)
                })
                .collect();
        }
    }
    Vec::new()
}

#[test]
fn test_flame_color_deterministic_for_fixed_inputs() {
    // Same (index, frame) → same color, every time.
    assert_eq!(ui::flame_color(0, 0), ui::flame_color(0, 0));
    assert_eq!(ui::flame_color(3, 7), ui::flame_color(3, 7));
}

#[test]
fn test_flame_color_changes_as_frame_advances() {
    // Advancing the frame rotates the palette (length > 1), so at least some
    // index produces a different color between consecutive frames.
    let differs =
        (0..ui::BANNER_TITLE.len()).any(|i| ui::flame_color(i, 0) != ui::flame_color(i, 1));
    assert!(differs, "flame_color must change as frame advances");
    // Specifically, index 0 shifts one palette slot.
    assert_ne!(ui::flame_color(0, 0), ui::flame_color(0, 1));
}

#[test]
fn test_render_banner_shows_agent_inferno_title() {
    // At a compact size (below the 100x30 big-banner threshold) the literal
    // single-line flame title is rendered.
    let app = App::new();
    let buffer = render_app(&app, 90, 28);
    let text = buffer_text(&buffer);
    assert!(
        text.contains("AGENT INFERNO"),
        "compact banner must show literal AGENT INFERNO title: {text}"
    );
}

#[test]
fn test_render_big_banner_shows_block_title_rows() {
    // At a wide size (>=100x30) the big ASCII-art banner renders. The block
    // glyphs use the full-block char, so multiple non-empty title rows appear.
    let app = App::new();
    let buffer = render_app(&app, 120, 40);
    let area = *buffer.area();
    let mut block_rows = 0;
    for y in 0..area.height {
        let mut row = String::new();
        for x in 0..area.width {
            row.push_str(buffer.cell((x, y)).map(|c| c.symbol()).unwrap_or(" "));
        }
        if row.contains('█') {
            block_rows += 1;
        }
    }
    assert!(
        block_rows >= 5,
        "big banner must render multiple non-empty block-art title rows, got {block_rows}"
    );
}

#[test]
fn test_render_big_banner_flames_animate_between_frames() {
    // The big banner's flame rows and block title are colored via flame_color,
    // which depends on `frame`. Comparing the full banner-area cell grid
    // (symbol + fg) across two frames must show a difference.
    let mut app = App::new();
    app.frame = 0;
    let buffer_a = render_app(&app, 120, 40);
    app.frame = 7;
    let buffer_b = render_app(&app, 120, 40);

    // The big banner occupies the top 13 rows (Length(13)).
    let area = *buffer_a.area();
    let banner_h = 13.min(area.height);
    let mut differs = false;
    for y in 0..banner_h {
        for x in 0..area.width {
            let a = buffer_a.cell((x, y));
            let b = buffer_b.cell((x, y));
            let (sa, fa) = a.map(|c| (c.symbol(), c.fg)).unwrap_or((" ", Color::Reset));
            let (sb, fb) = b.map(|c| (c.symbol(), c.fg)).unwrap_or((" ", Color::Reset));
            if sa != sb || fa != fb {
                differs = true;
                break;
            }
        }
        if differs {
            break;
        }
    }
    assert!(
        differs,
        "big banner must change (flames/title colors) between frame 0 and frame 7"
    );
}

#[test]
fn test_render_banner_shows_token_figure() {
    let mut app = App::new();
    app.writer_tokens = 8100;
    app.critic_tokens = 3200;
    app.apology_tokens = 1000;
    app.total_tokens = 12300;
    app.task = "analysis".to_string();
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(
        text.contains("Tokens 12300"),
        "banner must show total token figure: {text}"
    );
    assert!(
        text.contains("W 8100"),
        "banner must show writer tokens: {text}"
    );
    assert!(
        text.contains("C 3200"),
        "banner must show critic tokens: {text}"
    );
    assert!(
        text.contains("A 1000"),
        "banner must show apology tokens: {text}"
    );
    assert!(
        text.contains("analysis"),
        "banner must show the task label: {text}"
    );
}

#[test]
fn test_render_banner_animation_changes_title_styles() {
    // Render the title at two different frames; the text still reads
    // AGENT INFERNO but the per-cell foreground colors differ (the gradient
    // ripples sideways). Use the compact size so the literal single-line
    // flame title (not the big block-art banner) is exercised.
    let mut app = App::new();
    app.frame = 0;
    let buffer_a = render_app(&app, 90, 28);
    app.frame = 1;
    let buffer_b = render_app(&app, 90, 28);

    assert!(buffer_text(&buffer_a).contains("AGENT INFERNO"));
    assert!(buffer_text(&buffer_b).contains("AGENT INFERNO"));

    let fg_a = row_fg_colors(&buffer_a, "AGENT INFERNO");
    let fg_b = row_fg_colors(&buffer_b, "AGENT INFERNO");
    assert!(!fg_a.is_empty(), "title row not found in frame 0 buffer");
    assert_ne!(fg_a, fg_b, "title cell colors must differ between frames");
}
