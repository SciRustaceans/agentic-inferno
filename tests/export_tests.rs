//! Tests for the SVG export used by the README screenshot generator.
//!
//! Seeds a small `App`, renders the real `render()` headlessly via
//! `TestBackend`, converts the buffer with `buffer_to_svg`, and asserts the
//! output is well-formed SVG containing the dark canvas and at least one flame
//! `rgb(` cell. No terminal, no API keys.

use agentic_inferno::tui::export::buffer_to_svg;
use agentic_inferno::tui::ui::{self, App};

use ratatui::backend::TestBackend;
use ratatui::Terminal;

#[test]
fn test_buffer_to_svg_is_well_formed_with_bg_and_flame() {
    let mut app = App::new();
    app.task = "analysis".to_string();
    app.frame = 3;
    app.apply_writer_output("a living document\nthat is never finished");
    app.reveal_all();

    // 120x40 is wide/tall enough for the flame banner (real Rgb cells) and well
    // above the 80x24 "too small" threshold.
    let backend = TestBackend::new(120, 40);
    let mut terminal = Terminal::new(backend).expect("terminal creation");
    terminal
        .draw(|frame| ui::render(frame, &app))
        .expect("draw");
    let svg = buffer_to_svg(terminal.backend().buffer());

    assert!(
        svg.starts_with("<svg"),
        "SVG must start with <svg: {:?}",
        &svg[..svg.len().min(40)]
    );
    assert!(svg.ends_with("</svg>"), "SVG must end with </svg>");
    assert!(
        svg.contains("#0d1117"),
        "SVG must contain the dark canvas bg"
    );
    assert!(
        svg.contains("rgb("),
        "SVG must contain at least one flame rgb() cell"
    );
}
