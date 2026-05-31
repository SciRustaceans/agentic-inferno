//! Dependency-free SVG export of a rendered ratatui [`Buffer`].
//!
//! The TUI's `render()` is a pure function of `App` state, so the tests render
//! it headlessly into an in-memory [`Buffer`] via `TestBackend`. This module
//! turns such a buffer into a self-contained SVG string — a per-cell grid of
//! background `<rect>`s and foreground `<text>` glyphs colored from each cell's
//! `fg`/`bg`/`modifier`. The `examples/screenshots.rs` generator feeds these to
//! `rsvg-convert` to produce the README screenshots, with no API calls and no
//! external crates.
//!
//! Colors map ANSI names to a GitHub-dark-ish palette so the flame `Rgb` cells,
//! the box-drawing borders, and the body text all read faithfully.

use ratatui::buffer::Buffer;
use ratatui::style::{Color, Modifier};

/// Cell width in pixels (matches a 16px monospace em advance closely enough).
const CW: f64 = 9.6;
/// Cell height in pixels.
const CH: f64 = 20.0;
/// Terminal background color (GitHub dark canvas).
const BG: &str = "#0d1117";
/// Default light foreground for `Color::Reset` / unmapped colors.
const FG_DEFAULT: &str = "#c9d1d9";

/// Map a ratatui [`Color`] to a CSS color string for the foreground.
///
/// `Color::Reset` (the default style carried by almost every body/border cell)
/// and any unmapped variant fall back to a light gray so panes never render as
/// empty. Flame `Rgb` cells pass through as `rgb(r,g,b)`.
fn fg_css(color: Color) -> String {
    match color {
        Color::Rgb(r, g, b) => format!("rgb({r},{g},{b})"),
        Color::Yellow | Color::LightYellow => "#e5c07b".to_string(),
        Color::Red => "#e06c75".to_string(),
        Color::LightRed => "#ff7b72".to_string(),
        Color::Green | Color::LightGreen => "#98c379".to_string(),
        Color::Cyan | Color::LightCyan => "#56b6c2".to_string(),
        Color::Blue | Color::LightBlue => "#61afef".to_string(),
        Color::Magenta | Color::LightMagenta => "#c678dd".to_string(),
        Color::White | Color::Gray => FG_DEFAULT.to_string(),
        Color::DarkGray => "#6e7681".to_string(),
        Color::Black => BG.to_string(),
        // Color::Reset and anything else → default light foreground.
        _ => FG_DEFAULT.to_string(),
    }
}

/// Map a ratatui [`Color`] to a CSS color string for the background, or `None`
/// when the cell should keep the canvas color (no `<rect>` emitted).
fn bg_css(color: Color) -> Option<String> {
    match color {
        Color::Reset => None,
        other => Some(fg_css(other)),
    }
}

/// XML-escape the five characters that are significant inside SVG text/attrs.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Render a ratatui [`Buffer`] to a self-contained SVG string.
///
/// The output begins exactly with `<svg` and ends exactly with `</svg>` (no XML
/// prolog, no trailing newline) so it embeds and asserts cleanly. The canvas is
/// `cols*CW × rows*CH` pixels on a dark background; each cell contributes an
/// optional background `<rect>` and, for non-blank symbols, a `<text>` glyph.
pub fn buffer_to_svg(buffer: &Buffer) -> String {
    let area = buffer.area();
    let cols = area.width;
    let rows = area.height;
    let width = cols as f64 * CW;
    let height = rows as f64 * CH;

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    ));
    // Full-canvas dark background.
    svg.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"{BG}\"/>"
    ));

    for y in 0..rows {
        for x in 0..cols {
            let Some(cell) = buffer.cell((x, y)) else {
                continue;
            };
            let px = x as f64 * CW;
            let py = y as f64 * CH;

            // Background rect for any non-Reset background.
            if let Some(bg) = bg_css(cell.bg) {
                svg.push_str(&format!(
                    "<rect x=\"{px}\" y=\"{py}\" width=\"{CW}\" height=\"{CH}\" fill=\"{bg}\"/>"
                ));
            }

            // Foreground glyph for any non-blank symbol.
            let sym = cell.symbol();
            if !sym.is_empty() && sym != " " {
                let baseline = (y as f64 + 1.0) * CH - 5.0;
                let fill = fg_css(cell.fg);
                let mut extra = String::new();
                if cell.modifier.contains(Modifier::BOLD) {
                    extra.push_str(" font-weight=\"bold\"");
                }
                if cell.modifier.contains(Modifier::DIM) {
                    extra.push_str(" opacity=\"0.55\"");
                }
                svg.push_str(&format!(
                    "<text x=\"{px}\" y=\"{baseline}\" fill=\"{fill}\" \
                     font-family=\"Menlo, DejaVu Sans Mono, monospace\" font-size=\"16\"{extra}>{}</text>",
                    xml_escape(sym)
                ));
            }
        }
    }

    svg.push_str("</svg>");
    svg
}
