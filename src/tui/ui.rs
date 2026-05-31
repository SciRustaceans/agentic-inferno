use std::sync::{Arc, RwLock};

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use crate::app::AppState;
use crate::error::AppError;
use crate::tui::pane::PaneBuffer;

/// Which pane is currently focused for keyboard input.
///
/// Focus determines which pane receives scroll events and which pane
/// gets a highlighted border in the render phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FocusTarget {
    /// The Writer (left) pane — displays LLM-generated content.
    #[default]
    Writer,
    /// The Critic (right) pane — displays critique analysis.
    Critic,
}

/// Application UI state — holds all data for the three-pane layout.
///
/// Writer/Critic buffers are wrapped in `Arc<RwLock<PaneBuffer>>` so the
/// orchestrator can write content concurrently with the TUI render loop.
/// The version counters are incremented by the event handler whenever new
/// content arrives, allowing the renderer to skip re-rendering unchanged panes
/// once differential rendering is added.
pub struct App {
    /// Writer (left) pane content buffer — thread-safe.
    pub writer_buffer: Arc<RwLock<PaneBuffer>>,
    /// Critic (right) pane content buffer — thread-safe.
    pub critic_buffer: Arc<RwLock<PaneBuffer>>,
    /// The latest generated apology text, if any.
    pub apology_text: Option<String>,
    /// Animation frames remaining before the apology popup auto-dismisses.
    ///
    /// Set when an apology arrives; decremented each animation tick. When it
    /// reaches 0 the popup is dismissed by clearing [`App::apology_text`].
    pub apology_ttl: u16,
    /// Overall application lifecycle state (Idle, Running, Stopping, Done).
    pub state: AppState,
    /// Last error received via `AppEvent::Error`, if any.
    pub error: Option<AppError>,
    /// Cumulative cost spent so far, in USD.
    pub cost_spent: f64,
    /// Cost ceiling from configuration, in USD.
    pub cost_limit: f64,
    /// Writer agent cumulative cost, in USD.
    pub writer_cost: f64,
    /// Critic agent cumulative cost, in USD.
    pub critic_cost: f64,
    /// Apology agent cumulative cost, in USD.
    pub apology_cost: f64,
    /// Monotonic version counter for Writer content — bumped on each `WriterOutput` event.
    pub writer_version: u64,
    /// Monotonic version counter for Critic content — bumped on each `CriticOutput` event.
    pub critic_version: u64,
    /// Which pane currently has keyboard focus (highlighted border).
    pub focused_pane: FocusTarget,
    /// Seconds remaining on the apology cooldown, if active.
    ///
    /// `Some(secs)` when the cooldown is in effect (shown in status bar);
    /// `None` when the cooldown has expired or no apology has occurred yet.
    pub apology_cooldown: Option<u64>,
    /// Cumulative tokens attributed to the Writer agent.
    pub writer_tokens: u64,
    /// Cumulative tokens attributed to the Critic agent.
    pub critic_tokens: u64,
    /// Cumulative tokens attributed to the Apology agent.
    pub apology_tokens: u64,
    /// Total tokens across all agents (sum of the three above).
    pub total_tokens: u64,
    /// The active inferno task label shown in the banner (e.g. "analysis").
    pub task: String,
    /// Animation frame counter — drives the flame title gradient. Bumped on
    /// each animation tick in the TUI loop; pure render input.
    pub frame: u64,
    /// Full text the Writer pane is typing toward (the latest revision).
    pub writer_target: String,
    /// How many characters of `writer_target` have been revealed so far.
    pub writer_revealed: usize,
    /// Accumulated text the Critic pane is typing toward (all critiques so far,
    /// with `── vN ──` headers). Capped to a maximum length.
    pub critic_target: String,
    /// How many characters of `critic_target` have been revealed so far.
    pub critic_revealed: usize,
    /// Characters revealed per animation tick (the typewriter step). Derived
    /// from the configured reveal speed; set before the loop starts.
    pub reveal_step: usize,
}

/// Maximum length of the accumulated Critic `critic_target`, in characters.
///
/// When exceeded, the oldest characters are dropped (and the revealed count
/// decremented accordingly) so the buffer stays bounded over a long run.
const CRITIC_TARGET_MAX_CHARS: usize = 20_000;

/// Default typewriter step (characters revealed per animation tick).
///
/// Chosen so that at ~8 fps the Normal speed (40 cps) reveals 5 chars/tick.
const DEFAULT_REVEAL_STEP: usize = 5;

impl App {
    pub fn new() -> Self {
        Self {
            writer_buffer: Arc::new(RwLock::new(PaneBuffer::new())),
            critic_buffer: Arc::new(RwLock::new(PaneBuffer::new())),
            apology_text: None,
            apology_ttl: 0,
            state: AppState::Idle,
            error: None,
            cost_spent: 0.0,
            cost_limit: 0.0,
            writer_cost: 0.0,
            critic_cost: 0.0,
            apology_cost: 0.0,
            writer_version: 0,
            critic_version: 0,
            focused_pane: FocusTarget::default(),
            apology_cooldown: None,
            writer_tokens: 0,
            critic_tokens: 0,
            apology_tokens: 0,
            total_tokens: 0,
            task: String::new(),
            frame: 0,
            writer_target: String::new(),
            writer_revealed: 0,
            critic_target: String::new(),
            critic_revealed: 0,
            reveal_step: DEFAULT_REVEAL_STEP,
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Replace the Writer pane's typing target with a new document revision.
    ///
    /// The writer pane shows the current living document, so each revision
    /// resets the reveal: `writer_target` becomes `text`, the revealed count
    /// is reset to 0, and the pane buffer is rebuilt to show the (empty)
    /// revealed prefix. The text then types out over subsequent reveal ticks.
    ///
    /// Empty `text` leaves the buffer empty — the rebuilt prefix is empty so
    /// no lines are pushed.
    pub fn apply_writer_output(&mut self, text: &str) {
        self.writer_target = text.to_string();
        self.writer_revealed = 0;
        self.rebuild_writer();
    }

    /// Append a new critique to the Critic pane's typing target.
    ///
    /// The critic pane accumulates critiques: each call appends a
    /// `── v{critic_version} ──` header followed by the critique text to
    /// `critic_target`. Already-revealed text stays shown; the new tail types
    /// out over subsequent reveal ticks. `critic_version` is read at call
    /// time, so the orchestrator must emit `CritiqueReady` before
    /// `CriticOutput`.
    ///
    /// `critic_target` is capped to [`CRITIC_TARGET_MAX_CHARS`]: when it grows
    /// past the cap, the oldest characters are dropped (char-wise, never
    /// byte-wise) and `critic_revealed` is decremented by the dropped count so
    /// the reveal stays aligned with the (now shorter) target.
    pub fn apply_critic_output(&mut self, text: &str) {
        self.critic_target
            .push_str(&format!("── v{} ──\n{}\n", self.critic_version, text));

        // Cap the accumulated target char-wise (multi-byte safe).
        let len = self.critic_target.chars().count();
        if len > CRITIC_TARGET_MAX_CHARS {
            let drop = len - CRITIC_TARGET_MAX_CHARS;
            self.critic_target = self.critic_target.chars().skip(drop).collect();
            self.critic_revealed = self.critic_revealed.saturating_sub(drop);
        }

        self.rebuild_critic();
    }

    /// Advance the typewriter reveal by `reveal_step` characters on each pane.
    ///
    /// Each `*_revealed` count grows by `reveal_step`, clamped to the target's
    /// character count. Only the pane(s) whose revealed count actually changed
    /// are rebuilt, so a fully-typed pane stops re-rendering.
    pub fn tick_reveal(&mut self) {
        let writer_len = self.writer_target.chars().count();
        let new_writer = (self.writer_revealed + self.reveal_step).min(writer_len);
        if new_writer != self.writer_revealed {
            self.writer_revealed = new_writer;
            self.rebuild_writer();
        }

        let critic_len = self.critic_target.chars().count();
        let new_critic = (self.critic_revealed + self.reveal_step).min(critic_len);
        if new_critic != self.critic_revealed {
            self.critic_revealed = new_critic;
            self.rebuild_critic();
        }
    }

    /// Reveal both panes' full targets immediately (skip the typing animation).
    ///
    /// Used by tests and available for shutdown so the final content is shown
    /// in full rather than mid-reveal.
    pub fn reveal_all(&mut self) {
        self.writer_revealed = self.writer_target.chars().count();
        self.critic_revealed = self.critic_target.chars().count();
        self.rebuild_writer();
        self.rebuild_critic();
    }

    /// Advance the apology popup's auto-dismiss timer by one tick.
    ///
    /// Decrements [`App::apology_ttl`] while it is positive; when it reaches 0
    /// the popup is dismissed by clearing [`App::apology_text`]. A no-op once
    /// the timer is already at 0.
    pub fn tick_apology(&mut self) {
        if self.apology_ttl > 0 {
            self.apology_ttl -= 1;
            if self.apology_ttl == 0 {
                self.apology_text = None;
            }
        }
    }

    /// Rebuild the Writer pane buffer from the revealed prefix of its target.
    ///
    /// Takes the first `writer_revealed` characters (char-safe), splits into
    /// lines, repopulates the buffer, and scrolls to the bottom so the view
    /// follows the typing.
    fn rebuild_writer(&self) {
        let prefix: String = self
            .writer_target
            .chars()
            .take(self.writer_revealed)
            .collect();
        if let Ok(mut buf) = self.writer_buffer.write() {
            buf.clear();
            for line in prefix.lines() {
                buf.push(line);
            }
            buf.scroll_to_bottom();
        }
    }

    /// Rebuild the Critic pane buffer from the revealed prefix of its target.
    ///
    /// Takes the first `critic_revealed` characters (char-safe), splits into
    /// lines, repopulates the buffer, and scrolls to the bottom so the view
    /// follows the typing.
    fn rebuild_critic(&self) {
        let prefix: String = self
            .critic_target
            .chars()
            .take(self.critic_revealed)
            .collect();
        if let Ok(mut buf) = self.critic_buffer.write() {
            buf.clear();
            for line in prefix.lines() {
                buf.push(line);
            }
            buf.scroll_to_bottom();
        }
    }
}

// ── Animated flame title ───────────────────────────────────────────

/// The animated title text shown in the banner.
pub const BANNER_TITLE: &str = "AGENT INFERNO";

/// Fire-gradient palette, ordered deep-red → red → orange-red → orange →
/// gold → pale-yellow. Indexed by `(char_index + frame) % PALETTE.len()` so the
/// gradient ripples sideways as the frame counter advances.
const FLAME_PALETTE: [Color; 6] = [
    Color::Rgb(139, 0, 0),     // deep red
    Color::Rgb(220, 20, 20),   // red
    Color::Rgb(255, 69, 0),    // orange-red
    Color::Rgb(255, 140, 0),   // orange
    Color::Rgb(255, 200, 0),   // gold
    Color::Rgb(255, 245, 150), // pale yellow
];

/// Pick the flame color for the character at `char_index` on animation `frame`.
///
/// Pure function of its inputs: the same `(char_index, frame)` always yields
/// the same color, and advancing `frame` rotates the palette so the gradient
/// appears to move across the title.
pub fn flame_color(char_index: usize, frame: u64) -> Color {
    let idx = (char_index + frame as usize) % FLAME_PALETTE.len();
    FLAME_PALETTE[idx]
}

/// Build the animated banner title as a centered `Line` of per-character
/// `Span`s, each colored by [`flame_color`].
fn flame_title_line(frame: u64) -> Line<'static> {
    let spans: Vec<Span<'static>> = BANNER_TITLE
        .chars()
        .enumerate()
        .map(|(i, ch)| {
            Span::styled(
                ch.to_string(),
                Style::default()
                    .fg(flame_color(i, frame))
                    .add_modifier(Modifier::BOLD),
            )
        })
        .collect();
    Line::from(spans).alignment(Alignment::Center)
}

// ── Big ASCII-art block title ──────────────────────────────────────

/// Number of rows in the block font.
const GLYPH_ROWS: usize = 5;

/// Filled-cell character for the block font.
const BLOCK: char = '█';

/// Map a glyph to its 5-row block-font representation.
///
/// Only the characters used in [`BANNER_TITLE`] ("AGENT INFERNO") plus space
/// are covered. Unknown characters render as blank columns. Each row is a
/// `&str` of `█` (filled) and space (blank); rows are kept equal-width per
/// glyph with a trailing blank column acting as the inter-letter gap.
fn glyph_rows(ch: char) -> [&'static str; GLYPH_ROWS] {
    match ch {
        'A' => [
            " ███ ",
            "█   █",
            "█████",
            "█   █",
            "█   █", //
        ],
        'G' => [
            " ████",
            "█    ",
            "█  ██",
            "█   █",
            " ████", //
        ],
        'E' => [
            "█████",
            "█    ",
            "███  ",
            "█    ",
            "█████", //
        ],
        'N' => [
            "█   █",
            "██  █",
            "█ █ █",
            "█  ██",
            "█   █", //
        ],
        'T' => [
            "█████",
            "  █  ",
            "  █  ",
            "  █  ",
            "  █  ", //
        ],
        'I' => [
            "███",
            " █ ",
            " █ ",
            " █ ",
            "███", //
        ],
        'F' => [
            "█████",
            "█    ",
            "███  ",
            "█    ",
            "█    ", //
        ],
        'R' => [
            "████ ",
            "█   █",
            "████ ",
            "█  █ ",
            "█   █", //
        ],
        'O' => [
            " ███ ",
            "█   █",
            "█   █",
            "█   █",
            " ███ ", //
        ],
        // Space (word gap) — a few blank columns.
        _ => [
            "   ", "   ", "   ", "   ", "   ", //
        ],
    }
}

/// Build the big block-art title ("AGENT INFERNO") as five centered [`Line`]s.
///
/// Each glyph's rows are concatenated (with a 1-column gap between glyphs);
/// every filled cell is colored via [`flame_color`] keyed on `col + row` so the
/// fire gradient ripples diagonally through the letters and animates with
/// `frame`. Blank cells stay uncolored spaces.
fn big_title_lines(frame: u64) -> Vec<Line<'static>> {
    // Assemble, per row, the spans across all glyphs of BANNER_TITLE.
    let mut lines: Vec<Line<'static>> = Vec::with_capacity(GLYPH_ROWS);
    let glyphs: Vec<[&'static str; GLYPH_ROWS]> = BANNER_TITLE.chars().map(glyph_rows).collect();

    for row in 0..GLYPH_ROWS {
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut col = 0usize;
        for (g_idx, glyph) in glyphs.iter().enumerate() {
            for cell in glyph[row].chars() {
                if cell == BLOCK {
                    spans.push(Span::styled(
                        BLOCK.to_string(),
                        Style::default()
                            .fg(flame_color(col + row, frame))
                            .add_modifier(Modifier::BOLD),
                    ));
                } else {
                    spans.push(Span::raw(" "));
                }
                col += 1;
            }
            // 1-column gap between glyphs (not after the last).
            if g_idx + 1 < glyphs.len() {
                spans.push(Span::raw(" "));
                col += 1;
            }
        }
        lines.push(Line::from(spans).alignment(Alignment::Center));
    }
    lines
}

/// The sparse ASCII flame glyph set. Heavily biased toward spaces so the flames
/// read as flickering sparks rather than a solid wall.
const FLAME_GLYPHS: [char; 14] = [
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', '\'', '^', '*', '(', ')', ',', '.',
];

/// Build one animated ASCII flame row spanning `width` columns.
///
/// Each column picks a glyph from [`FLAME_GLYPHS`] via a deterministic hash of
/// `(col, frame, phase)` — no RNG, no wall-clock — so the flames flicker and
/// drift as `frame` advances while remaining reproducible for a given input.
/// Non-space glyphs are colored via [`flame_color`].
fn flame_row(width: u16, frame: u64, phase: u64) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::with_capacity(width as usize);
    for col in 0..width as u64 {
        // Simple deterministic wrapping mul/xor hash of the three inputs.
        let mut h = col
            .wrapping_mul(2_654_435_761)
            .wrapping_add(frame.wrapping_mul(40_503))
            .wrapping_add(phase.wrapping_mul(2_246_822_519));
        h ^= h >> 15;
        h = h.wrapping_mul(2_246_822_519);
        h ^= h >> 13;
        let glyph = FLAME_GLYPHS[(h as usize) % FLAME_GLYPHS.len()];
        if glyph == ' ' {
            spans.push(Span::raw(" "));
        } else {
            spans.push(Span::styled(
                glyph.to_string(),
                Style::default().fg(flame_color(col as usize, frame)),
            ));
        }
    }
    Line::from(spans).alignment(Alignment::Center)
}

/// Format the token-meter line shown beneath the flame title.
fn token_meter_line(app: &App) -> String {
    let task = if app.task.is_empty() {
        String::new()
    } else {
        format!("Task: {}    ", app.task)
    };
    format!(
        "{task}Tokens: {}  (Writer {} · Critic {} · Apology {})",
        app.total_tokens, app.writer_tokens, app.critic_tokens, app.apology_tokens,
    )
}

/// Render the three-pane TUI layout into the given frame.
///
/// Layout is a vertical split — top banner (11 rows of big ASCII art + flames
/// on a wide terminal, else a 5-row compact flame title), main panes, status
/// bar (3 rows) — then a horizontal split of the middle chunk (50% Writer, 50%
/// Critic). When an apology is active (and no error), a centered popup is drawn
/// over the panes. If the terminal is smaller than 80×24 a centered warning
/// replaces the layout.
pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();

    if area.width < 80 || area.height < 24 {
        let msg = format!(
            "Terminal too small. Minimum: 80x24. Current: {}x{}",
            area.width, area.height
        );
        let paragraph = Paragraph::new(msg)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Red));
        frame.render_widget(paragraph, area);
        return;
    }

    // Adaptive banner: a wide/tall terminal gets the big ASCII-art title with
    // flame rows; otherwise the compact single-line flame title is used.
    let big = area.width >= 100 && area.height >= 30;
    let banner_height: u16 = if big { 11 } else { 5 };

    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(banner_height),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(area);
    let banner_area = vertical[0];
    let main_area = vertical[1];
    let status_area = vertical[2];

    // ── Banner: animated AGENT INFERNO title + token meter ────────────
    let banner_lines: Vec<Line<'static>> = if big {
        // Inner width inside the borders for the flame rows.
        let inner_w = banner_area.width.saturating_sub(2);
        let mut lines: Vec<Line<'static>> = Vec::with_capacity(GLYPH_ROWS + 4);
        lines.push(flame_row(inner_w, app.frame, 1));
        lines.extend(big_title_lines(app.frame));
        lines.push(flame_row(inner_w, app.frame, 2));
        lines.push(Line::from(""));
        lines.push(Line::from(token_meter_line(app)).alignment(Alignment::Center));
        lines
    } else {
        vec![
            flame_title_line(app.frame),
            Line::from(token_meter_line(app)).alignment(Alignment::Center),
        ]
    };
    let banner = Paragraph::new(ratatui::text::Text::from(banner_lines))
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);
    frame.render_widget(banner, banner_area);

    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(main_area);
    let writer_area = horizontal[0];
    let critic_area = horizontal[1];

    let focus_style = Style::default().fg(Color::Yellow);
    let default_style = Style::default();
    let critic_unfocused_style = Style::default().fg(Color::Red);

    let writer_border = if app.focused_pane == FocusTarget::Writer {
        focus_style
    } else {
        default_style
    };
    let critic_border = if app.focused_pane == FocusTarget::Critic {
        focus_style
    } else {
        critic_unfocused_style
    };

    let inner_h = writer_area.height.saturating_sub(2) as usize;
    let writer_text = {
        let buf = app
            .writer_buffer
            .read()
            .expect("writer_buffer RwLock poisoned");
        let lines = buf.visible_lines(inner_h);
        lines.join("\n")
    };
    let writer_title = if app.writer_version > 0 {
        format!("Writer [v{}]", app.writer_version)
    } else {
        "Writer".to_string()
    };
    let writer_paragraph = Paragraph::new(writer_text)
        .block(
            Block::default()
                .title(writer_title)
                .borders(Borders::ALL)
                .border_style(writer_border),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(writer_paragraph, writer_area);

    let inner_h = critic_area.height.saturating_sub(2) as usize;
    let critic_text = {
        let buf = app
            .critic_buffer
            .read()
            .expect("critic_buffer RwLock poisoned");
        let lines = buf.visible_lines(inner_h);
        lines.join("\n")
    };
    let critic_title = if app.critic_version > 0 {
        format!("Critic [v{}]", app.critic_version)
    } else {
        "Critic".to_string()
    };
    let critic_paragraph = Paragraph::new(critic_text)
        .block(
            Block::default()
                .title(critic_title)
                .borders(Borders::ALL)
                .border_style(critic_border),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(critic_paragraph, critic_area);

    // ── Bottom bar: error (priority) else slim status ─────────────────
    if let Some(error) = &app.error {
        let error_paragraph = Paragraph::new(error.to_string())
            .block(
                Block::default()
                    .title("Error")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Red)),
            )
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD));
        frame.render_widget(error_paragraph, status_area);
    } else {
        let cooldown_info = match app.apology_cooldown {
            Some(secs) if secs > 0 => format!("  │  Apology cooldown: {secs}s"),
            _ => String::new(),
        };
        let state_prefix = match app.state {
            AppState::Idle | AppState::Running => "Running…",
            AppState::Stopping => "Stopping…",
            AppState::Done => "Done",
        };
        let status = format!(
            "Status: {state_prefix}  │  ${:.2} / ${:.2}  │  Esc to stop{cooldown_info}",
            app.cost_spent, app.cost_limit,
        );
        let status_paragraph = Paragraph::new(status)
            .block(Block::default().title("Status").borders(Borders::ALL))
            .alignment(Alignment::Center);
        frame.render_widget(status_paragraph, status_area);
    }

    // ── Apology: centered popup overlay (no error in flight) ──────────
    if app.error.is_none() {
        if let Some(apology) = &app.apology_text {
            // Height roughly tracks the wrapped content but is capped.
            let popup = centered_rect(60, 40, area);
            frame.render_widget(Clear, popup);
            let apology_paragraph = Paragraph::new(apology.clone())
                .block(Block::default().title("Apology").borders(Borders::ALL))
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: true })
                .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD));
            frame.render_widget(apology_paragraph, popup);
        }
    }
}

/// Compute a horizontally and vertically centered `Rect` within `area`.
///
/// The result is `percent_x` percent of `area`'s width, and its height is
/// `max_height` percent of `area`'s height, each floored at a few rows so the
/// box always has a usable interior. Used to position the apology popup.
fn centered_rect(percent_x: u16, percent_max_height: u16, area: Rect) -> Rect {
    let width = (area.width * percent_x / 100).max(20).min(area.width);
    let height = (area.height * percent_max_height / 100)
        .max(5)
        .min(area.height);
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    Rect {
        x,
        y,
        width,
        height,
    }
}
