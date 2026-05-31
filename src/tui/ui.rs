use std::sync::{Arc, RwLock};

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
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
}

impl App {
    pub fn new() -> Self {
        Self {
            writer_buffer: Arc::new(RwLock::new(PaneBuffer::new())),
            critic_buffer: Arc::new(RwLock::new(PaneBuffer::new())),
            apology_text: None,
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
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Replace the Writer pane with a new document revision.
    ///
    /// The writer pane shows the current living document, so each revision
    /// clears the buffer and re-fills it line-by-line (one buffer line per
    /// text line, so scrolling moves by lines, not whole revisions), then
    /// scrolls to the top to read the document from the start.
    ///
    /// Empty `text` leaves the buffer empty — `"".lines()` yields no lines,
    /// so no panic and no spurious blank line.
    pub fn apply_writer_output(&self, text: &str) {
        if let Ok(mut buf) = self.writer_buffer.write() {
            buf.clear();
            for line in text.lines() {
                buf.push(line);
            }
            buf.scroll_to_top();
        }
    }

    /// Append a new critique to the Critic pane.
    ///
    /// The critic pane accumulates critiques: each call pushes a
    /// `── v{critic_version} ──` header followed by the critique text
    /// line-by-line, then scrolls to the bottom so the newest critique is
    /// visible. `critic_version` is read at call time, so the orchestrator
    /// must emit `CritiqueReady` before `CriticOutput`.
    pub fn apply_critic_output(&self, text: &str) {
        if let Ok(mut buf) = self.critic_buffer.write() {
            buf.push(&format!("── v{} ──", self.critic_version));
            for line in text.lines() {
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
/// Layout is a vertical split — top banner (5 rows), main panes, status bar
/// (3 rows) — then a horizontal split of the middle chunk (50% Writer, 50%
/// Critic). If the terminal is smaller than 80×24 a centered warning replaces
/// the layout.
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

    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(area);
    let banner_area = vertical[0];
    let main_area = vertical[1];
    let apology_area = vertical[2];

    // ── Banner: animated AGENT INFERNO title + token meter ────────────
    let banner_text = ratatui::text::Text::from(vec![
        flame_title_line(app.frame),
        Line::from(token_meter_line(app)).alignment(Alignment::Center),
    ]);
    let banner = Paragraph::new(banner_text)
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
        frame.render_widget(error_paragraph, apology_area);
    } else if let Some(apology) = &app.apology_text {
        let apology_paragraph = Paragraph::new(apology.clone())
            .block(Block::default().title("Apology").borders(Borders::ALL))
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD));
        frame.render_widget(apology_paragraph, apology_area);
    } else {
        let wv = app.writer_version;
        let cv = app.critic_version;
        let version_info = if wv > 0 || cv > 0 {
            format!(" | Writer v{wv} | Critic v{cv}")
        } else {
            String::new()
        };
        let cooldown_info = match app.apology_cooldown {
            Some(secs) if secs > 0 => format!(" | Apology cooldown: {secs}s"),
            _ => String::new(),
        };
        let state_prefix = match app.state {
            AppState::Idle | AppState::Running => "Running…",
            AppState::Stopping => "Stopping…",
            AppState::Done => "Done",
        };
        let status = format!(
            "{state_prefix} | Cost: ${:.2}/${:.2}{version_info} | Writer: ${:.4} | Critic: ${:.4} | Apologies: ${:.4}{cooldown_info} | Esc to stop",
            app.cost_spent, app.cost_limit, app.writer_cost, app.critic_cost, app.apology_cost,
        );
        let status_paragraph = Paragraph::new(status)
            .block(Block::default().title("Status").borders(Borders::ALL))
            .alignment(Alignment::Center);
        frame.render_widget(status_paragraph, apology_area);
    }
}
