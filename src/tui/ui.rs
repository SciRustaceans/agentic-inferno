use std::sync::{Arc, RwLock};

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
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
    /// Monotonic version counter for Writer content — bumped on each `WriterOutput` event.
    pub writer_version: u64,
    /// Monotonic version counter for Critic content — bumped on each `CriticOutput` event.
    pub critic_version: u64,
    /// Which pane currently has keyboard focus (highlighted border).
    pub focused_pane: FocusTarget,
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
            writer_version: 0,
            critic_version: 0,
            focused_pane: FocusTarget::default(),
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

/// Render the three-pane TUI layout into the given frame.
///
/// Layout is vertical split (85% main, 3-row apology bar) then horizontal
/// split (50% Writer, 50% Critic). If the terminal is smaller than 80×24 a
/// centered warning replaces the layout.
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
        .constraints([Constraint::Percentage(85), Constraint::Length(3)])
        .split(area);
    let main_area = vertical[0];
    let apology_area = vertical[1];

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

    let writer_text = app
        .writer_buffer
        .read()
        .expect("writer_buffer RwLock poisoned")
        .content();
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

    let critic_text = app
        .critic_buffer
        .read()
        .expect("critic_buffer RwLock poisoned")
        .content();
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
            .style(
                Style::default()
                    .fg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            );
        frame.render_widget(error_paragraph, apology_area);
    } else if let Some(apology) = &app.apology_text {
        let apology_paragraph = Paragraph::new(apology.clone())
            .block(
                Block::default()
                    .title("Penance")
                    .borders(Borders::ALL),
            )
            .alignment(Alignment::Center)
            .style(
                Style::default()
                    .fg(Color::Red)
                    .add_modifier(Modifier::RAPID_BLINK | Modifier::BOLD),
            );
        frame.render_widget(apology_paragraph, apology_area);
    } else {
        let wv = app.writer_version;
        let cv = app.critic_version;
        let version_info = if wv > 0 || cv > 0 {
            format!(" | Writer v{wv} | Critic v{cv}")
        } else {
            String::new()
        };
        let status = format!(
            "Running... | Cost: ${:.2}/${:.2}{version_info} | Esc to stop",
            app.cost_spent, app.cost_limit,
        );
        let status_paragraph = Paragraph::new(status)
            .block(Block::default().title("Penance").borders(Borders::ALL))
            .alignment(Alignment::Center);
        frame.render_widget(status_paragraph, apology_area);
    }
}
