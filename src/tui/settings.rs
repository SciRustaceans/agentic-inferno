//! The live, in-TUI settings menu (opened with `s` while the spectacle runs).
//!
//! Mirrors the apology popup's modal pattern: a centered overlay drawn last,
//! over the panes. The menu stages changes in a [`RuntimeSettings`] *draft* and
//! applies them to the shared `Arc<RwLock<RuntimeSettings>>` on Enter, so the
//! Writer/Critic loops pick them up on their next cycle.

use clap::ValueEnum;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

use crate::config::{CriticStyle, RuntimeSettings, Speed};
use crate::tui::input::ControlFlow;
use crate::tui::ui::{centered_rect, App};

/// Common model presets offered in the Writer/Critic model dropdowns.
///
/// A synthetic `Custom…` entry is appended at the dropdown level (it is not a
/// real model) to switch the field into free-text editing.
pub const MODEL_PRESETS: &[&str] = &[
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
    "deepseek-chat",
    "deepseek-reasoner",
    "claude-sonnet-4-20250514",
    "claude-haiku",
    "kimi-k2",
];

/// Label shown for the synthetic free-text dropdown entry.
const CUSTOM_LABEL: &str = "Custom…";

/// Number of selectable fields (0..=5).
const FIELD_COUNT: usize = 6;

// Field indices.
const FIELD_WRITER: usize = 0;
const FIELD_CRITIC: usize = 1;
const FIELD_TONE: usize = 2;
const FIELD_SPEED: usize = 3;
const FIELD_COST: usize = 4;
const FIELD_PROMPT: usize = 5;

/// Cost-cap stepper increment, in USD.
const COST_STEP: f64 = 0.05;
/// Hard upper bound on the cost cap, in USD.
const COST_MAX: f64 = 1000.0;

/// Staged state for the live settings overlay.
///
/// The derived `Default` gives a closed menu (`open=false`), `field=0`,
/// `draft = RuntimeSettings::default()`, empty buffers, `editing=false`, and no
/// message — exactly the desired initial state.
#[derive(Default)]
pub struct SettingsMenu {
    /// Whether the overlay is currently shown (and capturing all keys).
    pub open: bool,
    /// Selected field, `0..=5`: WriterModel, CriticModel, CriticTone, Speed,
    /// CostCap, Prompt.
    pub field: usize,
    /// Staged settings, initialised from `runtime` on open, applied on Enter.
    pub draft: RuntimeSettings,
    /// Free-text buffer for the Writer model (used when the dropdown is on
    /// `Custom…`).
    pub writer_buf: String,
    /// Free-text buffer for the Critic model (used when on `Custom…`).
    pub critic_buf: String,
    /// True while a model field's dropdown is parked on `Custom…` (text entry).
    pub editing: bool,
    /// Inline message (e.g. a model-validation error) shown in red.
    pub message: Option<String>,
}

impl SettingsMenu {
    /// Construct a fresh, closed menu with default draft values.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Open the settings menu: snapshot the current runtime into the draft.
///
/// Seeds the writer/critic free-text buffers from the current models so a
/// non-preset model shows up as `Custom…` with its text pre-filled.
pub fn open_menu(app: &mut App) {
    let draft = app.runtime.read().map(|g| g.clone()).unwrap_or_default();
    app.settings.writer_buf = draft.writer_model.clone();
    app.settings.critic_buf = draft.critic_model.clone();
    app.settings.draft = draft;
    app.settings.field = 0;
    app.settings.editing = false;
    app.settings.message = None;
    app.settings.open = true;
}

/// Handle a key while the settings menu is open. Always returns
/// `ControlFlow::Continue` — the menu never stops the app; `Esc` only closes it.
pub fn handle_menu_key(app: &mut App, key: KeyEvent) -> ControlFlow {
    match key.code {
        KeyCode::Esc => {
            app.settings.open = false;
            app.settings.message = None;
        }
        KeyCode::Enter => apply_menu(app),
        KeyCode::Up | KeyCode::BackTab => {
            app.settings.field = (app.settings.field + FIELD_COUNT - 1) % FIELD_COUNT;
            sync_editing(app);
        }
        KeyCode::Down | KeyCode::Tab => {
            app.settings.field = (app.settings.field + 1) % FIELD_COUNT;
            sync_editing(app);
        }
        KeyCode::Left => cycle_field(app, -1),
        KeyCode::Right => cycle_field(app, 1),
        KeyCode::Char(c) => edit_text(app, Some(c)),
        KeyCode::Backspace => edit_text(app, None),
        _ => {}
    }
    ControlFlow::Continue
}

/// Validate (if a config is present) and apply the draft to the shared runtime.
fn apply_menu(app: &mut App) {
    if let Some(cfg) = app.config.clone() {
        if let Err(e) =
            crate::orchestrator::validate_model(&app.settings.draft.writer_model, "Writer", &cfg)
        {
            app.settings.message = Some(e.to_string());
            return;
        }
        if let Err(e) =
            crate::orchestrator::validate_model(&app.settings.draft.critic_model, "Critic", &cfg)
        {
            app.settings.message = Some(e.to_string());
            return;
        }
    }

    let draft = app.settings.draft.clone();
    if let Ok(mut guard) = app.runtime.write() {
        *guard = draft.clone();
    }
    app.reveal_step = (draft.speed.cps() as usize / 8).max(1);
    app.settings.open = false;
    app.settings.message = None;
}

/// Keep `editing` in sync with whether the focused model field is on `Custom…`.
fn sync_editing(app: &mut App) {
    app.settings.editing = match app.settings.field {
        FIELD_WRITER => !MODEL_PRESETS.contains(&app.settings.draft.writer_model.as_str()),
        FIELD_CRITIC => !MODEL_PRESETS.contains(&app.settings.draft.critic_model.as_str()),
        _ => false,
    };
}

/// Cycle the focused dropdown field by `dir` (+1 right / -1 left).
fn cycle_field(app: &mut App, dir: i32) {
    match app.settings.field {
        FIELD_WRITER => cycle_model(app, dir, true),
        FIELD_CRITIC => cycle_model(app, dir, false),
        FIELD_TONE => {
            let variants = CriticStyle::value_variants();
            app.settings.draft.critic_style =
                cycle_enum(variants, app.settings.draft.critic_style, dir);
        }
        FIELD_SPEED => {
            let variants = Speed::value_variants();
            app.settings.draft.speed = cycle_enum(variants, app.settings.draft.speed, dir);
        }
        FIELD_COST => {
            let floor = app.cost_spent.max(0.01);
            let stepped = app.settings.draft.max_cost_usd + (dir as f64) * COST_STEP;
            let clamped = stepped.clamp(floor, COST_MAX);
            // Round to 2 decimals to avoid float drift in the displayed cap.
            app.settings.draft.max_cost_usd = (clamped * 100.0).round() / 100.0;
        }
        // Prompt is a text field — Left/Right do nothing.
        _ => {}
    }
}

/// Cycle the Writer (`is_writer`) or Critic model dropdown by `dir`.
///
/// The option list is `MODEL_PRESETS` followed by a synthetic `Custom…` slot.
/// Landing on a preset sets the model to it and leaves edit mode; landing on
/// `Custom…` enters edit mode and restores the model from the free-text buffer.
fn cycle_model(app: &mut App, dir: i32, is_writer: bool) {
    let n = MODEL_PRESETS.len() + 1; // +1 for Custom…
    let custom_idx = MODEL_PRESETS.len();
    let current = if is_writer {
        &app.settings.draft.writer_model
    } else {
        &app.settings.draft.critic_model
    };
    let cur_idx = MODEL_PRESETS
        .iter()
        .position(|m| *m == current.as_str())
        .unwrap_or(custom_idx);
    let next_idx = (((cur_idx as i32 + dir).rem_euclid(n as i32)) as usize) % n;

    if next_idx == custom_idx {
        app.settings.editing = true;
        let buf = if is_writer {
            app.settings.writer_buf.clone()
        } else {
            app.settings.critic_buf.clone()
        };
        if is_writer {
            app.settings.draft.writer_model = buf;
        } else {
            app.settings.draft.critic_model = buf;
        }
    } else {
        app.settings.editing = false;
        let preset = MODEL_PRESETS[next_idx].to_string();
        if is_writer {
            app.settings.draft.writer_model = preset;
        } else {
            app.settings.draft.critic_model = preset;
        }
    }
}

/// Cycle through an enum's `value_variants()` by `dir`, wrapping around.
fn cycle_enum<T: Copy + PartialEq>(variants: &[T], current: T, dir: i32) -> T {
    if variants.is_empty() {
        return current;
    }
    let n = variants.len() as i32;
    let cur = variants.iter().position(|v| *v == current).unwrap_or(0) as i32;
    let next = (cur + dir).rem_euclid(n) as usize;
    variants[next]
}

/// Edit the focused text field. `ch = Some(c)` appends; `None` is a backspace.
///
/// Only the Prompt field (always) and a model field parked on `Custom…` are
/// editable. Single-line: append/pop only, no cursor movement.
fn edit_text(app: &mut App, ch: Option<char>) {
    match app.settings.field {
        FIELD_PROMPT => {
            let mut s = app.settings.draft.prompt.take().unwrap_or_default();
            apply_edit(&mut s, ch);
            app.settings.draft.prompt = Some(s);
        }
        FIELD_WRITER if app.settings.editing => {
            apply_edit(&mut app.settings.writer_buf, ch);
            app.settings.draft.writer_model = app.settings.writer_buf.clone();
        }
        FIELD_CRITIC if app.settings.editing => {
            apply_edit(&mut app.settings.critic_buf, ch);
            app.settings.draft.critic_model = app.settings.critic_buf.clone();
        }
        _ => {}
    }
}

/// Apply a single append-or-backspace edit to `s` (printable chars only).
fn apply_edit(s: &mut String, ch: Option<char>) {
    match ch {
        Some(c) if !c.is_control() => s.push(c),
        Some(_) => {}
        None => {
            s.pop();
        }
    }
}

// ── Render ─────────────────────────────────────────────────────────

/// Render the settings modal: a centered, bordered `Settings` overlay with one
/// row per field plus a footer (and an error line when present).
pub fn render_settings(frame: &mut Frame, app: &App, area: Rect) {
    let popup = centered_rect(70, 70, area);
    frame.render_widget(Clear, popup);

    let s = &app.settings;
    let prompt_active = app.task == "prompt";

    let mut lines: Vec<Line<'static>> = Vec::with_capacity(FIELD_COUNT + 3);
    lines.push(field_line(
        s.field == FIELD_WRITER,
        "Writer model",
        &dropdown(&model_display(&s.draft.writer_model)),
        false,
    ));
    lines.push(field_line(
        s.field == FIELD_CRITIC,
        "Critic model",
        &dropdown(&model_display(&s.draft.critic_model)),
        false,
    ));
    lines.push(field_line(
        s.field == FIELD_TONE,
        "Critic tone",
        &dropdown(&s.draft.critic_style.to_string()),
        false,
    ));
    lines.push(field_line(
        s.field == FIELD_SPEED,
        "Speed",
        &dropdown(&s.draft.speed.to_string()),
        false,
    ));
    lines.push(field_line(
        s.field == FIELD_COST,
        "Max cost",
        &dropdown(&format!("${:.2}", s.draft.max_cost_usd)),
        false,
    ));
    let prompt_val = s.draft.prompt.clone().unwrap_or_default();
    lines.push(field_line(
        s.field == FIELD_PROMPT,
        "Prompt (prompt mode only)",
        &prompt_val,
        !prompt_active,
    ));

    lines.push(Line::from(""));
    lines.push(
        Line::from("↑↓ field · ←→ change · Enter apply · Esc cancel")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center),
    );
    if let Some(msg) = &s.message {
        lines.push(
            Line::from(msg.clone())
                .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
                .alignment(Alignment::Center),
        );
    }

    let paragraph = Paragraph::new(lines)
        .block(Block::default().title("Settings").borders(Borders::ALL))
        .alignment(Alignment::Left);
    frame.render_widget(paragraph, popup);
}

/// Build a single `"{marker} {label}: {value}"` line, highlighted when
/// selected and dimmed when `dim` (the inactive Prompt row).
fn field_line(selected: bool, label: &str, value: &str, dim: bool) -> Line<'static> {
    let marker = if selected { ">" } else { " " };
    let text = format!("{marker} {label}: {value}");
    let style = if selected {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else if dim {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
    };
    Line::from(Span::styled(text, style))
}

/// Wrap a dropdown value in the `◄ value ►` chrome.
fn dropdown(value: &str) -> String {
    format!("◄ {value} ►")
}

/// Display a model string, falling back to `Custom…(text)` for non-presets.
fn model_display(model: &str) -> String {
    if MODEL_PRESETS.contains(&model) {
        model.to_string()
    } else if model.is_empty() {
        format!("{CUSTOM_LABEL}(…)")
    } else {
        format!("{CUSTOM_LABEL}({model})")
    }
}
