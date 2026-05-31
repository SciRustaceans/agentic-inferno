//! Settings-menu tests — the live in-TUI settings overlay (opened with `s`).
//!
//! Covers the pure menu state machine (open seeds draft from runtime; Up/Down
//! moves field; Left/Right cycles dropdowns and steps the cost cap; typing
//! edits the prompt buffer; Enter writes draft → runtime; Esc discards),
//! rendering, input routing, and the C2 non-wrapping token meter.
//!
//! `app.config` is left `None` throughout so Enter applies without API-key
//! validation.

use std::sync::Arc;

use agentic_inferno::config::{Config, CriticStyle, InfernoTask, RuntimeSettings, Speed};
use agentic_inferno::tui::input::{self, ControlFlow};
use agentic_inferno::tui::settings::{self, MODEL_PRESETS};
use agentic_inferno::tui::ui::{self, App};

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::Terminal;

// ── Helpers ───────────────────────────────────────────────────────

fn press(key_code: KeyCode) -> KeyEvent {
    KeyEvent {
        code: key_code,
        modifiers: KeyModifiers::empty(),
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

/// A known, non-default runtime so we can detect that the draft was seeded and
/// that Enter writes it back unchanged when nothing was edited.
fn known_runtime() -> RuntimeSettings {
    RuntimeSettings {
        writer_model: "deepseek-chat".into(),
        critic_model: "deepseek-reasoner".into(),
        critic_style: CriticStyle::Theatrical,
        speed: Speed::Slow,
        prompt: Some("seed prompt".into()),
        max_cost_usd: 1.50,
    }
}

/// An App whose `runtime` holds `known_runtime()` and `config` is None.
fn app_with_runtime() -> App {
    let app = App::new();
    *app.runtime.write().unwrap() = known_runtime();
    app
}

fn render_app(app: &App, width: u16, height: u16) -> Buffer {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal creation");
    terminal.draw(|frame| ui::render(frame, app)).expect("draw");
    terminal.backend().buffer().clone()
}

fn buffer_text(buffer: &Buffer) -> String {
    let area = buffer.area();
    let mut lines: Vec<String> = Vec::with_capacity(area.height as usize);
    for y in 0..area.height {
        let mut row = String::with_capacity(area.width as usize);
        for x in 0..area.width {
            row.push_str(buffer.cell((x, y)).map(|c| c.symbol()).unwrap_or(" "));
        }
        lines.push(row.trim_end().to_string());
    }
    lines.join("\n")
}

// ── Open seeds the draft from runtime ─────────────────────────────

#[test]
fn open_menu_seeds_draft_from_runtime() {
    let mut app = app_with_runtime();
    settings::open_menu(&mut app);

    assert!(app.settings.open, "menu should be open after open_menu");
    assert_eq!(app.settings.field, 0, "field resets to 0 on open");
    assert!(app.settings.message.is_none());
    let d = &app.settings.draft;
    assert_eq!(d.writer_model, "deepseek-chat");
    assert_eq!(d.critic_model, "deepseek-reasoner");
    assert_eq!(d.critic_style, CriticStyle::Theatrical);
    assert_eq!(d.speed, Speed::Slow);
    assert_eq!(d.prompt.as_deref(), Some("seed prompt"));
    assert_eq!(d.max_cost_usd, 1.50);
    // Buffers seeded from the models.
    assert_eq!(app.settings.writer_buf, "deepseek-chat");
    assert_eq!(app.settings.critic_buf, "deepseek-reasoner");
}

// ── Up/Down moves field ───────────────────────────────────────────

#[test]
fn updown_moves_field_and_wraps() {
    let mut app = app_with_runtime();
    settings::open_menu(&mut app);
    assert_eq!(app.settings.field, 0);

    settings::handle_menu_key(&mut app, press(KeyCode::Down));
    assert_eq!(app.settings.field, 1);
    settings::handle_menu_key(&mut app, press(KeyCode::Down));
    assert_eq!(app.settings.field, 2);
    settings::handle_menu_key(&mut app, press(KeyCode::Up));
    assert_eq!(app.settings.field, 1);

    // Wrap below 0.
    settings::handle_menu_key(&mut app, press(KeyCode::Up));
    assert_eq!(app.settings.field, 0);
    settings::handle_menu_key(&mut app, press(KeyCode::Up));
    assert_eq!(app.settings.field, 5, "Up from 0 wraps to last field");
    // Wrap above last.
    settings::handle_menu_key(&mut app, press(KeyCode::Down));
    assert_eq!(app.settings.field, 0, "Down from last wraps to 0");
}

// ── Left/Right cycles dropdowns ────────────────────────────────────

#[test]
fn leftright_cycles_critic_tone() {
    let mut app = app_with_runtime();
    settings::open_menu(&mut app);
    // Move to CriticTone (field 2).
    settings::handle_menu_key(&mut app, press(KeyCode::Down));
    settings::handle_menu_key(&mut app, press(KeyCode::Down));
    assert_eq!(app.settings.field, 2);

    let before = app.settings.draft.critic_style;
    settings::handle_menu_key(&mut app, press(KeyCode::Right));
    let after = app.settings.draft.critic_style;
    assert_ne!(before, after, "Right should change the critic tone");

    // Right then Left returns to the original.
    settings::handle_menu_key(&mut app, press(KeyCode::Left));
    assert_eq!(app.settings.draft.critic_style, before);
}

#[test]
fn leftright_cycles_speed() {
    let mut app = app_with_runtime();
    settings::open_menu(&mut app);
    // Move to Speed (field 3).
    for _ in 0..3 {
        settings::handle_menu_key(&mut app, press(KeyCode::Down));
    }
    assert_eq!(app.settings.field, 3);

    let before = app.settings.draft.speed; // Slow
    settings::handle_menu_key(&mut app, press(KeyCode::Right));
    assert_ne!(
        before, app.settings.draft.speed,
        "Right should change speed"
    );
    settings::handle_menu_key(&mut app, press(KeyCode::Left));
    assert_eq!(app.settings.draft.speed, before, "Left reverts the speed");
}

// ── Left/Right steps the cost cap (with floor clamp) ──────────────

#[test]
fn leftright_steps_cost_cap() {
    let mut app = app_with_runtime();
    settings::open_menu(&mut app);
    // Move to CostCap (field 4).
    for _ in 0..4 {
        settings::handle_menu_key(&mut app, press(KeyCode::Down));
    }
    assert_eq!(app.settings.field, 4);

    let start = app.settings.draft.max_cost_usd; // 1.50
    settings::handle_menu_key(&mut app, press(KeyCode::Right));
    assert_eq!(app.settings.draft.max_cost_usd, 1.55);
    settings::handle_menu_key(&mut app, press(KeyCode::Left));
    settings::handle_menu_key(&mut app, press(KeyCode::Left));
    assert_eq!(app.settings.draft.max_cost_usd, 1.45);
    let _ = start;
}

#[test]
fn cost_cap_floor_clamps_to_current_spend() {
    let mut app = app_with_runtime();
    app.cost_spent = 1.49; // floor
    settings::open_menu(&mut app);
    for _ in 0..4 {
        settings::handle_menu_key(&mut app, press(KeyCode::Down));
    }
    assert_eq!(app.settings.field, 4);
    assert_eq!(app.settings.draft.max_cost_usd, 1.50);

    // One step down would be 1.45, but the floor is current spend 1.49.
    settings::handle_menu_key(&mut app, press(KeyCode::Left));
    assert_eq!(
        app.settings.draft.max_cost_usd, 1.49,
        "cost cap must not drop below current spend"
    );
    // Further Left stays clamped.
    settings::handle_menu_key(&mut app, press(KeyCode::Left));
    assert_eq!(app.settings.draft.max_cost_usd, 1.49);
}

// ── Typing edits the prompt buffer ────────────────────────────────

#[test]
fn typing_edits_prompt() {
    let mut app = app_with_runtime();
    settings::open_menu(&mut app);
    // Move to Prompt (field 5).
    for _ in 0..5 {
        settings::handle_menu_key(&mut app, press(KeyCode::Down));
    }
    assert_eq!(app.settings.field, 5);
    // Clear the seeded prompt with backspaces, then type "hi".
    for _ in 0..("seed prompt".len()) {
        settings::handle_menu_key(&mut app, press(KeyCode::Backspace));
    }
    assert_eq!(app.settings.draft.prompt.as_deref(), Some(""));
    settings::handle_menu_key(&mut app, press(KeyCode::Char('h')));
    settings::handle_menu_key(&mut app, press(KeyCode::Char('i')));
    assert_eq!(app.settings.draft.prompt.as_deref(), Some("hi"));
}

// ── Enter writes draft → runtime; Esc discards ────────────────────

#[test]
fn enter_writes_draft_to_runtime() {
    let mut app = app_with_runtime();
    settings::open_menu(&mut app);
    // Change the critic tone, then apply.
    for _ in 0..2 {
        settings::handle_menu_key(&mut app, press(KeyCode::Down));
    }
    let original = app.runtime.read().unwrap().critic_style;
    settings::handle_menu_key(&mut app, press(KeyCode::Right));
    let staged = app.settings.draft.critic_style;
    assert_ne!(original, staged);

    let flow = settings::handle_menu_key(&mut app, press(KeyCode::Enter));
    assert_eq!(flow, ControlFlow::Continue);
    assert!(!app.settings.open, "Enter closes the menu");
    assert_eq!(
        app.runtime.read().unwrap().critic_style,
        staged,
        "Enter must write the staged draft into runtime"
    );
}

#[test]
fn enter_updates_reveal_step_from_speed() {
    let mut app = app_with_runtime();
    settings::open_menu(&mut app);
    // Move to Speed and set it to Fast (cps 80 → reveal_step 10).
    for _ in 0..3 {
        settings::handle_menu_key(&mut app, press(KeyCode::Down));
    }
    // From Slow → Fast: cycle Right twice (Slow→Normal→Fast).
    settings::handle_menu_key(&mut app, press(KeyCode::Right));
    settings::handle_menu_key(&mut app, press(KeyCode::Right));
    assert_eq!(app.settings.draft.speed, Speed::Fast);

    settings::handle_menu_key(&mut app, press(KeyCode::Enter));
    assert_eq!(app.reveal_step, (Speed::Fast.cps() as usize / 8).max(1));
}

#[test]
fn esc_discards_draft_and_leaves_runtime_unchanged() {
    let mut app = app_with_runtime();
    let before = app.runtime.read().unwrap().clone();
    settings::open_menu(&mut app);
    // Edit several fields.
    for _ in 0..2 {
        settings::handle_menu_key(&mut app, press(KeyCode::Down));
    }
    settings::handle_menu_key(&mut app, press(KeyCode::Right)); // change tone

    let flow = settings::handle_menu_key(&mut app, press(KeyCode::Esc));
    assert_eq!(flow, ControlFlow::Continue);
    assert!(!app.settings.open, "Esc closes the menu");

    let after = app.runtime.read().unwrap().clone();
    assert_eq!(
        before.critic_style, after.critic_style,
        "Esc must leave runtime unchanged"
    );
    assert_eq!(before.writer_model, after.writer_model);
    assert_eq!(before.max_cost_usd, after.max_cost_usd);
}

// ── Model dropdown cycling lands on presets / Custom ──────────────

#[test]
fn writer_model_dropdown_cycles_presets_and_custom() {
    let mut app = app_with_runtime();
    settings::open_menu(&mut app);
    // Field 0 is WriterModel; seeded model "deepseek-chat" is a preset.
    assert!(MODEL_PRESETS.contains(&"deepseek-chat"));
    let idx = MODEL_PRESETS
        .iter()
        .position(|m| *m == "deepseek-chat")
        .unwrap();

    // Right lands on the next preset.
    settings::handle_menu_key(&mut app, press(KeyCode::Right));
    assert_eq!(app.settings.draft.writer_model, MODEL_PRESETS[idx + 1]);

    // Cycle all the way to the synthetic Custom… slot (one past the last preset).
    settings::open_menu(&mut app); // reset
                                   // Go Left once from the first time we land — simpler: step Left from a preset.
                                   // Walk forward through every preset then onto Custom….
    let mut seen_custom = false;
    for _ in 0..(MODEL_PRESETS.len() + 1) {
        settings::handle_menu_key(&mut app, press(KeyCode::Right));
        if app.settings.editing {
            seen_custom = true;
        }
    }
    assert!(
        seen_custom,
        "cycling the writer dropdown must reach the Custom… (editing) slot"
    );
}

// ── Render ─────────────────────────────────────────────────────────

#[test]
fn render_shows_settings_and_field_labels() {
    let mut app = app_with_runtime();
    settings::open_menu(&mut app);
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);

    assert!(text.contains("Settings"), "modal title missing: {text}");
    for label in [
        "Writer model",
        "Critic model",
        "Critic tone",
        "Speed",
        "Max cost",
        "Prompt",
    ] {
        assert!(
            text.contains(label),
            "field label '{label}' missing: {text}"
        );
    }
    assert!(text.contains("Enter apply"), "footer hint missing: {text}");
}

#[test]
fn render_prompt_row_shows_prompt_mode_note_when_not_prompt_task() {
    let mut app = app_with_runtime();
    app.task = "writing".to_string();
    settings::open_menu(&mut app);
    let buffer = render_app(&app, 120, 40);
    let text = buffer_text(&buffer);
    assert!(
        text.contains("(prompt mode only)"),
        "Prompt row must carry the prompt-mode note when task != prompt: {text}"
    );
}

// ── Input routing ──────────────────────────────────────────────────

#[test]
fn s_opens_the_menu() {
    let mut app = App::new();
    assert!(!app.settings.open);
    let flow = input::handle_key(&mut app, press(KeyCode::Char('s')));
    assert_eq!(flow, ControlFlow::Continue);
    assert!(app.settings.open, "'s' should open the settings menu");
}

#[test]
fn esc_while_open_does_not_stop_app() {
    let mut app = app_with_runtime();
    input::handle_key(&mut app, press(KeyCode::Char('s')));
    assert!(app.settings.open);
    let flow = input::handle_key(&mut app, press(KeyCode::Esc));
    assert_eq!(
        flow,
        ControlFlow::Continue,
        "Esc while the menu is open must NOT stop the app"
    );
    assert!(!app.settings.open, "Esc closes the menu");
}

#[test]
fn q_while_open_does_not_stop_app() {
    let mut app = app_with_runtime();
    input::handle_key(&mut app, press(KeyCode::Char('s')));
    assert!(app.settings.open);
    let flow = input::handle_key(&mut app, press(KeyCode::Char('q')));
    assert_eq!(
        flow,
        ControlFlow::Continue,
        "'q' while the menu is open must NOT stop the app"
    );
    assert!(
        app.settings.open,
        "'q' is captured as text, menu stays open"
    );
}

#[test]
fn ctrl_c_quits_even_with_menu_open() {
    let mut app = app_with_runtime();
    input::handle_key(&mut app, press(KeyCode::Char('s')));
    assert!(app.settings.open);
    let ctrl_c = KeyEvent {
        code: KeyCode::Char('c'),
        modifiers: KeyModifiers::CONTROL,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    };
    let flow = input::handle_key(&mut app, ctrl_c);
    assert_eq!(flow, ControlFlow::Quit, "Ctrl+C is a hard quit, always");
}

// ── C2 — token meter does not split a word at ~80 cols ────────────

#[test]
fn token_meter_no_midword_split_with_large_counts() {
    let mut app = App::new();
    app.writer_tokens = 99999;
    app.critic_tokens = 99999;
    app.apology_tokens = 99999;
    app.total_tokens = 299997;
    app.task = "prompt".to_string();

    // Render in an 80-wide terminal (compact banner path) and confirm the
    // abbreviated labels appear contiguous with their counts (not split).
    let buffer = render_app(&app, 80, 28);
    let text = buffer_text(&buffer);
    assert!(
        text.contains("W 99999"),
        "writer count split or missing: {text}"
    );
    assert!(
        text.contains("C 99999"),
        "critic count split or missing: {text}"
    );
    assert!(
        text.contains("A 99999"),
        "apology count split or missing: {text}"
    );
    // The long words must never appear (they were the source of the wrap bug).
    assert!(
        !text.contains("Apology"),
        "meter must not use the long 'Apology' label: {text}"
    );
}

// ── Enter validation (config = Some) ──────────────────────────────

/// Build a minimal `Config` directly (all fields `pub`) for the validation
/// path. No filesystem/env setup needed — `detect_provider` rejects an unknown
/// model name before any API key is consulted.
fn minimal_config() -> Config {
    Config {
        writer_model: "deepseek-chat".into(),
        critic_model: "deepseek-chat".into(),
        critic_style: CriticStyle::Random,
        speed: Speed::Normal,
        task: InfernoTask::Writing,
        prompt: None,
        input: std::path::PathBuf::new(),
        max_cost_usd: 2.0,
        temperature: 0.8,
        max_tokens: 8192,
        timeout_secs: 120,
        repo_root: std::path::PathBuf::new(),
        openai_base_url: None,
        deepseek_base_url: None,
        moonshot_base_url: None,
    }
}

#[test]
fn enter_with_invalid_model_sets_message_and_stays_open() {
    let mut app = app_with_runtime();
    app.config = Some(Arc::new(minimal_config()));
    settings::open_menu(&mut app);

    // Force an unrecognised writer model (no provider matches → UnknownModel,
    // which needs no env key) so validation fails on Enter.
    app.settings.draft.writer_model = "totally-not-a-real-model".into();

    let flow = settings::handle_menu_key(&mut app, press(KeyCode::Enter));
    assert_eq!(flow, ControlFlow::Continue);
    assert!(
        app.settings.open,
        "menu must stay open when validation fails"
    );
    assert!(
        app.settings.message.is_some(),
        "a validation error message must be shown"
    );
    // Runtime must NOT have been written with the bad model.
    assert_ne!(
        app.runtime.read().unwrap().writer_model,
        "totally-not-a-real-model",
        "invalid draft must not propagate to runtime"
    );
}
