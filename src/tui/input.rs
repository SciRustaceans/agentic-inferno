use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::tui::ui::{App, FocusTarget};

/// Signal returned by [`handle_key`] to tell the TUI event loop what to do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlFlow {
    /// Continue the event loop — no action needed.
    Continue,
    /// Graceful stop: cancel the cancel token and drain in-flight work.
    Stop,
    /// Immediate quit: exit the process without draining.
    Quit,
}

/// Process a keyboard event and update the application state.
///
/// Only `KeyEventKind::Press` events are processed — repeat and release
/// events are silently ignored.
///
/// # Key bindings
///
/// | Key | Action |
/// |-----|--------|
/// | `Esc` / `q` | `ControlFlow::Stop` — graceful shutdown |
/// | `Ctrl+C` | `ControlFlow::Quit` — immediate exit |
/// | `Tab` | Cycle focus between Writer and Critic panes |
/// | `Up` | Scroll focused pane up by 1 line |
/// | `Down` | Scroll focused pane down by 1 line |
/// | `PageUp` | Scroll focused pane up by 10 lines |
/// | `PageDown` | Scroll focused pane down by 10 lines |
/// | `Home` | Scroll focused pane to top |
/// | `End` | Scroll focused pane to bottom |
pub fn handle_key(app: &mut App, key: KeyEvent) -> ControlFlow {
    if key.kind != KeyEventKind::Press {
        return ControlFlow::Continue;
    }

    // Ctrl+C is a hard quit that always wins, even with the menu open.
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        return ControlFlow::Quit;
    }

    // While the settings menu is open it captures every other key (Esc closes
    // the menu rather than stopping the app; typing/q must not stop it).
    if app.settings.open {
        return crate::tui::settings::handle_menu_key(app, key);
    }

    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => ControlFlow::Stop,

        KeyCode::Char('s') => {
            crate::tui::settings::open_menu(app);
            ControlFlow::Continue
        }

        KeyCode::Tab => {
            app.focused_pane = match app.focused_pane {
                FocusTarget::Writer => FocusTarget::Critic,
                FocusTarget::Critic => FocusTarget::Writer,
            };
            ControlFlow::Continue
        }

        KeyCode::Up => {
            with_focused_buffer(app, |pane| pane.scroll_up(1));
            ControlFlow::Continue
        }
        KeyCode::Down => {
            with_focused_buffer(app, |pane| pane.scroll_down(1));
            ControlFlow::Continue
        }
        KeyCode::PageUp => {
            with_focused_buffer(app, |pane| pane.scroll_up(10));
            ControlFlow::Continue
        }
        KeyCode::PageDown => {
            with_focused_buffer(app, |pane| pane.scroll_down(10));
            ControlFlow::Continue
        }
        KeyCode::Home => {
            with_focused_buffer(app, |pane| pane.scroll_to_top());
            ControlFlow::Continue
        }
        KeyCode::End => {
            with_focused_buffer(app, |pane| pane.scroll_to_bottom());
            ControlFlow::Continue
        }

        _ => ControlFlow::Continue,
    }
}

/// Lock the buffer for the currently focused pane and apply `f`.
fn with_focused_buffer(app: &App, f: impl FnOnce(&mut crate::tui::pane::PaneBuffer)) {
    let lock_result = match app.focused_pane {
        FocusTarget::Writer => app.writer_buffer.write(),
        FocusTarget::Critic => app.critic_buffer.write(),
    };
    if let Ok(mut guard) = lock_result {
        f(&mut guard);
    }
}
