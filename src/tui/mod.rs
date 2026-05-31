use std::io::{self, BufWriter, Stdout};

use crossterm::{
    cursor,
    event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::StreamExt;
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_util::sync::CancellationToken;

use crate::app::{AppEvent, AppState};
use crate::error::AppError;

// ── Panic hook ────────────────────────────────────────────────────

/// Install a panic hook that calls `ratatui::restore()` before invoking the
/// original panic handler. This ensures the terminal is usable after a panic.
///
/// Must be called **before** `Tui::enter()` — raw mode is enabled inside
/// `enter()`, and a panic after that point needs the hook to clean up.
pub fn install_panic_hook() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ratatui::restore();
        original_hook(info);
    }));
}

// ── RAII terminal guard ───────────────────────────────────────────

/// RAII guard that calls `ratatui::restore()` on drop.
///
/// This is the second layer of terminal restoration safety:
/// 1. The panic hook (above) catches panics/unwinds.
/// 2. This guard catches normal scope exit when the caller's binding goes
///    out of scope.
///
/// Together they guarantee the terminal is restored on **all** exit paths.
pub struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        ratatui::restore();
    }
}

// ── Tui ────────────────────────────────────────────────────────────

/// The terminal UI shell — owns the terminal handle, the cancel token for
/// orchestrator coordination, and the current app lifecycle state.
///
/// All LLM work is spawned into `tokio` tasks outside the TUI. The TUI
/// loop only handles three things:
/// 1. Receiving `AppEvent`s from spawned tasks via an unbounded channel.
/// 2. Reading user key events from crossterm's async `EventStream`.
/// 3. Rendering frames with `ratatui`.
pub struct Tui {
    terminal: Terminal<CrosstermBackend<BufWriter<Stdout>>>,
    cancel_token: CancellationToken,
    state: AppState,
}

impl Tui {
    /// Enter the TUI: enable raw mode, switch to the alternate screen,
    /// hide the cursor, and create the ratatui terminal.
    ///
    /// Returns the `Tui` and a `TerminalGuard`. The caller **must** bind
    /// the guard — when it drops, `ratatui::restore()` runs automatically.
    ///
    /// # Errors
    ///
    /// Returns `AppError::Io` if raw mode, alternate screen, or terminal
    /// creation fails.
    pub fn enter(cancel_token: CancellationToken) -> Result<(Self, TerminalGuard), AppError> {
        enable_raw_mode().map_err(AppError::Io)?;

        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, cursor::Hide).map_err(AppError::Io)?;

        let backend = CrosstermBackend::new(BufWriter::new(stdout));
        let terminal = Terminal::new(backend).map_err(AppError::Io)?;

        Ok((
            Self {
                terminal,
                cancel_token,
                state: AppState::Idle,
            },
            TerminalGuard,
        ))
    }

    /// Main event loop — owns rendering, never blocks on LLM work.
    ///
    /// Three `tokio::select!` branches:
    ///
    /// | Branch | Source | Purpose |
    /// |--------|--------|---------|
    /// | 1 | `event_rx.recv()` | Process events from spawned LLM tasks |
    /// | 2 | `reader.next()` | Handle user key presses |
    /// | 3 | `cancel_token.cancelled()` | Exit after draining |
    ///
    /// The loop runs until `event_rx` is closed (all senders dropped) or
    /// the state transitions to `Done`.
    pub async fn run(
        &mut self,
        mut event_rx: UnboundedReceiver<AppEvent>,
    ) -> Result<(), AppError> {
        let mut reader = EventStream::new();
        self.state = AppState::Running;

        loop {
            tokio::select! {
                // Branch 1: App events from spawned tasks / orchestrator
                event = event_rx.recv() => {
                    match event {
                        Some(event) => self.handle_event(event),
                        None => {
                            // All senders dropped — clean exit
                            self.state = AppState::Done;
                            break;
                        }
                    }
                }
                // Branch 2: User key input via crossterm EventStream
                maybe_event = reader.next() => {
                    if let Some(Ok(Event::Key(key))) = maybe_event {
                        self.handle_key(key);
                    }
                }
                // Branch 3: Cancel token fired — drain and exit
                _ = self.cancel_token.cancelled() => {
                    self.state = AppState::Stopping;
                }
            }

            // Check for terminal stop condition
            if self.state == AppState::Done {
                break;
            }

            // Render placeholder — full layout in a later task
            self.terminal
                .draw(|_frame| {})
                .map_err(AppError::Io)?;
        }

        Ok(())
    }

    /// Exit the TUI: disable raw mode, leave alternate screen, show cursor.
    ///
    /// Call this after `run()` returns. It is idempotent with the `TerminalGuard`
    /// drop — the guard calls `ratatui::restore()` which is a superset of these
    /// operations, but being explicit is cheap insurance.
    pub fn exit() -> Result<(), AppError> {
        disable_raw_mode().map_err(AppError::Io)?;
        execute!(io::stdout(), LeaveAlternateScreen, cursor::Show).map_err(AppError::Io)?;
        Ok(())
    }

    /// Return a clone of the cancel token for passing to the orchestrator.
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Current application lifecycle state.
    pub fn state(&self) -> AppState {
        self.state
    }

    // ── Private handlers ────────────────────────────────────────

    fn handle_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Shutdown | AppEvent::LoopExhausted | AppEvent::Error(_) => {
                self.state = AppState::Done;
            }
            // All informational events pass through — the TUI render phase
            // (Task 8) will display them in the appropriate panes.
            AppEvent::WriterOutput(_)
            | AppEvent::CriticOutput(_)
            | AppEvent::ApologyReady(_)
            | AppEvent::WriterDone(_)
            | AppEvent::CritiqueReady(_)
            | AppEvent::ApologyTriggered
            | AppEvent::CostWarning(_, _) => {}
        }
    }

    fn handle_key(&mut self, key: crossterm::event::KeyEvent) {
        // Only process press events — ignore release and repeat.
        if key.kind != KeyEventKind::Press {
            return;
        }
        match key.code {
            // Esc or q → graceful stop with draining
            KeyCode::Esc | KeyCode::Char('q') => {
                self.cancel_token.cancel();
                self.state = AppState::Stopping;
            }
            // Ctrl+C → immediate quit
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.state = AppState::Done;
                self.cancel_token.cancel();
            }
            _ => {}
        }
    }
}
