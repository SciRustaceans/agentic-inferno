pub mod input;
pub mod pane;
pub mod ui;

use std::io::{self, BufWriter, Stdout};
use std::time::{Duration, Instant};

use crossterm::{
    cursor,
    event::{Event, EventStream},
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
    stopping_since: Option<Instant>,
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
                stopping_since: None,
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
        let mut app = ui::App::new();
        app.state = AppState::Running;
        self.state = AppState::Running;

        loop {
            tokio::select! {
                event = event_rx.recv() => {
                    match event {
                        Some(event) => handle_app_event(&mut app, event),
                        None => {
                            app.state = AppState::Done;
                            self.state = AppState::Done;
                            break;
                        }
                    }
                }
                    maybe_event = reader.next() => {
                    if let Some(Ok(Event::Key(key))) = maybe_event {
                        match input::handle_key(&mut app, key) {
                            input::ControlFlow::Stop => {
                                self.cancel_token.cancel();
                                app.state = AppState::Stopping;
                                self.state = AppState::Stopping;
                                if self.stopping_since.is_none() {
                                    self.stopping_since = Some(Instant::now());
                                }
                            }
                            input::ControlFlow::Quit => {
                                self.cancel_token.cancel();
                                app.state = AppState::Done;
                                self.state = AppState::Done;
                                break;
                            }
                            input::ControlFlow::Continue => {}
                        }
                    }
                }
                _ = self.cancel_token.cancelled() => {
                    app.state = AppState::Stopping;
                    self.state = AppState::Stopping;
                    if self.stopping_since.is_none() {
                        self.stopping_since = Some(Instant::now());
                    }
                }
            }

            if app.state == AppState::Done {
                break;
            }

            // Stopping timeout guard — prevents the TUI from spinning
            // indefinitely if Shutdown never arrives.
            const STOPPING_TIMEOUT_SECS: u64 = 10;
            if self.state == AppState::Stopping {
                if let Some(since) = self.stopping_since {
                    if since.elapsed() > Duration::from_secs(STOPPING_TIMEOUT_SECS) {
                        app.state = AppState::Done;
                        self.state = AppState::Done;
                        break;
                    }
                }
            }

            self.terminal
                .draw(|frame| ui::render(frame, &app))
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
}

// ── Event handler (free function) ─────────────────────────────────

fn handle_app_event(app: &mut ui::App, event: AppEvent) {
    match event {
        AppEvent::WriterOutput(chunk) => {
            if let Ok(mut buf) = app.writer_buffer.write() {
                buf.push(&chunk);
                buf.scroll_to_bottom();
            }
        }
        AppEvent::CriticOutput(chunk) => {
            if let Ok(mut buf) = app.critic_buffer.write() {
                buf.push(&chunk);
                buf.scroll_to_bottom();
            }
        }
        AppEvent::ApologyReady(text) => {
            app.apology_text = Some(text);
        }
        AppEvent::WriterDone(version) => {
            app.writer_version = version;
        }
        AppEvent::CritiqueReady(version) => {
            app.critic_version = version;
        }
        AppEvent::ApologyTriggered => {}
        AppEvent::Error(err) => {
            app.error = Some(err);
        }
        AppEvent::CostWarning {
            spent,
            limit,
            writer_cost,
            critic_cost,
            apology_cost,
        } => {
            app.cost_spent = spent;
            app.cost_limit = limit;
            app.writer_cost = writer_cost;
            app.critic_cost = critic_cost;
            app.apology_cost = apology_cost;
        }
        AppEvent::LoopExhausted => {
            app.state = AppState::Done;
        }
        AppEvent::ApologyCooldown(remaining) => {
            app.apology_cooldown = remaining;
        }
        AppEvent::Shutdown => {
            app.state = AppState::Done;
        }
    }
}
