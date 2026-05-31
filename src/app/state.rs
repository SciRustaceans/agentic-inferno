/// Overall application lifecycle state.
///
/// A simple four-state machine that drives TUI rendering and loop control:
///
/// ```text
/// Idle → Running → Stopping → Done
///                 ↑               │
///                 └─── re-run ────┘
/// ```
///
/// The TUI renders differently based on the current state:
/// - **Idle**: splash/loading screen before the loop starts.
/// - **Running**: live three-pane feed with streaming output.
/// - **Stopping**: draining in-flight work before exiting.
/// - **Done**: the run has ended (normal exit, guard trip, or user quit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    /// Initial state before the Writer-Critic loop starts.
    Idle,

    /// The Writer-Critic loop is actively running.
    Running,

    /// User has requested stop (Esc/q or cancel-token); draining in-flight
    /// LLM calls and pending events.
    Stopping,

    /// The run has ended: normal exit, a guard triggered, or user quit.
    Done,
}
