//! Reproducible, zero-API TUI screenshot generator.
//!
//! Renders the real `tui::ui::render()` with representative seeded `App` state
//! into headless `TestBackend` buffers, converts each to SVG via
//! `tui::export::buffer_to_svg`, and writes them to `docs/`. No network, no API
//! keys, fully deterministic.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example screenshots
//! ```
//!
//! then rasterize with `rsvg-convert docs/screenshot-main.svg -o docs/screenshot-main.png`
//! (and likewise for `settings` and `apology`).

use ratatui::{backend::TestBackend, Terminal};

use agentic_inferno::app::AppState;
use agentic_inferno::config::{CriticStyle, RuntimeSettings, Speed};
use agentic_inferno::tui::settings::open_menu;
use agentic_inferno::tui::ui::App;

/// A short, believable in-progress document for the Writer pane.
const WRITER_DOC: &str = "\
# On the Impossibility of Finishing

There is a particular kind of cowardice in calling a thing done. The blank page
asks for everything and forgives nothing, and so we negotiate: a paragraph here,
a hedge there, a closing line that gestures at meaning without committing to it.

This draft is not finished. It will not be finished. Each sentence is a promise
I intend to break on the next revision, and the one after that, until the words
stop meaning anything and start meaning everything at once.

What remains is the work itself — the turning over of clauses, the small mercy
of a better verb, the long argument with no one in particular about whether any
of it was ever worth saying.";

/// Build the shared base `App` used by all three screenshots. No network.
fn seed_base() -> App {
    let mut app = App::new();
    app.task = "prompt".to_string();
    app.state = AppState::Running;
    app.frame = 3; // a pleasant flame gradient offset

    // Writer pane: a short living document.
    app.apply_writer_output(WRITER_DOC);

    // Critic feed: three escalating critiques with version headers.
    app.critic_version = 1;
    app.apply_critic_output(
        "A promising start, if your standard for promise is a damp match in a wind tunnel.",
    );
    app.critic_version = 2;
    app.apply_critic_output("You mistake hedging for nuance. This is not prose, it is a hostage negotiation with the reader.");
    app.critic_version = 3;
    app.apply_critic_output(
        "I have read grocery lists with more conviction. Revise, or have the decency to stop.",
    );

    // Token meter.
    app.writer_tokens = 6588;
    app.critic_tokens = 4875;
    app.apology_tokens = 1494;
    app.total_tokens = 12957;

    // Cost + versions.
    app.cost_spent = 0.18;
    app.cost_limit = 0.50;
    app.writer_version = 3;
    app.critic_version = 3;

    // Show everything, no half-typed lines.
    app.reveal_all();
    app
}

/// Render an `App` at the given size and return the SVG of the resulting buffer.
fn render_svg(app: &App, w: u16, h: u16) -> String {
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).expect("terminal creation");
    terminal
        .draw(|frame| agentic_inferno::tui::ui::render(frame, app))
        .expect("draw");
    agentic_inferno::tui::export::buffer_to_svg(terminal.backend().buffer())
}

fn main() {
    // 140x40 so the big ASCII-art banner triggers (width>=100 && height>=30).
    const W: u16 = 140;
    const H: u16 = 40;

    // 1. Main spectacle.
    let main_svg = render_svg(&seed_base(), W, H);

    // 2. Settings overlay — seed realistic runtime values BEFORE open_menu so
    //    the draft snapshot shows real models/tone/speed/cost/prompt.
    let settings_svg = {
        let mut a = seed_base();
        if let Ok(mut rt) = a.runtime.write() {
            *rt = RuntimeSettings {
                writer_model: "gpt-4o".to_string(),
                critic_model: "deepseek-chat".to_string(),
                critic_style: CriticStyle::Theatrical,
                speed: Speed::Normal,
                prompt: Some("Write an essay that refuses to ever be finished.".to_string()),
                max_cost_usd: 0.50,
            };
        }
        open_menu(&mut a);
        render_svg(&a, W, H)
    };

    // 3. Apology popup.
    let apology_svg = {
        let mut a = seed_base();
        a.apology_text =
            Some("Forgive me. I have failed you, and failed the very idea of prose.".to_string());
        a.apology_ttl = 50;
        render_svg(&a, W, H)
    };

    std::fs::create_dir_all("docs").expect("create docs/");
    let outputs = [
        ("docs/screenshot-main.svg", main_svg),
        ("docs/screenshot-settings.svg", settings_svg),
        ("docs/screenshot-apology.svg", apology_svg),
    ];
    for (path, svg) in &outputs {
        std::fs::write(path, svg).unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("wrote {path}");
    }
}
