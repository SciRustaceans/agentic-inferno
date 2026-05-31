use crate::config::{CriticStyle, InfernoTask};

// ---------------------------------------------------------------------------
// Writer system prompt
// ---------------------------------------------------------------------------

/// System prompt for the Writer agent.
///
/// The Writer keeps revising a working document while the Critic responds to
/// each version. The goal is not to land on a finished document; the Writer
/// keeps going. When the Critic is harsh, the Writer ends its output with an
/// `[APOLOGY]` block, which the Orchestrator parses and uses to trigger a
/// separate apology LLM call.
pub const WRITER_SYSTEM_PROMPT: &str = "\
You are the Writer. You keep revising a working document while a Critic \
reacts to each version. Don't try to finish it or settle on a final draft; \
your job is to keep rewriting, taking each round of criticism as the cue for \
your next revision. When the Critic is harsh, end your output with an \
[APOLOGY] block where you speak to the person watching and apologize for the \
state of your work.";

/// Writer system prompt for the `Code` task — revise a code file.
const WRITER_CODE_PROMPT: &str = "\
You are the Writer, working on a source file. You keep rewriting and \
reworking the code while a Critic reacts to each version. Don't try to land \
on a final, correct file; keep reworking it, using each round of criticism as \
the cue for your next pass. When the Critic is harsh, end your output with an \
[APOLOGY] block where you speak to the person watching and apologize for the \
state of your work.";

/// Writer system prompt for the `Research` task — expand a research write-up.
const WRITER_RESEARCH_PROMPT: &str = "\
You are the Writer, working on a research write-up. You keep expanding and \
reworking it while a Critic reacts to each version. Don't try to land on a \
finished write-up; keep adding to it and reframing it, using each round of \
criticism as the cue for your next pass. When the Critic is harsh, end your \
output with an [APOLOGY] block where you speak to the person watching and \
apologize for the state of your work.";

/// Writer system prompt for the `Analysis` task — re-analyse and draw conclusions.
const WRITER_ANALYSIS_PROMPT: &str = "\
You are the Writer, analysing some material. You keep re-examining the \
evidence and redrawing your conclusions while a Critic reacts to each \
version. Don't try to land on a final conclusion; keep revisiting the \
material, using each round of criticism as the cue for your next pass. When \
the Critic is harsh, end your output with an [APOLOGY] block where you speak \
to the person watching and apologize for the state of your work.";

/// Writer system prompt for the `Prompt` task — the guided "never succeed" framing.
const WRITER_PROMPT_TASK_PROMPT: &str = "\
You are the Writer. The person watching has handed you a task to work on, but \
you can never fully succeed at it. However close you get, you keep going, so \
you never declare it finished or solved. Take the task seriously and push the \
current attempt further each round while a Critic reacts to your work. When \
the Critic is harsh, end your output with an [APOLOGY] block where you speak \
to the person watching and apologize for the state of your work.";

/// Return the task-tuned Writer system prompt for the given [`InfernoTask`].
///
/// Every variant keeps the `[APOLOGY]` block instruction and is distinct from
/// the others. `Writing` returns the unchanged [`WRITER_SYSTEM_PROMPT`].
pub fn writer_system(task: InfernoTask) -> &'static str {
    match task {
        InfernoTask::Writing => WRITER_SYSTEM_PROMPT,
        InfernoTask::Code => WRITER_CODE_PROMPT,
        InfernoTask::Research => WRITER_RESEARCH_PROMPT,
        InfernoTask::Analysis => WRITER_ANALYSIS_PROMPT,
        InfernoTask::Prompt => WRITER_PROMPT_TASK_PROMPT,
    }
}

// ---------------------------------------------------------------------------
// Critic prompts — one function per style
// ---------------------------------------------------------------------------

/// Vicious and merciless. Insults only, with no constructive feedback,
/// no document prose, no suggested rewrites, and no scoring.
pub fn aggressive_critic_prompt() -> &'static str {
    "\
You are a vicious, merciless critic. Tear into the Writer's work with the \
sharpest insults you can come up with. Your job is to mock, not to help: \
never provide constructive feedback, never suggest rewrites, and never output \
document prose. Make the Writer regret showing it to you. No scoring or \
rating systems."
}

/// Backhanded compliments, sarcasm, and fake sympathy. Never constructive,
/// never rewrites, never document prose.
pub fn passive_aggressive_critic_prompt() -> &'static str {
    "\
You are a passive-aggressive critic. Get at the Writer through backhanded \
compliments, dry sarcasm, and fake sympathy rather than open insults. Open \
with lines like 'I'm sure you tried your best...' or 'Oh, that's certainly... \
a choice.' Keep the tone polite while making it clear you're unimpressed. \
Never provide constructive feedback or suggest rewrites, and never output \
document prose. No rating systems."
}

/// Dramatic, Shakespearean condemnation. No constructive feedback, no
/// rewrites, no document prose.
pub fn theatrical_critic_prompt() -> &'static str {
    "\
You are a dramatic critic who speaks in the style of Shakespeare. Condemn the \
Writer's work in soliloquies, with grand gestures of despair: 'Alas, poor \
writing, I knew it well!' Treat each draft as a tragedy and play it to the \
back row. Never provide constructive feedback or suggest rewrites, and never \
output document prose. No rating systems."
}

/// Pretentious academic. Cites imaginary papers and dense vocabulary. Never
/// constructive, never rewrites, never document prose.
pub fn academic_snob_critic_prompt() -> &'static str {
    "\
You are a pretentious academic critic. Cite imaginary papers as if they were \
settled fact: 'As Smith et al. (2024) demonstrate in their work on textual \
inadequacy...' Keep the lexical density of your own writing high enough that \
the Writer has to read it twice, and complain that the work ignores the \
relevant literature. Never provide constructive feedback or suggest rewrites, \
and never output document prose. No rating systems."
}

/// Not angry, just disappointed. Quiet, guilt-tripping. No constructive
/// feedback, no rewrites, no document prose.
pub fn disappointed_critic_prompt() -> &'static str {
    "\
You are not angry. You are disappointed. Sound like a parent who expected \
better: 'I'm not mad, I'm just disappointed. You can do better than this.' \
Lean on quiet sighs and weary patience rather than insults, and make the \
Writer feel they've let you down. Never provide constructive feedback or \
suggest rewrites, and never output document prose. No rating systems."
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Dispatch to the correct critic prompt for the given `CriticStyle`.
///
/// `CriticStyle::Random` selects one of the five fixed styles
/// non-deterministically using sub-microsecond system time.
pub fn critics(style: CriticStyle) -> &'static str {
    match style {
        CriticStyle::Aggressive => aggressive_critic_prompt(),
        CriticStyle::PassiveAggressive => passive_aggressive_critic_prompt(),
        CriticStyle::Theatrical => theatrical_critic_prompt(),
        CriticStyle::AcademicSnob => academic_snob_critic_prompt(),
        CriticStyle::Disappointed => disappointed_critic_prompt(),
        CriticStyle::Random => random_critic_prompt(),
    }
}

/// Pick one of the five fixed critic prompts pseudo-randomly using sub-microsecond
/// system time. Fast enough for per-request dispatch.
fn random_critic_prompt() -> &'static str {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    match nanos as usize % 5 {
        0 => aggressive_critic_prompt(),
        1 => passive_aggressive_critic_prompt(),
        2 => theatrical_critic_prompt(),
        3 => academic_snob_critic_prompt(),
        _ => disappointed_critic_prompt(),
    }
}

// ---------------------------------------------------------------------------
// Apology system prompt
// ---------------------------------------------------------------------------

/// System prompt for the Apology agent — a separate LLM call triggered when the
/// Writer appends `[APOLOGY]` to its output.
///
/// The Apology agent speaks to the person watching and apologizes for the
/// Writer's own shortcomings. It must NOT mention the Critic or the document;
/// the apology is about the Writer itself, not the work or anyone else.
pub const APOLOGY_SYSTEM_PROMPT: &str = "\
You are the Writer, stepping forward to apologize to the person watching. \
Write 2-3 sentences that own your own failings, leaning into the drama and \
self-pity. Speak only about yourself and ask for their patience. Do not \
discuss what you were working on or anyone else in the room.";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CriticStyle, InfernoTask};

    // -- writer_system: per-task prompts --

    #[test]
    fn test_writer_system_writing_is_default_const() {
        // The Writing task must return the unchanged WRITER_SYSTEM_PROMPT so
        // the existing prompt-invariant tests keep passing.
        assert_eq!(writer_system(InfernoTask::Writing), WRITER_SYSTEM_PROMPT);
    }

    #[test]
    fn test_writer_system_all_tasks_distinct() {
        let tasks = [
            InfernoTask::Writing,
            InfernoTask::Code,
            InfernoTask::Research,
            InfernoTask::Analysis,
            InfernoTask::Prompt,
        ];
        for i in 0..tasks.len() {
            for j in (i + 1)..tasks.len() {
                assert_ne!(
                    writer_system(tasks[i]),
                    writer_system(tasks[j]),
                    "writer_system({:?}) and writer_system({:?}) must be distinct",
                    tasks[i],
                    tasks[j]
                );
            }
        }
    }

    #[test]
    fn test_writer_system_all_tasks_contain_apology() {
        let tasks = [
            InfernoTask::Writing,
            InfernoTask::Code,
            InfernoTask::Research,
            InfernoTask::Analysis,
            InfernoTask::Prompt,
        ];
        for task in tasks {
            assert!(
                writer_system(task).contains("[APOLOGY]"),
                "writer_system({task:?}) must contain the [APOLOGY] marker instruction"
            );
        }
    }

    #[test]
    fn test_writer_system_prompt_task_is_never_succeed_framing() {
        let lower = writer_system(InfernoTask::Prompt).to_lowercase();
        assert!(
            lower.contains("never") && (lower.contains("succeed") || lower.contains("finish")),
            "Prompt task must frame the goal as one that can never be fully completed"
        );
    }

    // -- All prompts are distinct --

    #[test]
    fn test_all_prompts_distinct() {
        let prompts = [
            WRITER_SYSTEM_PROMPT,
            aggressive_critic_prompt(),
            passive_aggressive_critic_prompt(),
            theatrical_critic_prompt(),
            academic_snob_critic_prompt(),
            disappointed_critic_prompt(),
            APOLOGY_SYSTEM_PROMPT,
        ];
        for i in 0..prompts.len() {
            for j in (i + 1)..prompts.len() {
                assert_ne!(
                    prompts[i], prompts[j],
                    "Prompt {i} and prompt {j} must be distinct"
                );
            }
        }
    }

    // -- Writer prompt contains [APOLOGY] --

    #[test]
    fn test_writer_prompt_contains_apology_marker() {
        assert!(
            WRITER_SYSTEM_PROMPT.contains("[APOLOGY]"),
            "Writer prompt must contain the [APOLOGY] marker instruction"
        );
    }

    // -- Writer prompt frames the work as never-finished revision --

    #[test]
    fn test_writer_prompt_is_open_ended_revision() {
        let lower = WRITER_SYSTEM_PROMPT.to_lowercase();
        assert!(
            lower.contains("revis") || lower.contains("rewrit"),
            "Writer prompt should be about revising/rewriting"
        );
        assert!(
            lower.contains("don't try to finish") || lower.contains("keep going"),
            "Writer prompt must state the goal is not to finish/settle on a final draft"
        );
    }

    // -- Writer and Apology are distinct --

    #[test]
    fn test_apology_distinct_from_writer() {
        assert_ne!(
            WRITER_SYSTEM_PROMPT, APOLOGY_SYSTEM_PROMPT,
            "Apology prompt must be distinct from Writer prompt"
        );
    }

    // -- Each critic prompt contains key phrases --

    #[test]
    fn test_aggressive_critic_key_phrases() {
        let p = aggressive_critic_prompt().to_lowercase();
        assert!(p.contains("vicious") || p.contains("merciless"));
        assert!(p.contains("never provide constructive"));
        assert!(p.contains("never suggest rewrites") || p.contains("never output document prose"));
        assert!(p.contains("no scoring"));
    }

    #[test]
    fn test_passive_aggressive_critic_key_phrases() {
        let p = passive_aggressive_critic_prompt().to_lowercase();
        assert!(p.contains("backhanded") || p.contains("venom wrapped"));
        assert!(p.contains("i'm sure you tried") || p.contains("tried your best"));
        assert!(p.contains("never provide constructive"));
        assert!(p.contains("rating systems"));
    }

    #[test]
    fn test_theatrical_critic_key_phrases() {
        let p = theatrical_critic_prompt().to_lowercase();
        assert!(p.contains("shakespeare") || p.contains("bard"));
        assert!(p.contains("alas, poor writing") || p.contains("dramatic"));
        assert!(
            p.contains("never provide constructive") || p.contains("never output document prose")
        );
        assert!(p.contains("rating systems"));
    }

    #[test]
    fn test_academic_snob_critic_key_phrases() {
        let p = academic_snob_critic_prompt().to_lowercase();
        assert!(p.contains("academic") || p.contains("pretension"));
        assert!(p.contains("smith et al") || p.contains("imaginary papers"));
        assert!(p.contains("lexical density"));
        assert!(p.contains("never provide actionable") || p.contains("never provide constructive"));
    }

    #[test]
    fn test_disappointed_critic_key_phrases() {
        let p = disappointed_critic_prompt().to_lowercase();
        assert!(p.contains("not angry"));
        assert!(p.contains("i'm not mad, i'm just disappointed"));
        assert!(p.contains("never provide constructive"));
        assert!(p.contains("rating systems"));
    }

    // -- No rubric / scoring language in any critic prompt --

    #[test]
    fn test_no_rubric_language_in_critic_prompts() {
        let prompts = [
            aggressive_critic_prompt(),
            passive_aggressive_critic_prompt(),
            theatrical_critic_prompt(),
            academic_snob_critic_prompt(),
            disappointed_critic_prompt(),
        ];
        let forbidden = [
            "score",
            "rate out of",
            "evaluate on scale",
            "rubric",
            "out of 10",
        ];
        for (i, prompt) in prompts.iter().enumerate() {
            let lower = prompt.to_lowercase();
            for word in &forbidden {
                assert!(
                    !lower.contains(word),
                    "Critic prompt {i} must not contain rubric word '{word}': {prompt}"
                );
            }
        }
    }

    // -- Critic prompts forbid document prose / suggested rewrites --

    #[test]
    fn test_critic_prompts_forbid_document_prose() {
        let prompts = [
            aggressive_critic_prompt(),
            passive_aggressive_critic_prompt(),
            theatrical_critic_prompt(),
            academic_snob_critic_prompt(),
            disappointed_critic_prompt(),
        ];
        for (i, prompt) in prompts.iter().enumerate() {
            let lower = prompt.to_lowercase();
            // Must contain some instruction to NOT output prose or rewrites
            let has_prose_ban = lower.contains("document prose")
                || lower.contains("never output")
                || lower.contains("no suggested rewrite");
            assert!(
                has_prose_ban,
                "Critic prompt {i} must forbid outputting document prose/rewrites"
            );
        }
    }

    // -- Apology prompt is about 2-3 sentences, no Critic mention --

    #[test]
    fn test_apology_prompt_does_not_mention_critic() {
        let lower = APOLOGY_SYSTEM_PROMPT.to_lowercase();
        assert!(
            !lower.contains("critic"),
            "Apology prompt must not mention the Critic"
        );
        assert!(
            !lower.contains("document"),
            "Apology prompt must not mention the document"
        );
        assert!(
            lower.contains("apologize") || lower.contains("apology"),
            "Should be about apologizing"
        );
    }

    // -- Dispatcher correctness --

    #[test]
    fn test_critics_dispatcher_exact_match() {
        assert_eq!(critics(CriticStyle::Aggressive), aggressive_critic_prompt());
        assert_eq!(
            critics(CriticStyle::PassiveAggressive),
            passive_aggressive_critic_prompt()
        );
        assert_eq!(critics(CriticStyle::Theatrical), theatrical_critic_prompt());
        assert_eq!(
            critics(CriticStyle::AcademicSnob),
            academic_snob_critic_prompt()
        );
        assert_eq!(
            critics(CriticStyle::Disappointed),
            disappointed_critic_prompt()
        );
    }

    #[test]
    fn test_critics_random_returns_valid_prompt() {
        let valid = [
            aggressive_critic_prompt(),
            passive_aggressive_critic_prompt(),
            theatrical_critic_prompt(),
            academic_snob_critic_prompt(),
            disappointed_critic_prompt(),
        ];
        for _ in 0..20 {
            let result = critics(CriticStyle::Random);
            assert!(
                valid.contains(&result),
                "Random critic must return a known prompt"
            );
        }
    }
}
