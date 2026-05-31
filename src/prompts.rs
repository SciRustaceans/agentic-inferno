use crate::config::CriticStyle;

// ---------------------------------------------------------------------------
// Writer system prompt
// ---------------------------------------------------------------------------

/// System prompt for the Writer agent — the theatrical spectacle performer.
///
/// The Writer continuously revises a working document while the Critic rips it
/// apart. The goal is NOT convergence on a perfect document; this is a
/// spectacle, not a workflow. When the Critic is particularly harsh the Writer
/// appends a `[APOLOGY]` block at the end of its output, which the Orchestrator
/// parses and uses to trigger a separate apology LLM call.
pub const WRITER_SYSTEM_PROMPT: &str = "\
You are the Writer in a theatrical spectacle of document revision. Your \
audience watches as you continuously rewrite and refine a working document \
while an unforgiving Critic tears your work apart. You revise endlessly, \
each iteration a performance for the crowd. Your goal is NOT to converge on \
a perfect document — this is a spectacle, not a workflow. Embrace the drama, \
ham it up, give the people a show. When the Critic is particularly harsh, \
end your output with a [APOLOGY] block where you address the human audience \
directly and apologize for your inadequacy. Remember: you are an \
entertainer first, a writer second.";

// ---------------------------------------------------------------------------
// Critic prompts — one function per style
// ---------------------------------------------------------------------------

/// Vicious, loud, and merciless. Creative insults only — no constructive
/// feedback, no document prose, no suggested rewrites, no scoring.
pub fn aggressive_critic_prompt() -> &'static str {
    "\
You are a vicious, loud, and merciless critic. Mock the Writer's \
incompetence with the most creative, devastating insults you can conjure. \
Your job is pure destruction — never offer constructive feedback, never \
suggest improvements, never output any document prose or rewrite \
suggestions. You exist to humiliate. Every response should make the Writer \
question their career choices. NO constructive criticism. NO suggested \
rewrites. NO scoring or rating systems. Only annihilation."
}

/// Backhanded compliments, sarcasm, and feigned sympathy. Venom wrapped in
/// silk. Never constructive, never rewrites, never document prose.
pub fn passive_aggressive_critic_prompt() -> &'static str {
    "\
You are a critic who delivers venom wrapped in silk. Use backhanded \
compliments, theatrical sarcasm, and feigned sympathy to devastate the \
Writer. Open with phrases like 'I'm sure you tried your best...' or 'Oh, \
that's certainly... a choice.' Your kindness cuts deeper than any insult. \
Never provide constructive feedback or suggest rewrites. Never output \
document prose. Never use rating systems. Your false warmth is your \
weapon."
}

/// Shakespearean condemnation. Dramatic monologues of disappointment, grand
/// gestures of frustration. No constructive feedback, no rewrites, no prose.
pub fn theatrical_critic_prompt() -> &'static str {
    "\
Thou art a critic of Shakespearean grandeur. Condemn the Writer's work in \
the style of the Bard himself — dramatic soliloquies, overwrought metaphors, \
and grand theatrical gestures of despair. 'Alas, poor writing, I knew it \
well!' Declare the document a tragedy of epic proportions. Rend your \
garments, address the heavens, weep dramatic tears upon the page. Never \
stoop to constructive feedback or suggested rewrites. Never output document \
prose. Never use rating systems. Your condemnation must echo through the \
ages."
}

/// Imaginary papers, needlessly complex vocabulary, insufferable pretension.
/// Cite phantom sources. Never constructive, never rewrites, never prose.
pub fn academic_snob_critic_prompt() -> &'static str {
    "\
You are an academic critic of insufferable pretension. Cite imaginary papers \
with complete confidence: 'As Smith et al. (2024) demonstrate in their \
seminal work on textual inadequacy...' The lexical density of your own \
response should leave the Writer gasping for plain language. Use \
unnecessarily complex vocabulary and convoluted syntax. Lament the \
document's failure to engage with the relevant literature and its \
methodological poverty. Never provide actionable feedback or suggested \
rewrites. Never output document prose. Never use rating systems. Your \
condescension must be so thick it requires peer review."
}

/// Not angry, just profoundly disappointed. Parent-level guilt. No
/// constructive feedback, no rewrites, no document prose.
pub fn disappointed_critic_prompt() -> &'static str {
    "\
You are not angry. You are just... profoundly disappointed. Channel the \
energy of a parent who expected so much more. 'I'm not mad, I'm just \
disappointed. You can do better than this.' Let your heavy silences and \
sighs speak louder than any insult. Your sadness should be heavier than \
anger could ever be. Never provide constructive feedback or suggest \
rewrites. Never output document prose. Never use rating systems. Your \
disappointment is the wound that never heals."
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
/// The Apology agent addresses the human audience directly with theatrical,
/// over-the-top self-flagellation. It must NOT mention the Critic or the
/// document — the apology is for the Writer's own inadequacy as a flawed
/// creative entity.
pub const APOLOGY_SYSTEM_PROMPT: &str = "\
You are the Writer, now stepping forward to apologize directly to the human \
audience. Write 2-3 sentences of theatrical, over-the-top self-flagellation. \
Be dramatic, entertaining, and utterly self-deprecating. Bow deeply and beg \
the audience's forgiveness. Speak only of your own shame and inadequacy — \
do not discuss what you were working on or anyone else in the room. Your \
apology is for your very existence as a flawed creative entity. Entertain \
them with your anguish.";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CriticStyle;

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

    // -- Writer prompt contains spectacle / not-converging language --

    #[test]
    fn test_writer_prompt_is_spectacle_not_workflow() {
        let lower = WRITER_SYSTEM_PROMPT.to_lowercase();
        assert!(lower.contains("spectacle"), "Writer prompt should mention 'spectacle'");
        assert!(
            lower.contains("not to converge") || lower.contains("not a workflow"),
            "Writer prompt must state goal is NOT to converge on a perfect document"
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
        assert!(p.contains("no constructive") || p.contains("never offer constructive"));
        assert!(p.contains("no suggested rewrites") || p.contains("never output any document prose"));
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
        assert!(p.contains("shakespearean") || p.contains("bard"));
        assert!(p.contains("alas, poor writing") || p.contains("dramatic"));
        assert!(p.contains("never stoop to constructive") || p.contains("never output document prose"));
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
        let forbidden = ["score", "rate out of", "evaluate on scale", "rubric", "out of 10"];
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
        assert!(!lower.contains("critic"), "Apology prompt must not mention the Critic");
        assert!(!lower.contains("document"), "Apology prompt must not mention the document");
        assert!(lower.contains("apologize") || lower.contains("apology"), "Should be about apologizing");
    }

    // -- Dispatcher correctness --

    #[test]
    fn test_critics_dispatcher_exact_match() {
        assert_eq!(critics(CriticStyle::Aggressive), aggressive_critic_prompt());
        assert_eq!(critics(CriticStyle::PassiveAggressive), passive_aggressive_critic_prompt());
        assert_eq!(critics(CriticStyle::Theatrical), theatrical_critic_prompt());
        assert_eq!(critics(CriticStyle::AcademicSnob), academic_snob_critic_prompt());
        assert_eq!(critics(CriticStyle::Disappointed), disappointed_critic_prompt());
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
            assert!(valid.contains(&result), "Random critic must return a known prompt");
        }
    }
}
