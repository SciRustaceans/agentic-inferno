use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Maximum number of historical versions kept in the buffer.
const MAX_HISTORY: usize = 50;

/// A single version of the document with metadata.
///
/// Every revision of the document carries a monotonically increasing version
/// number, the content at that revision, and a UTC timestamp of when the
/// version was created.
#[derive(Debug, Clone)]
pub struct DocumentVersion {
    pub version: u64,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

/// Buffer holding the current document version and a ring of past versions.
///
/// History is bounded at [`MAX_HISTORY`] entries. When the buffer overflows,
/// the oldest version is dropped.
#[derive(Debug)]
pub struct DocumentBuffer {
    current: DocumentVersion,
    history: VecDeque<DocumentVersion>,
}

impl DocumentBuffer {
    /// Create a new buffer with the given `initial_content` at version 0.
    pub fn new(initial_content: String) -> Self {
        Self {
            current: DocumentVersion {
                version: 0,
                content: initial_content,
                timestamp: Utc::now(),
            },
            history: VecDeque::with_capacity(MAX_HISTORY + 1),
        }
    }

    /// Replace the current document content with `text`.
    ///
    /// The previous current version is pushed to the history ring. The version
    /// counter is incremented. Returns the new version number.
    pub fn update(&mut self, text: String) -> u64 {
        let next_version = self.current.version + 1;
        let old = std::mem::replace(
            &mut self.current,
            DocumentVersion {
                version: next_version,
                content: text,
                timestamp: Utc::now(),
            },
        );
        self.history.push_back(old);
        if self.history.len() > MAX_HISTORY {
            self.history.pop_front();
        }
        next_version
    }

    /// Return the current version number.
    pub fn current_version(&self) -> u64 {
        self.current.version
    }

    /// Return a clone of the current document content.
    pub fn current_content(&self) -> String {
        self.current.content.clone()
    }
}

/// Tracks the cooldown state for apology LLM calls.
///
/// The cooldown rule is: minimum 30 seconds **or** 3 critique cycles since the
/// last apology, whichever is longer. Both conditions must be satisfied before
/// a new apology can fire.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApologyCooldown {
    /// Wall clock of the most recent apology LLM call.
    pub last_apology_time: Option<Instant>,
    /// Number of successful critic loop cycles since the last apology.
    pub cycles_since_apology: u32,
}

/// Shared state for the document, reference-counted and protected by a
/// read-write lock.
///
/// # Concurrency model
///
/// - **Writer** task acquires a write lock to update the document.
/// - **Critic** task acquires a read lock to inspect the latest version.
/// - **TUI render** acquires a read lock to display the current content.
///
/// Because [`std::sync::RwLock`] permits any number of concurrent readers, the
/// Critic and TUI can both read simultaneously without blocking each other.
/// A writer only blocks when it cannot acquire the exclusive write lock.
///
/// # Critique buffer
///
/// The `latest_critique` field holds the most recent critique from the Critic
/// loop: `(document_version_when_criticised, critique_text)`.  The Writer loop
/// reads this before each revision cycle to incorporate feedback.  A separate
/// `RwLock` avoids contention with document readers.
///
/// # Apology cooldown
///
/// The `apology_cooldown` field tracks the time and cycle count since the last
/// apology.  Both the Writer and Critic loops may read/write this state:
///
/// - Writer loop checks cooldown when detecting `[APOLOGY]` and resets it on
///   a successful apology.
/// - Critic loop increments `cycles_since_apology` after each successful cycle.
#[derive(Debug, Clone)]
pub struct SharedState {
    pub document: Arc<RwLock<DocumentBuffer>>,
    /// Latest critique: `(document_version, critique_text)`.
    pub latest_critique: Arc<RwLock<Option<(u64, String)>>>,
    /// Apology cooldown state shared between Writer and Critic loops.
    pub apology_cooldown: Arc<RwLock<ApologyCooldown>>,
}

impl SharedState {
    /// Create a new shared state with the given `initial_content` at version 0.
    pub fn new(initial_content: String) -> Self {
        Self {
            document: Arc::new(RwLock::new(DocumentBuffer::new(initial_content))),
            latest_critique: Arc::new(RwLock::new(None)),
            apology_cooldown: Arc::new(RwLock::new(ApologyCooldown::default())),
        }
    }

    /// Write a critique into the shared buffer.
    ///
    /// `version` is the document version the critique was based on.
    pub fn write_critique(&self, version: u64, text: String) {
        let mut guard = self
            .latest_critique
            .write()
            .expect("SharedState::write_critique: latest_critique lock poisoned");
        *guard = Some((version, text));
    }

    /// Read the latest critique, if any.
    ///
    /// Returns `(document_version, critique_text)` or `None` if no critique
    /// has been written yet.
    pub fn read_critique(&self) -> Option<(u64, String)> {
        self.latest_critique
            .read()
            .expect("SharedState::read_critique: latest_critique lock poisoned")
            .clone()
    }

    /// Atomically read the current version and content under a single read
    /// lock guard.
    ///
    /// Because both values are read from the *same* lock scope, callers are
    /// guaranteed that the returned `(version, content)` pair is consistent:
    /// the content corresponds exactly to that version.
    pub fn snapshot(&self) -> (u64, String) {
        let guard = self
            .document
            .read()
            .expect("SharedState::snapshot: document lock poisoned");
        (guard.current_version(), guard.current_content())
    }

    /// Return the current document version number under a read lock.
    pub fn current_version(&self) -> u64 {
        self.document
            .read()
            .expect("SharedState::current_version: document lock poisoned")
            .current_version()
    }

    /// Update the document content under an exclusive write lock.
    ///
    /// Returns the new version number.
    pub fn update(&self, text: String) -> u64 {
        let mut guard = self
            .document
            .write()
            .expect("SharedState::update: document lock poisoned");
        guard.update(text)
    }

    // -----------------------------------------------------------------------
    // Apology cooldown helpers
    // -----------------------------------------------------------------------

    /// Read the current apology cooldown state.
    pub fn read_apology_cooldown(&self) -> ApologyCooldown {
        *self
            .apology_cooldown
            .read()
            .expect("SharedState::read_apology_cooldown: lock poisoned")
    }

    /// Mark an apology as having just fired — records the current instant and
    /// resets the critique cycle counter.
    pub fn mark_apology_fired(&self) {
        let mut guard = self
            .apology_cooldown
            .write()
            .expect("SharedState::mark_apology_fired: lock poisoned");
        guard.last_apology_time = Some(Instant::now());
        guard.cycles_since_apology = 0;
    }

    /// Increment the critique cycle counter (called by the Critic loop after
    /// each successful LLM cycle).
    pub fn increment_critique_cycles(&self) {
        let mut guard = self
            .apology_cooldown
            .write()
            .expect("SharedState::increment_critique_cycles: lock poisoned");
        guard.cycles_since_apology = guard.cycles_since_apology.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_version_is_zero() {
        let state = SharedState::new("hello".to_string());
        let (ver, content) = state.snapshot();
        assert_eq!(ver, 0);
        assert_eq!(content, "hello");
    }

    #[test]
    fn test_version_increments_on_update() {
        let state = SharedState::new("v0".to_string());
        assert_eq!(state.update("v1".to_string()), 1);
        assert_eq!(state.update("v2".to_string()), 2);

        let (ver, content) = state.snapshot();
        assert_eq!(ver, 2);
        assert_eq!(content, "v2");
    }

    #[test]
    fn test_snapshot_is_consistent() {
        let state = SharedState::new("start".to_string());
        state.update("second".to_string());
        state.update("third".to_string());

        let (ver, content) = state.snapshot();
        assert_eq!(ver, 2);
        assert_eq!(content, "third");
    }

    #[test]
    fn test_history_is_capped_at_fifty() {
        let state = SharedState::new("init".to_string());
        for i in 1..=55 {
            state.update(format!("update {i}"));
        }

        let (ver, content) = state.snapshot();
        assert_eq!(ver, 55);
        assert_eq!(content, "update 55");

        let guard = state.document.read().expect("lock poisoned");
        assert_eq!(guard.history.len(), MAX_HISTORY);

        let oldest = guard.history.front().expect("history should not be empty");
        assert_eq!(oldest.version, 5);
        assert_eq!(oldest.content, "update 5");
    }

    #[test]
    fn test_concurrent_reads_do_not_block() {
        let state = SharedState::new("concurrent".to_string());
        let reader = state.clone();
        let writer = state.clone();

        let reader_handle = std::thread::spawn(move || reader.snapshot());
        let writer_handle =
            std::thread::spawn(move || writer.update("written concurrently".to_string()));

        let (ver, _content) = reader_handle.join().expect("reader panicked");
        let new_ver = writer_handle.join().expect("writer panicked");

        assert!(
            ver == 0 || ver == 1,
            "version should be 0 or 1 depending on ordering, got {ver}"
        );
        assert_eq!(new_ver, 1);
    }
}
