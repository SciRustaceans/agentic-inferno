use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

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
#[derive(Debug, Clone)]
pub struct SharedState {
    pub document: Arc<RwLock<DocumentBuffer>>,
}

impl SharedState {
    /// Create a new shared state with the given `initial_content` at version 0.
    pub fn new(initial_content: String) -> Self {
        Self {
            document: Arc::new(RwLock::new(DocumentBuffer::new(initial_content))),
        }
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
        let writer_handle = std::thread::spawn(move || writer.update("written concurrently".to_string()));

        let (ver, _content) = reader_handle.join().expect("reader panicked");
        let new_ver = writer_handle.join().expect("writer panicked");

        assert!(
            ver == 0 || ver == 1,
            "version should be 0 or 1 depending on ordering, got {ver}"
        );
        assert_eq!(new_ver, 1);
    }
}
