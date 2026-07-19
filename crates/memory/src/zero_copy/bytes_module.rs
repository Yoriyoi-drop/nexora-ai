use bytes::Bytes;

pub use bytes::Bytes as BytesBuf;

#[derive(Debug, Clone)]
pub struct SharedString {
    inner: Bytes,
}

impl SharedString {
    pub fn new(s: impl Into<String>) -> Self {
        Self {
            inner: Bytes::from(s.into().into_bytes()),
        }
    }

    pub fn from_bytes(bytes: Bytes) -> Self {
        Self { inner: bytes }
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.inner).unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> Option<Self> {
        if range.end > self.inner.len() {
            return None;
        }
        Some(Self {
            inner: self.inner.slice(range),
        })
    }
}

impl From<String> for SharedString {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for SharedString {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl std::ops::Deref for SharedString {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}
