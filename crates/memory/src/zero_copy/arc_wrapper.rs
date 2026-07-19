use std::ops::Deref;
use std::sync::Arc;

/// Arc-based zero-copy wrapper — hindari clone data besar
#[derive(Debug, Clone, Hash)]
pub struct ArcBuffer {
    inner: Arc<[u8]>,
    offset: usize,
    len: usize,
}

impl ArcBuffer {
    pub fn new(data: impl Into<Vec<u8>>) -> Self {
        let vec: Vec<u8> = data.into();
        let len = vec.len();
        Self {
            inner: vec.into(),
            offset: 0,
            len,
        }
    }

    pub fn slice(&self, start: usize, end: usize) -> Option<Self> {
        if start > end || end > self.len {
            return None;
        }
        Some(Self {
            inner: self.inner.clone(),
            offset: self.offset + start,
            len: end - start,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Deref for ArcBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.inner[self.offset..self.offset + self.len]
    }
}

impl AsRef<[u8]> for ArcBuffer {
    fn as_ref(&self) -> &[u8] {
        self.deref()
    }
}

impl From<Vec<u8>> for ArcBuffer {
    fn from(v: Vec<u8>) -> Self {
        Self::new(v)
    }
}

impl From<String> for ArcBuffer {
    fn from(s: String) -> Self {
        Self::new(s.into_bytes())
    }
}

impl From<&str> for ArcBuffer {
    fn from(s: &str) -> Self {
        Self::new(s.as_bytes().to_vec())
    }
}
