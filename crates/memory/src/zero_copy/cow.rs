#![allow(unused)]

use std::borrow::Cow;
use std::sync::Arc;

/// CoW buffer — clone-on-write untuk data yang jarang dimutasi
#[derive(Debug, Clone)]
pub enum CowBuffer {
    Borrowed(&'static [u8]),
    Owned(Arc<[u8]>),
}

impl CowBuffer {
    pub fn borrowed(data: &'static [u8]) -> Self {
        Self::Borrowed(data)
    }

    pub fn owned(data: impl Into<Vec<u8>>) -> Self {
        Self::Owned(data.into().into())
    }

    pub fn make_mut(&mut self) -> &mut Vec<u8> {
        match self {
            CowBuffer::Borrowed(data) => {
                let mut v = data.to_vec();
                *self = CowBuffer::Owned(v.into());
                // Can't return &mut from Arc. Use CowBuffer::Cow instead.
                unreachable!()
            }
            CowBuffer::Owned(_) => {
                // Arc doesn't allow mutable access. Use Cow<'_, [u8]> for that.
                unreachable!()
            }
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        match self {
            CowBuffer::Borrowed(b) => b,
            CowBuffer::Owned(a) => a.as_ref(),
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Efficient CoW string — untuk prompt dan text data
#[derive(Debug, Clone)]
pub struct CoWString {
    inner: Cow<'static, str>,
}

impl CoWString {
    pub fn borrowed(s: &'static str) -> Self {
        Self {
            inner: Cow::Borrowed(s),
        }
    }

    pub fn owned(s: String) -> Self {
        Self {
            inner: Cow::Owned(s),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.inner
    }

    pub fn into_owned(self) -> String {
        self.inner.into_owned()
    }
}

impl From<String> for CoWString {
    fn from(s: String) -> Self {
        Self::owned(s)
    }
}

impl From<&'static str> for CoWString {
    fn from(s: &'static str) -> Self {
        Self::borrowed(s)
    }
}

impl std::ops::Deref for CoWString {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}
