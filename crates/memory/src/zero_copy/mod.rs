pub mod arc_wrapper;
pub mod bytes_module;
pub mod cow;
pub mod memmap;
pub mod arena;
pub mod object_pool;

pub use arc_wrapper::*;
pub use bytes_module::*;
pub use cow::*;
pub use memmap::*;
pub use arena::*;
pub use object_pool::*;

/// Zero-copy buffer — hindari clone data besar
#[derive(Debug)]
pub enum ZeroCopyBuffer {
    Arc(std::sync::Arc<[u8]>),
    Bytes(bytes::Bytes),
    Mmap(memmap2::Mmap),
    Arena(usize, usize),
}

impl Clone for ZeroCopyBuffer {
    fn clone(&self) -> Self {
        match self {
            ZeroCopyBuffer::Arc(a) => ZeroCopyBuffer::Arc(a.clone()),
            ZeroCopyBuffer::Bytes(b) => ZeroCopyBuffer::Bytes(b.clone()),
            ZeroCopyBuffer::Mmap(m) => ZeroCopyBuffer::Arc(m.as_ref().into()),
            ZeroCopyBuffer::Arena(a, b) => ZeroCopyBuffer::Arena(*a, *b),
        }
    }
}

impl ZeroCopyBuffer {
    pub fn len(&self) -> usize {
        match self {
            ZeroCopyBuffer::Arc(a) => a.len(),
            ZeroCopyBuffer::Bytes(b) => b.len(),
            ZeroCopyBuffer::Mmap(m) => m.len(),
            ZeroCopyBuffer::Arena(_, _) => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        match self {
            ZeroCopyBuffer::Arc(a) => a.as_ref(),
            ZeroCopyBuffer::Bytes(b) => b.as_ref(),
            ZeroCopyBuffer::Mmap(m) => m.as_ref(),
            ZeroCopyBuffer::Arena(_, _) => &[],
        }
    }
}

impl From<Vec<u8>> for ZeroCopyBuffer {
    fn from(v: Vec<u8>) -> Self {
        ZeroCopyBuffer::Arc(v.into())
    }
}

impl From<bytes::Bytes> for ZeroCopyBuffer {
    fn from(b: bytes::Bytes) -> Self {
        ZeroCopyBuffer::Bytes(b)
    }
}

impl From<memmap2::Mmap> for ZeroCopyBuffer {
    fn from(m: memmap2::Mmap) -> Self {
        ZeroCopyBuffer::Mmap(m)
    }
}

impl From<String> for ZeroCopyBuffer {
    fn from(s: String) -> Self {
        ZeroCopyBuffer::Arc(s.into_bytes().into())
    }
}
