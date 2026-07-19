use memmap2::{Mmap, MmapOptions, MmapMut};
use std::fs::File;
use std::path::Path;

/// Memory-mapped file reader — zero-copy file I/O
pub struct MmapFile {
    mmap: Mmap,
    #[allow(dead_code)]
    path: String,
}

impl MmapFile {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self {
            mmap,
            path: path.as_ref().to_string_lossy().to_string(),
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    pub fn path(&self) -> &str {
        &self.path
    }
}

pub struct MmapFileMut {
    mmap: MmapMut,
    #[allow(dead_code)]
    path: String,
}

impl MmapFileMut {
    pub fn create(path: impl AsRef<Path>, size: usize) -> std::io::Result<Self> {
        let file = File::create(path.as_ref())?;
        file.set_len(size as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self {
            mmap,
            path: path.as_ref().to_string_lossy().to_string(),
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.mmap
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn flush(&self) -> std::io::Result<()> {
        self.mmap.flush()
    }
}
