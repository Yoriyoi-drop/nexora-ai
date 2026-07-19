use parking_lot::Mutex;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

fn align_up(offset: usize, align: usize) -> usize {
    (offset + align - 1) & !(align - 1)
}

pub struct Arena {
    data: NonNull<u8>,
    size: usize,
    offset: AtomicUsize,
}

impl Arena {
    pub fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, 64).expect("invalid arena layout");
        let ptr = unsafe { alloc(layout) };
        let data = NonNull::new(ptr).expect("arena alloc failed");
        Self {
            data,
            size,
            offset: AtomicUsize::new(0),
        }
    }

    pub fn allocate(&self, size: usize) -> Option<*mut u8> {
        let _aligned_offset = align_up(self.offset.load(Ordering::SeqCst), 64);
        let prev = self.offset.fetch_add(size, Ordering::SeqCst);
        let start = align_up(prev, 64);
        if start + size > self.size {
            self.offset.fetch_sub(size, Ordering::SeqCst);
            return None;
        }
        let ptr = unsafe { self.data.as_ptr().add(start) };
        Some(ptr)
    }

    pub fn allocate_slice(&self, size: usize) -> Option<&mut [u8]> {
        let ptr = self.allocate(size)?;
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, size) })
    }

    pub fn reset(&self) {
        self.offset.store(0, Ordering::SeqCst);
    }

    pub fn used(&self) -> usize {
        self.offset.load(Ordering::SeqCst)
    }

    pub fn remaining(&self) -> usize {
        self.size.saturating_sub(self.used())
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, 64).expect("invalid layout");
        unsafe { dealloc(self.data.as_ptr(), layout) }
    }
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

pub struct ArenaPool {
    arenas: Mutex<Vec<Arena>>,
    arena_size: usize,
}

impl ArenaPool {
    pub fn new(arena_size: usize, count: usize) -> Self {
        let mut arenas = Vec::with_capacity(count);
        for _ in 0..count {
            arenas.push(Arena::new(arena_size));
        }
        Self {
            arenas: Mutex::new(arenas),
            arena_size,
        }
    }

    pub fn acquire(&self) -> Option<ArenaHandle> {
        let mut arenas = self.arenas.lock();
        if arenas.is_empty() {
            arenas.push(Arena::new(self.arena_size));
        }
        let idx = arenas.len() - 1;
        Some(ArenaHandle {
            pool: self as *const ArenaPool,
            arena_idx: idx,
        })
    }
}

#[allow(dead_code)]
pub struct ArenaHandle {
    pool: *const ArenaPool,
    arena_idx: usize,
}
