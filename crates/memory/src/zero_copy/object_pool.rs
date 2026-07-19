use parking_lot::Mutex;
use std::any::Any;
use std::collections::HashMap;

/// Type-erased object pool
pub struct ObjectPool {
    pools: Mutex<HashMap<std::any::TypeId, Box<dyn Any + Send>>>,
}

impl ObjectPool {
    pub fn new() -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
        }
    }

    pub fn register<T: Default + Send + 'static>(&self, capacity: usize) {
        let mut pools = self.pools.lock();
        let pool: TypedObjectPool<T> = TypedObjectPool::new(capacity);
        pools.insert(std::any::TypeId::of::<T>(), Box::new(pool));
    }

    pub fn acquire<T: Default + Send + 'static>(&self) -> Option<T> {
        let mut pools = self.pools.lock();
        let pool = pools.get_mut(&std::any::TypeId::of::<T>())?;
        pool.downcast_mut::<TypedObjectPool<T>>()?.acquire()
    }

    pub fn release<T: Default + Send + 'static>(&self, obj: T) {
        let mut pools = self.pools.lock();
        if let Some(pool) = pools.get_mut(&std::any::TypeId::of::<T>()) {
            if let Some(p) = pool.downcast_mut::<TypedObjectPool<T>>() {
                p.release(obj);
            }
        }
    }
}

impl Default for ObjectPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Typed object pool — reuse objects instead of re-allocating
pub struct TypedObjectPool<T: Default> {
    pool: Mutex<Vec<T>>,
    capacity: usize,
    created: std::sync::atomic::AtomicUsize,
    reused: std::sync::atomic::AtomicUsize,
}

impl<T: Default> TypedObjectPool<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            pool: Mutex::new(Vec::with_capacity(capacity)),
            capacity,
            created: std::sync::atomic::AtomicUsize::new(0),
            reused: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn acquire(&self) -> Option<T> {
        let mut pool = self.pool.lock();
        if let Some(obj) = pool.pop() {
            self.reused.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(obj)
        } else {
            self.created.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(T::default())
        }
    }

    pub fn release(&self, obj: T) {
        let mut pool = self.pool.lock();
        if pool.len() < self.capacity {
            pool.push(obj);
        }
    }

    pub fn stats(&self) -> ObjectPoolStats {
        ObjectPoolStats {
            pool_size: self.pool.lock().len(),
            capacity: self.capacity,
            created: self.created.load(std::sync::atomic::Ordering::Relaxed),
            reused: self.reused.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ObjectPoolStats {
    pub pool_size: usize,
    pub capacity: usize,
    pub created: usize,
    pub reused: usize,
}
