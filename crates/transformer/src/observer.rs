use std::fmt;
use std::sync::Mutex;

/// Trait for objects that want to be notified when model weights change.
/// Fires after `sync_to_inference()`, `load_checkpoint()`, or any other
/// weight-mutating operation completes.
pub trait WeightObserver: Send + Sync {
    fn on_weights_changed(&self);
}

/// A lightweight notifier embedded in `CausalLM`.
/// Observers are called synchronously on every weight change.
pub struct WeightNotifier {
    observers: Mutex<Vec<Box<dyn WeightObserver>>>,
}

impl fmt::Debug for WeightNotifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightNotifier")
            .field("observer_count", &self.observers.lock().map(|o| o.len()).unwrap_or(0))
            .finish()
    }
}

impl WeightNotifier {
    pub fn new() -> Self {
        Self {
            observers: Mutex::new(Vec::new()),
        }
    }

    pub fn add(&self, observer: Box<dyn WeightObserver>) {
        if let Ok(mut obs) = self.observers.lock() {
            obs.push(observer);
        }
    }

    pub fn notify(&self) {
        if let Ok(obs) = self.observers.lock() {
            for o in obs.iter() {
                o.on_weights_changed();
            }
        }
    }
}

impl Default for WeightNotifier {
    fn default() -> Self {
        Self::new()
    }
}
