// Core Traits for Foundation Components
//
// Essential traits used across the foundation library

use std::fmt::Debug;

/// Trait for components that can be initialized
pub trait Initializable {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Initialize the component
    fn initialize(&mut self) -> Result<(), Self::Error>;
    
    /// Check if the component is initialized
    fn is_initialized(&self) -> bool;
}

/// Trait for components that can be reset
pub trait Resettable {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Reset the component to its initial state
    fn reset(&mut self) -> Result<(), Self::Error>;
}

/// Trait for serializable components
pub trait Serializable {
    type Error: std::error::Error + Send + Sync + 'static;
    type Output: serde::Serialize + for<'de> serde::Deserialize<'de>;
    
    /// Serialize the component
    fn serialize(&self) -> Result<Self::Output, Self::Error>;
    
    /// Deserialize into the component
    fn deserialize(&mut self, data: Self::Output) -> Result<(), Self::Error>;
}

/// Trait for configurable components
pub trait Configurable<C> {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Apply configuration to the component
    fn configure(&mut self, config: C) -> Result<(), Self::Error>;
    
    /// Get current configuration
    fn config(&self) -> &C;
}

/// Trait for validating components
pub trait Validatable {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Validate the component state
    fn validate(&self) -> Result<(), Self::Error>;
}

/// Trait for components that support cloning with configuration
pub trait CloneableWithConfig<C> {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Clone the component with new configuration
    fn clone_with_config(&self, config: C) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

/// Trait for components that can be measured/metric'd
pub trait Measurable {
    type Metrics: Debug + Clone;
    
    /// Get current metrics
    fn metrics(&self) -> Self::Metrics;
    
    /// Reset metrics
    fn reset_metrics(&mut self);
}
