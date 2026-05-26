//! Nexora Infrastructure - Infrastructure layer
//!
//! Provides shared infrastructure components:
//! - logging
//! - metrics
//! - storage
//! - security
//! - config
//! - errors

pub mod common;
pub mod utils;

#[cfg(test)]
mod tests {
    #[test]
    fn test_modules_accessible() {
        let _ts = crate::common::unix_timestamp();
        let _dur = crate::common::format_duration(3600);
        let _trunc = crate::utils::truncate_text("hello", 3);
        let _sanitized = crate::utils::sanitize_filename("test");
    }
}
