# Common Crate

## Purpose
Domain-specific shared types, configurations, and error handling for the Nexora AI system.

## Scope
This crate contains **domain-specific** shared components that are used across multiple Nexora modules:

### Core Types
- `config.rs` - Configuration structures for Nexora components
- `task_types.rs` - Task type definitions and enums
- `error_recovery.rs` - Error recovery strategies and utilities
- `query_utils.rs` - Query parsing and validation utilities

### Logging & Utilities
- `logging.rs` - Structured logging configuration and utilities
- `lib.rs` - Crate exports and common re-exports

## When to Use
- Use this crate for **domain-specific** shared functionality
- Use this for types that have business logic meaning in the Nexora system
- Use this for configuration and error handling that's specific to Nexora

## When NOT to Use
- For **pure generic** utilities (use `utils` crate instead)
- For mathematical operations (use `utils` crate)
- For string processing utilities (use `utils` crate)
- For generic data structures (use `utils` crate)

## Dependencies
This crate depends on workspace-level dependencies and is designed to be used by other Nexora crates.

## Examples
```rust
use nexora_common::config::AgentConfig;
use nexora_common::task_types::TaskType;
use nexora_common::error_recovery::RecoveryStrategy;
```

## Relationship to Utils Crate
- **Common**: Domain-specific, business logic, Nexora-specific
- **Utils**: Pure generic, reusable across any Rust project
