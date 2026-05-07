# Utils Crate

## Purpose
Pure generic utilities that can be used across any Rust project, not specific to Nexora AI system.

## Scope
This crate contains **pure generic** utilities and helper functions:

### Core Utilities
- `math.rs` - Mathematical operations and algorithms
- `string.rs` - String processing and manipulation utilities
- `time.rs` - Time-related utilities and conversions
- `crypto.rs` - Cryptographic functions and utilities
- `file_utils.rs` - File I/O and path operations
- `performance.rs` - Performance measurement and optimization utilities
- `validation.rs` - Data validation utilities
- `text_processing.rs` - Text processing and analysis utilities

## When to Use
- Use this crate for **pure generic** functionality
- Use this for utilities that have no business logic dependency
- Use this for reusable functions that could be used in any Rust project

## When NOT to Use
- For **domain-specific** Nexora functionality (use `common` crate instead)
- For Nexora-specific configurations (use `common` crate)
- For Nexora-specific error handling (use `common` crate)
- For Nexora task types (use `common` crate)

## Dependencies
This crate depends only on standard library and a few essential workspace dependencies.

## Examples
```rust
use nexora_utils::math::calculate_average;
use nexora_utils::string::normalize_text;
use nexora_utils::time::format_duration;
use nexora_utils::crypto::hash_string;
```

## Relationship to Common Crate
- **Utils**: Pure generic, reusable across any Rust project
- **Common**: Domain-specific, business logic, Nexora-specific

## Design Principles
- No dependencies on Nexora-specific types
- Pure functions without side effects
- Generic and reusable implementations
- Well-documented with examples
- Comprehensive test coverage
