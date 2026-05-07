# Experimental Research

This directory contains active, unstable experimental code.

**IMPORTANT:** Code in this directory is NOT production-ready and should NOT be used in production systems.

## Purpose

- Test new ideas and prototypes
- Experiment with unstable features
- Develop proof-of-concept implementations
- Active research projects that are still evolving

## Guidelines

1. **DO NOT** import experimental code into production crates
2. **DO NOT** depend on experimental code from stable code
3. **MUST** clearly mark as experimental with warnings
4. **SHOULD** move to appropriate crate once stable
5. **MUST** have tests before considering for promotion

## Promotion Process

When experimental code becomes stable:
1. Move to appropriate crate (foundation, cognition, runtime, etc.)
2. Add proper documentation
3. Ensure trait implementations if applicable
4. Update dependencies
5. Remove from experimental/

## Current Experiments

- (Add descriptions of active experiments here)
