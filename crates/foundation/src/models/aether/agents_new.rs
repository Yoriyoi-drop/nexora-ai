//! NXR-ÆTHER Agents System
//!
//! ╔══════════════════════════════════════════════════════════════════════════╗
//! ║  NOTICE: This file was a work-in-progress refactor of agents/mod.rs.   ║
//! ║                                                                        ║
//! ║  It is currently DEAD CODE — it is NOT declared in mod.rs and no       ║
//! ║  module imports it. The individual agent sub-modules                   ║
//! ║  (empathy_prime.rs, psyche_analyzer.rs, emotion_weaver.rs,             ║
//! ║   culture_adapter.rs) also exist on disk but are NOT wired into the    ║
//! ║  module tree (agents/mod.rs does not declare `pub mod` for them).      ║
//! ║                                                                        ║
//! ║  The live agent system lives in agents/mod.rs with a simpler,          ║
//! ║  self-contained implementation (EmpahCoreAgent, ToneMapperAgent,       ║
//! ║  ContextWeaveAgent, SoulMirrorAgent) and does NOT depend on types      ║
//! ║  from sub-module config files.                                         ║
//! ║                                                                        ║
//! ║  To revive:                                                            ║
//! ║    1. Wire sub-modules into agents/mod.rs with `pub mod <name>;`       ║
//! ║    2. Wire this file into aether/mod.rs with `mod agents_new;`         ║
//! ║    3. Resolve config field mappings (AetherConfig → per-agent configs) ║
//! ╚══════════════════════════════════════════════════════════════════════════╝
