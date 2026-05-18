// Semantic color coding system
export const SEMANTIC_COLORS = {
  // Cognitive states
  activeReasoning: '#06b6d4',    // Cyan - Active reasoning
  memorySystems: '#8b5cf6',      // Purple - Memory systems
  stableVerified: '#10b981',     // Green - Stable verified
  uncertain: '#f59e0b',          // Yellow - Uncertain
  contradiction: '#ef4444',      // Red - Contradiction
  orchestrator: '#ffffff',       // White - Orchestrator
  
  // Legacy compatibility
  active: '#06b6d4',
  idle: '#475569',
  warning: '#f59e0b',
  critical: '#ef4444',
  isolated: '#7c3aed',
} as const;

export const CLUSTER_COLORS: Record<string, string> = {
  Reasoning: '#06b6d4',      // Cyan - Active reasoning
  Memory: '#8b5cf6',         // Purple - Memory systems
  Planning: '#3b82f6',       // Blue - Planning
  Validation: '#10b981',     // Green - Stable verified
  Orchestration: '#ffffff', // White - Orchestrator
};

// Edge styles with semantic meaning
export const EDGE_STYLES = {
  async: { lineDash: [4, 6], lineWidth: 0.8, pulse: false },      // Dotted = async
  direct: { lineDash: [], lineWidth: 1.2, pulse: false },          // Solid = direct inference
  active: { lineDash: [], lineWidth: 1.5, pulse: true },          // Pulse = active token exchange
} as const;

// Canvas configuration
export const CANVAS_CONFIG = {
  BASE_WIDTH: 1000,
  BASE_HEIGHT: 680,
  BACKGROUND_PARTICLE_COUNT: 50,
  FOG_POINT_COUNT: 8,
  GRID_SIZE: 40,
  CLUSTER_BLOB_RADIUS: 120,
  HEAT_RADIUS_BASE: 60,
  DIFFUSION_RADIUS_BASE: 80,
  CLICK_THRESHOLD: 20,
} as const;

// Animation timing
export const ANIMATION_CONFIG = {
  TIME_STEP: 0.016,
  BACKGROUND_SPEED: 0.1,
  FOG_SPEED: 0.05,
  PULSE_SPEED: 2,
  ENTROPY_FLICKER_SPEED: 15,
  SCANLINE_SPEED: 8,
} as const;
