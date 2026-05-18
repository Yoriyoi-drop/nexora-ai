import { Agent, AIModel, MemoryNode, TokenFlow, TimelineSnapshot, InfraNode, CognitiveEvent, AgentStatus } from '../store/nexoraStore';

const CLUSTERS = ['Reasoning', 'Memory', 'Planning', 'Validation', 'Orchestration'];

export function generateAgents(): Agent[] {
  const roles = [
    { role: 'Orchestrator', model: 'Nexum' },
    { role: 'Reasoner', model: 'Omnis' },
    { role: 'Validator', model: 'Axiom' },
    { role: 'Memory-Curator', model: 'Kronos' },
    { role: 'Planner', model: 'Omnis' },
    { role: 'Critic', model: 'Axiom' },
    { role: 'Synthesizer', model: 'Genesis' },
    { role: 'Router', model: 'Nexum' },
    { role: 'Embedder', model: 'Swift' },
    { role: 'Inspector', model: 'Vortex' },
  ];

  const statuses: AgentStatus[] = ['active', 'active', 'active', 'active', 'idle', 'warning', 'critical', 'active'];

  const agents = Array.from({ length: 49 }, (_, i) => {
    const roleData = roles[i % roles.length];
    const cluster = CLUSTERS[i % CLUSTERS.length];
    const status = statuses[i % statuses.length];
    const entropy = status === 'critical' ? 0.75 + Math.random() * 0.25
      : status === 'warning' ? 0.5 + Math.random() * 0.25
      : Math.random() * 0.45;

    // Cluster layout
    const clusterIdx = CLUSTERS.indexOf(cluster);
    const clusterAngle = (clusterIdx / CLUSTERS.length) * Math.PI * 2;
    const clusterR = 200;
    const clusterCX = 500 + clusterR * Math.cos(clusterAngle);
    const clusterCY = 340 + clusterR * Math.sin(clusterAngle);
    const localAngle = ((i % 10) / 10) * Math.PI * 2;
    const localR = 60 + Math.random() * 40;

    return {
      id: `agent-${i + 1}`,
      name: `${roleData.role}-${String(i + 1).padStart(2, '0')}`,
      role: roleData.role,
      status,
      model: roleData.model,
      tokensPerSec: Math.floor(50 + Math.random() * 200),
      entropy,
      memoryUsage: 0.3 + Math.random() * 0.5,
      cpuLoad: 20 + Math.random() * 60,
      gpuLoad: 30 + Math.random() * 60,
      hallucinations: status === 'critical' ? Math.floor(5 + Math.random() * 15) : Math.floor(Math.random() * 4),
      reasoningDepth: Math.floor(3 + Math.random() * 8),
      x: clusterCX + localR * Math.cos(localAngle),
      y: clusterCY + localR * Math.sin(localAngle),
      cluster,
      connections: [] as string[],
      thoughtsPending: Math.floor(Math.random() * 12),
      uptime: Math.floor(3600 + Math.random() * 86400),
      lastEvent: ['token_generated', 'memory_accessed', 'reasoning_step', 'validation_passed', 'planning_updated'][i % 5],
    };
  });

  // Add connections after generation
  agents.forEach((agent) => {
    const numConns = 2 + Math.floor(Math.random() * 4);
    agent.connections = Array.from({ length: numConns }, () => {
      const target = agents[Math.floor(Math.random() * agents.length)];
      return target.id;
    }).filter(id => id !== agent.id);
  });

  return agents;
}

export function generateModels(): AIModel[] {
  return [
    { id: 'm1', name: 'Omnis', type: 'reasoning', provider: 'NXR-OMNIS', gpuMemory: 52, gpuMemoryTotal: 80, throughput: 1840, latency: 142, accuracy: 0.94, activeAgents: 11, temperature: 0.7, quantization: 'FP16' },
    { id: 'm2', name: 'Vortex', type: 'llm', provider: 'NXR-VORTEX', gpuMemory: 44, gpuMemoryTotal: 80, throughput: 1620, latency: 168, accuracy: 0.96, activeAgents: 9, temperature: 0.6, quantization: 'FP16' },
    { id: 'm3', name: 'Aether', type: 'reasoning', provider: 'NXR-ÆTHER', gpuMemory: 38, gpuMemoryTotal: 80, throughput: 2100, latency: 124, accuracy: 0.93, activeAgents: 8, temperature: 0.8, quantization: 'BF16' },
    { id: 'm4', name: 'Spectra', type: 'vision', provider: 'NXR-SPECTRA', gpuMemory: 48, gpuMemoryTotal: 80, throughput: 890, latency: 210, accuracy: 0.89, activeAgents: 7, temperature: 0.75, quantization: 'INT8' },
    { id: 'm5', name: 'Nexum', type: 'validator', provider: 'NXR-NEXUM', gpuMemory: 28, gpuMemoryTotal: 40, throughput: 1240, latency: 155, accuracy: 0.91, activeAgents: 6, temperature: 0.65, quantization: 'FP16' },
    { id: 'm6', name: 'Axiom', type: 'reasoning', provider: 'NXR-AXIOM', gpuMemory: 56, gpuMemoryTotal: 80, throughput: 3400, latency: 62, accuracy: 0.87, activeAgents: 5, temperature: 0.9, quantization: 'INT8' },
    { id: 'm7', name: 'Cipher', type: 'llm', provider: 'NXR-CIPHER', gpuMemory: 24, gpuMemoryTotal: 40, throughput: 12000, latency: 18, accuracy: 0.97, activeAgents: 3, temperature: 0, quantization: 'FP32' },
    { id: 'm8', name: 'Swift', type: 'embedding', provider: 'NXR-SWIFT', gpuMemory: 8, gpuMemoryTotal: 16, throughput: 22000, latency: 8, accuracy: 0.92, activeAgents: 2, temperature: 0, quantization: 'INT4' },
    { id: 'm9', name: 'Kronos', type: 'validator', provider: 'NXR-KRONOS', gpuMemory: 32, gpuMemoryTotal: 40, throughput: 560, latency: 280, accuracy: 0.95, activeAgents: 4, temperature: 0.5, quantization: 'FP16' },
    { id: 'm10', name: 'Genesis', type: 'reasoning', provider: 'NXR-GENESIS', gpuMemory: 64, gpuMemoryTotal: 80, throughput: 720, latency: 340, accuracy: 0.88, activeAgents: 3, temperature: 0.85, quantization: 'BF16' },
  ];
}

export function generateMemoryNodes(): MemoryNode[] {
  const types: ('episodic' | 'semantic' | 'procedural' | 'working')[] = ['episodic', 'semantic', 'procedural', 'working'];
  const clusters = ['Knowledge-Core', 'Event-Log', 'Skill-Tree', 'Context-Buffer'];

  return Array.from({ length: 80 }, (_, i) => {
    const type = types[i % types.length];
    const cluster = clusters[i % clusters.length];
    const strength = Math.random();
    const angle = (i / 80) * Math.PI * 2;
    const r = 80 + Math.random() * 200;

    return {
      id: `mem-${i}`,
      label: `${type}-${String(i).padStart(3, '0')}`,
      type,
      strength,
      stability: Math.max(0, strength + (Math.random() - 0.5) * 0.4),
      accessCount: Math.floor(Math.random() * 500),
      lastAccess: Date.now() - Math.floor(Math.random() * 3600000),
      connections: [] as string[],
      x: 400 + r * Math.cos(angle),
      y: 300 + r * Math.sin(angle),
      cluster,
    };
  });
}

export function generateTimeline(): TimelineSnapshot[] {
  const now = Date.now();
  return Array.from({ length: 60 }, (_, i) => {
    const t = now - (59 - i) * 30000;
    const base = i / 60;
    return {
      timestamp: t,
      label: new Date(t).toLocaleTimeString(),
      hallucinationRate: 0.05 + Math.sin(base * Math.PI * 3) * 0.04 + Math.random() * 0.02,
      reasoningDepth: 5 + Math.sin(base * Math.PI * 2) * 2 + Math.random(),
      memoryFragmentation: 0.2 + base * 0.15 + Math.random() * 0.05,
      activeAgents: Math.floor(44 + Math.random() * 6),
      entropy: 0.3 + Math.sin(base * Math.PI * 4) * 0.15 + Math.random() * 0.05,
      tokenThroughput: 8000 + Math.sin(base * Math.PI * 2) * 2000 + Math.random() * 500,
    };
  });
}

export function generateInfraNodes(): InfraNode[] {
  return [
    { id: 'inf-1', label: 'GPU Cluster A', type: 'gpu-cluster', status: 'healthy', load: 0.72, connections: ['inf-3', 'inf-5'] },
    { id: 'inf-2', label: 'GPU Cluster B', type: 'gpu-cluster', status: 'degraded', load: 0.88, connections: ['inf-3', 'inf-5'] },
    { id: 'inf-3', label: 'NATS JetStream', type: 'nats', status: 'healthy', load: 0.45, connections: ['inf-1', 'inf-2', 'inf-4', 'inf-6'] },
    { id: 'inf-4', label: 'Delta Lake', type: 'delta-lake', status: 'healthy', load: 0.34, connections: ['inf-3', 'inf-5'] },
    { id: 'inf-5', label: 'Vector DB', type: 'vector-db', status: 'healthy', load: 0.61, connections: ['inf-1', 'inf-2', 'inf-4'] },
    { id: 'inf-6', label: 'Orchestrator', type: 'orchestrator', status: 'healthy', load: 0.55, connections: ['inf-3'] },
    { id: 'inf-7', label: 'API Gateway', type: 'gateway', status: 'critical', load: 0.95, connections: ['inf-6'] },
  ];
}

export function generateTokenFlows(): TokenFlow[] {
  return [
    { source: 'Omnis', target: 'Nexum', volume: 4200, efficiency: 0.91, bottleneck: false },
    { source: 'Vortex', target: 'Nexum', volume: 3800, efficiency: 0.88, bottleneck: false },
    { source: 'Swift', target: 'Kronos', volume: 2100, efficiency: 0.72, bottleneck: true },
    { source: 'Aether', target: 'Nexum', volume: 5600, efficiency: 0.94, bottleneck: false },
    { source: 'Nexum', target: 'Axiom', volume: 8900, efficiency: 0.85, bottleneck: true },
    { source: 'Axiom', target: 'Cipher', volume: 3200, efficiency: 0.90, bottleneck: false },
    { source: 'Kronos', target: 'Axiom', volume: 12000, efficiency: 0.97, bottleneck: false },
    { source: 'Genesis', target: 'Spectra', volume: 2800, efficiency: 0.82, bottleneck: false },
  ];
}

export function generateCognitiveEvents(): CognitiveEvent[] {
  const types: CognitiveEvent['type'][] = ['reasoning', 'memory', 'planning', 'validation', 'hallucination', 'completion'];
  const msgs = {
    reasoning: ['Multi-hop chain activated', 'Causal inference step', 'Counterfactual branch explored', 'Deductive closure reached'],
    memory: ['Episodic recall triggered', 'Semantic cluster updated', 'Working mem overflow', 'Consolidation cycle'],
    planning: ['Goal decomposition', 'Sub-task allocated', 'Resource constraint detected', 'Plan revision initiated'],
    validation: ['Cross-agent consensus', 'Fact verification passed', 'Logical consistency OK', 'Uncertainty flagged'],
    hallucination: ['Confabulation detected', 'Ungrounded assertion', 'Context window drift', 'Source conflict'],
    completion: ['Task fulfilled', 'Pipeline closed', 'Result synthesized', 'Agent handoff complete'],
  };
  const now = Date.now();
  return Array.from({ length: 40 }, (_, i) => {
    const type = types[i % types.length];
    return {
      id: `evt-${i}`,
      timestamp: now - i * 8000,
      agentId: `agent-${(i % 49) + 1}`,
      type,
      message: msgs[type][i % msgs[type].length],
      entropy: Math.random(),
      severity: type === 'hallucination' ? 'critical' : type === 'validation' ? 'warn' : 'info',
    };
  });
}
