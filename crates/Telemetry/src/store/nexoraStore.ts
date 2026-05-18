import { create } from 'zustand';
import {
  generateAgents,
  generateModels,
  generateMemoryNodes,
  generateTimeline,
  generateInfraNodes,
  generateTokenFlows,
  generateCognitiveEvents,
} from '../utils/mockDataGenerator';
import { telemetryService } from '../services/telemetryService';
import type { DataSource, SystemMetrics, NexoraAIMetrics, AggregatedTelemetry, AgentTelemetry, MemoryTelemetry, MemorySummary, InferenceTelemetry, HallucinationTelemetry, TrainingTelemetry, ModelTelemetry, TokenFlowTelemetry, PipelineTelemetry } from '../services/telemetryService';

export type AgentStatus = 'active' | 'idle' | 'warning' | 'critical' | 'isolated';
export type ModelType = 'llm' | 'vision' | 'embedding' | 'reasoning' | 'validator';
export type TabView = 'topology' | 'memory' | 'tokenflow' | 'infrastructure' | 'timeline' | 'entropy';

export interface Agent {
  id: string;
  name: string;
  role: string;
  status: AgentStatus;
  model: string;
  tokensPerSec: number;
  entropy: number;
  memoryUsage: number;
  cpuLoad: number;
  gpuLoad: number;
  hallucinations: number;
  reasoningDepth: number;
  x: number;
  y: number;
  cluster: string;
  connections: string[];
  thoughtsPending: number;
  uptime: number;
  lastEvent: string;
}

export interface AIModel {
  id: string;
  name: string;
  type: ModelType;
  provider: string;
  gpuMemory: number;
  gpuMemoryTotal: number;
  throughput: number;
  latency: number;
  accuracy: number;
  activeAgents: number;
  temperature: number;
  quantization: string;
}

export interface MemoryNode {
  id: string;
  label: string;
  type: 'episodic' | 'semantic' | 'procedural' | 'working';
  strength: number;
  stability: number;
  accessCount: number;
  lastAccess: number;
  connections: string[];
  x: number;
  y: number;
  cluster: string;
}

export interface TokenFlow {
  source: string;
  target: string;
  volume: number;
  efficiency: number;
  bottleneck: boolean;
}

export interface TimelineSnapshot {
  timestamp: number;
  label: string;
  hallucinationRate: number;
  reasoningDepth: number;
  memoryFragmentation: number;
  activeAgents: number;
  entropy: number;
  tokenThroughput: number;
}

export interface InfraNode {
  id: string;
  label: string;
  type: 'gpu-cluster' | 'nats' | 'delta-lake' | 'vector-db' | 'orchestrator' | 'gateway';
  status: 'healthy' | 'degraded' | 'critical';
  load: number;
  connections: string[];
}

export interface CognitiveEvent {
  id: string;
  timestamp: number;
  agentId: string;
  type: 'reasoning' | 'memory' | 'planning' | 'validation' | 'hallucination' | 'completion';
  message: string;
  entropy: number;
  severity: 'info' | 'warn' | 'critical';
}

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

interface NexoraState {
  // Tab states
  activeTab: TabView;
  selectedAgent: string | null;
  isLive: boolean;
  timeOffset: number;
  compareMode: boolean;
  compareRunA: string;
  compareRunB: string;

  // Data source
  dataSource: DataSource;
  connectionStatus: ConnectionStatus;
  aiConnectionUrl: string;

  // Mock data (fallback)
  agents: Agent[];
  models: AIModel[];
  memoryNodes: MemoryNode[];
  tokenFlows: TokenFlow[];
  timeline: TimelineSnapshot[];
  infraNodes: InfraNode[];
  cognitiveEvents: CognitiveEvent[];

  // Real telemetry data (live)
  realSystemMetrics: SystemMetrics | null;
  realAIMetrics: NexoraAIMetrics | null;
  realAggregated: AggregatedTelemetry | null;
  realAgents: AgentTelemetry[];
  realMemoryNodes: MemoryTelemetry[];
  realMemorySummary: MemorySummary | null;
  realInference: InferenceTelemetry | null;
  realHallucinations: HallucinationTelemetry | null;
  realTraining: TrainingTelemetry | null;
  realModels: ModelTelemetry[];
  realTokenFlows: TokenFlowTelemetry[];
  realPipelines: PipelineTelemetry[];

  systemMetrics: {
    totalAgents: number;
    activeAgents: number;
    totalTokensPerSec: number;
    avgEntropy: number;
    hallucinationRate: number;
    gpuUtilization: number;
    natsEventRate: number;
    deltaLakeSizeTB: number;
    cognitiveAnomalies: number;
    reasoningDepthAvg: number;
    memoryFragmentation: number;
    swarmCoherence: number;
    cpuPercent: number;
    ramPercent: number;
    ramUsedGb: number;
    ramTotalGb: number;
    inferenceLatencyMs: number;
    kvCacheHitRate: number;
    trainingLoss: number | null;
  };

  // Actions
  setActiveTab: (tab: TabView) => void;
  setSelectedAgent: (id: string | null) => void;
  setIsLive: (live: boolean) => void;
  setTimeOffset: (offset: number) => void;
  toggleCompareMode: () => void;
  tickSimulation: () => void;
  setDataSource: (source: DataSource) => void;
  connectToAI: (url: string) => Promise<void>;
  disconnectFromAI: () => Promise<void>;
  setConnectionStatus: (status: ConnectionStatus) => void;
  updateFromSystemMetrics: (metrics: SystemMetrics) => void;
  updateFromAIMetrics: (metrics: NexoraAIMetrics) => void;
  updateFromAggregatedTelemetry: (agg: AggregatedTelemetry) => void;
  convertAgentTelemetry: (agents: AgentTelemetry[]) => Agent[];
  convertMemoryTelemetry: (nodes: MemoryTelemetry[]) => MemoryNode[];
  convertModelTelemetry: (models: ModelTelemetry[]) => AIModel[];
  convertTokenFlowTelemetry: (flows: TokenFlowTelemetry[]) => TokenFlow[];
}

const initialAgents = generateAgents();

export const useNexoraStore = create<NexoraState>((set, get) => ({
  activeTab: 'topology',
  selectedAgent: null,
  isLive: true,
  timeOffset: 0,
  compareMode: false,
  compareRunA: 'Run-Alpha',
  compareRunB: 'Run-Beta',

  dataSource: 'auto',
  connectionStatus: 'disconnected',
  aiConnectionUrl: '',

  agents: initialAgents,
  models: generateModels(),
  memoryNodes: generateMemoryNodes(),
  tokenFlows: generateTokenFlows(),
  timeline: generateTimeline(),
  infraNodes: generateInfraNodes(),
  cognitiveEvents: generateCognitiveEvents(),

  realSystemMetrics: null,
  realAIMetrics: null,
  realAggregated: null,
  realAgents: [],
  realMemoryNodes: [],
  realMemorySummary: null,
  realInference: null,
  realHallucinations: null,
  realTraining: null,
  realModels: [],
  realTokenFlows: [],
  realPipelines: [],

  systemMetrics: {
    totalAgents: 49,
    activeAgents: 42,
    totalTokensPerSec: 11840,
    avgEntropy: 0.31,
    hallucinationRate: 0.047,
    gpuUtilization: 0.74,
    natsEventRate: 4900,
    deltaLakeSizeTB: 2.84,
    cognitiveAnomalies: 3,
    reasoningDepthAvg: 6.2,
    memoryFragmentation: 0.28,
    swarmCoherence: 0.87,
    cpuPercent: 45,
    ramPercent: 62,
    ramUsedGb: 12.8,
    ramTotalGb: 32,
    inferenceLatencyMs: 0,
    kvCacheHitRate: 0,
    trainingLoss: null,
  },

  // ===== Actions =====

  setActiveTab: (tab) => set({ activeTab: tab }),
  setSelectedAgent: (id) => set({ selectedAgent: id }),
  setIsLive: (live) => set({ isLive: live }),
  setTimeOffset: (offset) => set({ timeOffset: offset }),
  toggleCompareMode: () => set(s => ({ compareMode: !s.compareMode })),

  setDataSource: (source) => {
    set({ dataSource: source });
    telemetryService.setDataSource(source);
  },

  connectToAI: async (url: string) => {
    set({ connectionStatus: 'connecting', aiConnectionUrl: url });
    try {
      await telemetryService.connectToAI(url);
      set({ connectionStatus: 'connected' });
    } catch {
      set({ connectionStatus: 'error' });
    }
  },

  disconnectFromAI: async () => {
    await telemetryService.disconnect();
    set({ connectionStatus: 'disconnected', aiConnectionUrl: '' });
  },

  setConnectionStatus: (status) => set({ connectionStatus: status }),

  updateFromSystemMetrics: (metrics: SystemMetrics) => {
    set(s => ({
      realSystemMetrics: metrics,
      systemMetrics: {
        ...s.systemMetrics,
        cpuPercent: metrics.cpu_usage,
        ramPercent: metrics.ram_percent,
        ramUsedGb: metrics.ram_used_gb,
        ramTotalGb: metrics.ram_total_gb,
      },
    }));
  },

  updateFromAIMetrics: (metrics: NexoraAIMetrics) => {
    if (!metrics.health) return;
    const h = metrics.health;
    set(s => ({
      realAIMetrics: metrics,
      systemMetrics: {
        ...s.systemMetrics,
        cpuPercent: h.cpu_usage_percent > 0 ? h.cpu_usage_percent : s.systemMetrics.cpuPercent,
        activeAgents: h.active_connections > 0 ? h.active_connections : s.systemMetrics.activeAgents,
      },
    }));
  },

  updateFromAggregatedTelemetry: (agg: AggregatedTelemetry) => {
    const s = get();

    // Convert live agents
    const liveAgents = agg.agents.length > 0
      ? get().convertAgentTelemetry(agg.agents)
      : s.agents;

    // Convert live memory nodes
    const liveMemory = agg.memory_nodes.length > 0
      ? get().convertMemoryTelemetry(agg.memory_nodes)
      : s.memoryNodes;

    // Convert live models
    const liveModels = agg.models.length > 0
      ? get().convertModelTelemetry(agg.models)
      : s.models;

    // Convert live token flows
    const liveFlows = agg.token_flows.length > 0
      ? get().convertTokenFlowTelemetry(agg.token_flows)
      : s.tokenFlows;

    // Calculate derived metrics
    const avgEntropy = agg.agents.length > 0
      ? agg.agents.reduce((acc, a) => acc + a.entropy, 0) / agg.agents.length
      : s.systemMetrics.avgEntropy;

    const avgReasoning = agg.agents.length > 0
      ? agg.agents.reduce((acc, a) => acc + a.reasoning_depth, 0) / agg.agents.length
      : s.systemMetrics.reasoningDepthAvg;

    const fracIndex = agg.memory_summary?.fragmentation_index ?? s.systemMetrics.memoryFragmentation;

    const halRate = agg.hallucinations?.hallucination_rate ?? s.systemMetrics.hallucinationRate;

    const tokenRate = agg.inference?.tokens_per_second ?? s.systemMetrics.totalTokensPerSec;

    const cacheHit = agg.inference?.cache_hit_rate ?? s.systemMetrics.kvCacheHitRate;

    const latencyMs = agg.inference?.latency_ms ?? s.systemMetrics.inferenceLatencyMs;

    const trainingLoss = agg.training?.loss ?? s.systemMetrics.trainingLoss;

    const activeCount = agg.agents.filter(a => a.status === 'active').length;
    const agentCount = agg.agents.length > 0 ? agg.agents.length : s.systemMetrics.totalAgents;

    set({
      realAggregated: agg,
      realAgents: agg.agents,
      realMemoryNodes: agg.memory_nodes,
      realMemorySummary: agg.memory_summary,
      realInference: agg.inference,
      realHallucinations: agg.hallucinations,
      realTraining: agg.training,
      realModels: agg.models,
      realTokenFlows: agg.token_flows,
      realPipelines: agg.pipelines,
      agents: liveAgents,
      memoryNodes: liveMemory,
      models: liveModels,
      tokenFlows: liveFlows,
      systemMetrics: {
        ...s.systemMetrics,
        avgEntropy,
        reasoningDepthAvg: avgReasoning,
        memoryFragmentation: fracIndex,
        hallucinationRate: halRate,
        totalTokensPerSec: tokenRate,
        kvCacheHitRate: cacheHit,
        inferenceLatencyMs: latencyMs,
        trainingLoss,
        activeAgents: activeCount > 0 ? activeCount : s.systemMetrics.activeAgents,
        totalAgents: agentCount,
      },
    });
  },

  convertAgentTelemetry: (agents: AgentTelemetry[]): Agent[] => {
    // Layout agents in a circle by cluster
    const clusterCounts: Record<string, number> = {};
    const clusterIndexes: Record<string, number> = {};
    for (const a of agents) {
      clusterCounts[a.cluster] = (clusterCounts[a.cluster] || 0) + 1;
    }
    const clusterOrder = Object.keys(clusterCounts).sort();
    const clusterAngles: Record<string, number> = {};
    clusterOrder.forEach((c, i) => {
      clusterAngles[c] = (i / clusterOrder.length) * Math.PI * 2;
    });

    return agents.map((a, idx) => {
      const cIdx = (clusterIndexes[a.cluster] ?? 0);
      clusterIndexes[a.cluster] = cIdx + 1;
      const total = clusterCounts[a.cluster] || 1;
      const angle = clusterAngles[a.cluster] ?? 0;
      const spread = (cIdx / total) * Math.PI * 2;
      const radius = 150 + (Object.keys(clusterCounts).indexOf(a.cluster) * 60);
      const x = 400 + Math.cos(angle + spread) * radius;
      const y = 400 + Math.sin(angle + spread) * radius;

      return {
        id: a.agent_id,
        name: a.agent_name,
        role: a.role,
        status: a.status as AgentStatus,
        model: a.agent_name,
        tokensPerSec: a.tokens_consumed / Math.max(1, a.uptime_seconds),
        entropy: a.entropy,
        memoryUsage: a.memory_usage,
        cpuLoad: a.cpu_load,
        gpuLoad: a.gpu_load,
        hallucinations: a.hallucinations,
        reasoningDepth: a.reasoning_depth,
        x, y,
        cluster: a.cluster,
        connections: [],
        thoughtsPending: a.thoughts_pending,
        uptime: a.uptime_seconds,
        lastEvent: a.loop_detected ? 'loop_detected' : 'active',
      };
    });
  },

  convertMemoryTelemetry: (nodes: MemoryTelemetry[]): MemoryNode[] => {
    const layoutNodes = nodes.map((n, idx) => {
      const angle = (idx / nodes.length) * Math.PI * 2;
      const clusterIdx = ['cluster-1', 'cluster-2', 'cluster-3', 'cluster-4', 'cluster-5']
        .indexOf(n.cluster);
      const radius = 100 + Math.max(0, clusterIdx) * 50;
      return {
        id: n.node_id,
        label: n.label,
        type: n.memory_type as MemoryNode['type'],
        strength: n.strength,
        stability: n.stability,
        accessCount: n.access_count,
        lastAccess: n.last_access,
        connections: n.connections,
        x: 400 + Math.cos(angle) * radius,
        y: 400 + Math.sin(angle) * radius,
        cluster: n.cluster,
      };
    });
    return layoutNodes;
  },

  convertModelTelemetry: (models: ModelTelemetry[]): AIModel[] => {
    const typeMap: Record<string, ModelType> = {
      llm: 'llm', vision: 'vision', embedding: 'embedding',
      reasoning: 'reasoning', validator: 'validator',
    };
    return models.map(m => ({
      id: m.model_id,
      name: m.model_name,
      type: typeMap[m.model_type] || 'llm',
      provider: m.provider,
      gpuMemory: m.gpu_memory_mb,
      gpuMemoryTotal: m.gpu_memory_total_mb,
      throughput: m.throughput_tokens_per_sec,
      latency: m.latency_ms,
      accuracy: m.accuracy,
      activeAgents: m.active_agents,
      temperature: m.temperature,
      quantization: m.quantization,
    }));
  },

  convertTokenFlowTelemetry: (flows: TokenFlowTelemetry[]): TokenFlow[] => {
    return flows.map(f => ({
      source: f.source,
      target: f.target,
      volume: f.volume,
      efficiency: f.efficiency,
      bottleneck: f.is_bottleneck,
    }));
  },

  tickSimulation: () => {
    const state = get();
    if (!state.isLive) return;

    // If connected to live data with real agents, skip mock simulation
    if (state.dataSource === 'live' && state.connectionStatus === 'connected' && state.realAgents.length > 0) {
      return;
    }

    set(s => {
      const agents = s.agents.map(a => ({
        ...a,
        tokensPerSec: Math.max(10, a.tokensPerSec + (Math.random() - 0.5) * 20),
        entropy: Math.max(0, Math.min(1, a.entropy + (Math.random() - 0.48) * 0.02)),
        gpuLoad: Math.max(0, Math.min(100, a.gpuLoad + (Math.random() - 0.5) * 5)),
        cpuLoad: Math.max(0, Math.min(100, a.cpuLoad + (Math.random() - 0.5) * 4)),
        thoughtsPending: Math.max(0, a.thoughtsPending + Math.floor((Math.random() - 0.5) * 3)),
      }));

      const now = Date.now();
      const snap: TimelineSnapshot = {
        timestamp: now,
        label: new Date(now).toLocaleTimeString(),
        hallucinationRate: 0.04 + Math.random() * 0.03,
        reasoningDepth: 5 + Math.random() * 3,
        memoryFragmentation: 0.22 + Math.random() * 0.12,
        activeAgents: Math.floor(40 + Math.random() * 9),
        entropy: 0.25 + Math.random() * 0.2,
        tokenThroughput: 9000 + Math.random() * 3000,
      };

      const avgEntropy = agents.reduce((acc, a) => acc + a.entropy, 0) / agents.length;
      const activeAgents = agents.filter(a => a.status === 'active').length;

      const eventTypes: CognitiveEvent['type'][] = ['reasoning', 'memory', 'planning', 'validation', 'completion'];
      const eventMsgs = ['Multi-hop chain', 'Episodic recall', 'Goal decomposition', 'Fact check OK', 'Task fulfilled'];
      const eIdx = Math.floor(Math.random() * eventTypes.length);
      const newEvent: CognitiveEvent = {
        id: `evt-live-${now}`,
        timestamp: now,
        agentId: `agent-${Math.floor(Math.random() * 49) + 1}`,
        type: eventTypes[eIdx],
        message: eventMsgs[eIdx],
        entropy: Math.random() * 0.6,
        severity: 'info',
      };

      return {
        agents,
        timeline: [...s.timeline.slice(-59), snap],
        cognitiveEvents: [newEvent, ...s.cognitiveEvents.slice(0, 39)],
        systemMetrics: {
          ...s.systemMetrics,
          avgEntropy: Math.round(avgEntropy * 1000) / 1000,
          activeAgents,
          totalTokensPerSec: Math.floor(10000 + Math.random() * 3000),
          natsEventRate: Math.floor(4500 + Math.random() * 800),
          gpuUtilization: 0.65 + Math.random() * 0.2,
          swarmCoherence: Math.max(0.6, Math.min(1, s.systemMetrics.swarmCoherence + (Math.random() - 0.5) * 0.02)),
        },
      };
    });
  },
}));
