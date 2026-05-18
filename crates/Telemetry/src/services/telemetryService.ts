import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';

// ===== System Metrics =====
export interface SystemMetrics {
  cpu_usage: number;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_percent: number;
  disk_used_gb: number;
  disk_total_gb: number;
  disk_read_bytes: number;
  disk_write_bytes: number;
  network_rx_bytes: number;
  network_tx_bytes: number;
  processes: number;
  uptime_secs: number;
  cpu_cores: number;
  cpu_per_core: number[];
  gpu_usage: number | null;
  gpu_vram_used_gb: number | null;
  gpu_vram_total_gb: number | null;
}

// ===== AI Health =====
export interface ComponentHealth {
  name: string;
  status: string;
  message: string;
}

export interface AiHealthTelemetry {
  healthy: boolean;
  uptime_seconds: number;
  version: string;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  average_response_time_ms: number;
  requests_per_second: number;
  active_connections: number;
  memory_usage_mb: number;
  cpu_usage_percent: number;
  thread_count: number;
  error_rate_percent: number;
  active_models: string[];
  component_health: ComponentHealth[];
}

export interface NexoraAIMetrics {
  connected: boolean;
  url: string;
  health: AiHealthTelemetry | null;
  error: string | null;
}

// ===== Inference Telemetry =====
export interface InferenceTelemetry {
  model_name: string;
  tokens_per_second: number;
  latency_ms: number;
  latency_p50_ms: number;
  latency_p99_ms: number;
  context_length: number;
  batch_size: number;
  cache_hit_rate: number;
  cache_total_entries: number;
  cache_memory_bytes: number;
  active_requests: number;
  total_requests: number;
  error_rate: number;
  speculative_acceptance_rate: number | null;
  kv_cache_usage_pct: number | null;
}

// ===== Agent Telemetry =====
export interface AgentTelemetry {
  agent_id: string;
  agent_name: string;
  status: string;
  role: string;
  entropy: number;
  reasoning_depth: number;
  tokens_consumed: number;
  tool_calls: number;
  memory_hits: number;
  memory_usage: number;
  uptime_seconds: number;
  hallucinations: number;
  loop_detected: boolean;
  cluster: string;
  cpu_load: number;
  gpu_load: number;
  thoughts_pending: number;
}

// ===== Memory Telemetry =====
export interface MemoryTelemetry {
  node_id: string;
  label: string;
  memory_type: string;
  strength: number;
  stability: number;
  access_count: number;
  last_access: number;
  connections: string[];
  cluster: string;
}

export interface MemorySummary {
  total_nodes: number;
  episodic_count: number;
  semantic_count: number;
  procedural_count: number;
  working_count: number;
  avg_strength: number;
  avg_stability: number;
  fragmentation_index: number;
  high_strength_nodes: number;
  unstable_nodes: number;
}

export interface MemoryResponse {
  nodes: MemoryTelemetry[];
  summary: MemorySummary | null;
}

// ===== Pipeline Telemetry =====
export interface FilterBreakdown {
  filter_name: string;
  rejected: number;
  percent: number;
}

export interface PipelineTelemetry {
  pipeline_name: string;
  samples_loaded: number;
  samples_filtered: number;
  samples_accepted: number;
  total_latency_ms: number;
  throughput_per_sec: number;
  filter_breakdown: FilterBreakdown[];
  backpressure_level: number;
}

// ===== Hallucination Telemetry =====
export interface HallucinationTelemetry {
  total_checked: number;
  total_blocked: number;
  total_flagged: number;
  hallucination_rate: number;
  risk_score_avg: number;
  pre_gen_blocked: number;
  in_gen_corrected: number;
  post_gen_flagged: number;
  top_risk_sources: [string, number][];
}

// ===== Training Telemetry =====
export interface TrainingTelemetry {
  is_training: boolean;
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  loss: number | null;
  learning_rate: number | null;
  grad_norm: number | null;
  tokens_per_second: number | null;
  samples_processed: number;
  time_elapsed_secs: number;
  estimated_time_remaining_secs: number | null;
}

// ===== Model Telemetry =====
export interface ModelTelemetry {
  model_id: string;
  model_name: string;
  model_type: string;
  provider: string;
  status: string;
  gpu_memory_mb: number;
  gpu_memory_total_mb: number;
  throughput_tokens_per_sec: number;
  latency_ms: number;
  accuracy: number;
  active_agents: number;
  quantization: string;
  temperature: number;
}

// ===== Token Flow Telemetry =====
export interface TokenFlowTelemetry {
  source: string;
  target: string;
  volume: number;
  efficiency: number;
  is_bottleneck: boolean;
  latency_ms: number;
}

// ===== Aggregated Telemetry =====
export interface AggregatedTelemetry {
  system: SystemMetrics | null;
  ai_health: AiHealthTelemetry | null;
  inference: InferenceTelemetry | null;
  agents: AgentTelemetry[];
  memory_nodes: MemoryTelemetry[];
  memory_summary: MemorySummary | null;
  pipelines: PipelineTelemetry[];
  hallucinations: HallucinationTelemetry | null;
  training: TrainingTelemetry | null;
  models: ModelTelemetry[];
  token_flows: TokenFlowTelemetry[];
}

// ===== Event types =====
export interface TelemetryEvent {
  timestamp: number;
  source: string;
  event_type: string;
  value: number;
  severity: string;
  label: string;
}

export interface TelemetrySummary {
  cpu_usage: number;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_percent: number;
  cpu_cores: number;
  processes: number;
  uptime_secs: number;
  timestamp: number;
}

export type DataSource = 'mock' | 'live' | 'auto';

export interface ConnectionStatus {
  connected: boolean;
  url: string;
  system_metrics: boolean;
  ai_metrics: boolean;
}

export interface TelemetryState {
  dataSource: DataSource;
  connectionStatus: ConnectionStatus;
  systemMetrics: SystemMetrics | null;
  aiMetrics: NexoraAIMetrics | null;
  aiHealth: AiHealthTelemetry | null;
  inference: InferenceTelemetry | null;
  agents: AgentTelemetry[];
  memoryNodes: MemoryTelemetry[];
  memorySummary: MemorySummary | null;
  pipelines: PipelineTelemetry[];
  hallucinations: HallucinationTelemetry | null;
  training: TrainingTelemetry | null;
  models: ModelTelemetry[];
  tokenFlows: TokenFlowTelemetry[];
  systemHistory: SystemMetrics[];
  telemetryEvents: TelemetryEvent[];
  lastSummary: TelemetrySummary | null;
  aggregated: AggregatedTelemetry | null;
  error: string | null;
}

type TelemetryListener = (state: Partial<TelemetryState>) => void;

class TelemetryService {
  private state: TelemetryState = {
    dataSource: 'auto',
    connectionStatus: { connected: false, url: '', system_metrics: false, ai_metrics: false },
    systemMetrics: null,
    aiMetrics: null,
    aiHealth: null,
    inference: null,
    agents: [],
    memoryNodes: [],
    memorySummary: null,
    pipelines: [],
    hallucinations: null,
    training: null,
    models: [],
    tokenFlows: [],
    systemHistory: [],
    telemetryEvents: [],
    lastSummary: null,
    aggregated: null,
    error: null,
  };

  private listeners: Set<TelemetryListener> = new Set();
  private unlistenFns: Array<() => void> = [];
  private pollInterval: ReturnType<typeof setInterval> | null = null;
  private aiPollInterval: ReturnType<typeof setInterval> | null = null;
  private isRunning = false;

  subscribe(listener: TelemetryListener): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  private notify(update: Partial<TelemetryState>) {
    this.state = { ...this.state, ...update };
    this.listeners.forEach(l => l(update));
  }

  getState(): TelemetryState {
    return this.state;
  }

  async start(url?: string) {
    if (this.isRunning) return;
    this.isRunning = true;

    // Listen for Tauri events
    try {
      const unlistenSystem = await listen<TelemetryEvent>('telemetry:system', (event) => {
        const events = [event.payload, ...this.state.telemetryEvents].slice(0, 100);
        this.notify({ telemetryEvents: events });
      });
      this.unlistenFns.push(unlistenSystem);

      const unlistenSummary = await listen<TelemetrySummary>('telemetry:summary', (event) => {
        this.notify({ lastSummary: event.payload });
      });
      this.unlistenFns.push(unlistenSummary);
    } catch (e) {
      console.log('Tauri events not available, using mock fallback');
    }

    // Poll system metrics every 2s
    this.pollInterval = setInterval(async () => {
      try {
        const metrics = await invoke<SystemMetrics>('get_system_metrics');
        const history = [...this.state.systemHistory.slice(-59), metrics];
        this.notify({ systemMetrics: metrics, systemHistory: history, error: null });
      } catch (e) {
        this.notify({ error: `System metrics error: ${e}` });
      }
    }, 2000);

    // Poll all AI telemetry every 5s when connected
    this.aiPollInterval = setInterval(async () => {
      if (this.state.connectionStatus.connected) {
        await this.pollAllTelemetry();
      }
    }, 5000);

    // Connect if URL provided
    if (url) {
      await this.connectToAI(url);
    }
  }

  private async pollAllTelemetry() {
    try {
      // Single aggregated call is most efficient
      const aggregated = await invoke<AggregatedTelemetry>('get_aggregated_telemetry');
      this.notify({
        aggregated,
        aiHealth: aggregated.ai_health,
        inference: aggregated.inference,
        agents: aggregated.agents,
        memoryNodes: aggregated.memory_nodes,
        memorySummary: aggregated.memory_summary,
        pipelines: aggregated.pipelines,
        hallucinations: aggregated.hallucinations,
        training: aggregated.training,
        models: aggregated.models,
        tokenFlows: aggregated.token_flows,
        dataSource: 'live',
      });
    } catch (_e) {
      // Fallback: fetch individual endpoints
      await this.pollIndividualTelemetry();
    }
  }

  private async pollIndividualTelemetry() {
    try {
      const [health, inference, agents, memory, pipelines, hallucinations, training, models, tokenFlows] =
        await Promise.all([
          invoke<AiHealthTelemetry | null>('get_ai_health').catch(() => null),
          invoke<InferenceTelemetry | null>('get_inference_telemetry').catch(() => null),
          invoke<AgentTelemetry[]>('get_agent_telemetry').catch(() => []),
          invoke<MemoryResponse>('get_memory_telemetry').catch(() => ({ nodes: [], summary: null })),
          invoke<PipelineTelemetry[]>('get_pipeline_telemetry').catch(() => []),
          invoke<HallucinationTelemetry | null>('get_hallucination_telemetry').catch(() => null),
          invoke<TrainingTelemetry | null>('get_training_telemetry').catch(() => null),
          invoke<ModelTelemetry[]>('get_model_telemetry').catch(() => []),
          invoke<TokenFlowTelemetry[]>('get_token_flow_telemetry').catch(() => []),
        ]);
      this.notify({
        aiHealth: health,
        inference,
        agents,
        memoryNodes: memory.nodes,
        memorySummary: memory.summary,
        pipelines,
        hallucinations,
        training,
        models,
        tokenFlows,
        dataSource: 'live',
      });
    } catch { /* silent */ }
  }

  async connectToAI(url: string) {
    try {
      const result = await invoke<NexoraAIMetrics>('connect_nexora_ai', { url });
      this.notify({
        aiMetrics: result,
        aiHealth: result.health,
        connectionStatus: {
          connected: result.connected,
          url: result.url,
          system_metrics: true,
          ai_metrics: result.connected,
        },
        dataSource: result.connected ? 'live' : 'mock',
      });

      // Immediately fetch all telemetry on connect
      if (result.connected) {
        await this.pollAllTelemetry();
      }
    } catch (e) {
      this.notify({
        error: `Failed to connect: ${e}`,
        connectionStatus: { connected: false, url, system_metrics: true, ai_metrics: false },
      });
    }
  }

  async disconnect() {
    try { await invoke('disconnect_nexora_ai'); } catch { /* ignore */ }
    this.notify({
      aiMetrics: null, aiHealth: null,
      inference: null, agents: [], memoryNodes: [], memorySummary: null,
      pipelines: [], hallucinations: null, training: null,
      models: [], tokenFlows: [], aggregated: null,
      connectionStatus: { connected: false, url: '', system_metrics: true, ai_metrics: false },
      dataSource: 'mock',
    });
  }

  async checkConnection(): Promise<ConnectionStatus> {
    try {
      return await invoke<ConnectionStatus>('get_connection_status');
    } catch {
      return { connected: false, url: '', system_metrics: false, ai_metrics: false };
    }
  }

  setDataSource(source: DataSource) {
    this.notify({ dataSource: source });
  }

  stop() {
    this.isRunning = false;
    this.unlistenFns.forEach(fn => fn());
    this.unlistenFns = [];
    if (this.pollInterval) { clearInterval(this.pollInterval); this.pollInterval = null; }
    if (this.aiPollInterval) { clearInterval(this.aiPollInterval); this.aiPollInterval = null; }
  }
}

export const telemetryService = new TelemetryService();
