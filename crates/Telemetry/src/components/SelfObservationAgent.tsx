import { useState, useEffect } from 'react';
import { useNexoraStore } from '../store/nexoraStore';
import { Eye, Brain, AlertTriangle, ChevronDown, ChevronUp, Zap, Shield } from 'lucide-react';

interface Anomaly {
  id: string;
  agent: string;
  type: string;
  severity: 'low' | 'medium' | 'high';
  message: string;
  entropy: number;
  timestamp: number;
}

function generateAnomaly(agents: ReturnType<typeof useNexoraStore.getState>['agents']): Anomaly | null {
  const highEntropyAgents = agents.filter(a => a.entropy > 0.6);
  if (highEntropyAgents.length === 0) return null;
  const agent = highEntropyAgents[Math.floor(Math.random() * highEntropyAgents.length)];
  const types = [
    { type: 'reasoning_loop', msg: 'Circular reasoning detected in chain' },
    { type: 'memory_drift', msg: 'Context window fragmentation above threshold' },
    { type: 'confabulation', msg: 'Ungrounded assertion without source' },
    { type: 'entropy_spike', msg: 'Sudden entropy increase > 0.3 in 500ms' },
    { type: 'consensus_fail', msg: 'Multi-agent validation failed 3 times' },
  ];
  const t = types[Math.floor(Math.random() * types.length)];
  return {
    id: `anom-${Date.now()}`,
    agent: agent.name,
    type: t.type,
    severity: agent.entropy > 0.8 ? 'high' : agent.entropy > 0.65 ? 'medium' : 'low',
    message: t.msg,
    entropy: agent.entropy,
    timestamp: Date.now(),
  };
}

const SEVERITY_COLORS = { low: '#10b981', medium: '#f59e0b', high: '#ef4444' };

export default function SelfObservationAgent() {
  const [collapsed, setCollapsed] = useState(false);
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [thinking, setThinking] = useState('Monitoring swarm cognitive state...');
  const { agents, systemMetrics } = useNexoraStore();

  const thoughts = [
    'Analyzing entropy distribution across clusters...',
    'Cross-referencing hallucination patterns with memory access logs...',
    'Detecting reasoning topology shifts in Validation cluster...',
    'Monitoring token flow for dead-loop signatures...',
    'Evaluating swarm coherence delta over last 60 snapshots...',
    'Scanning for semantic terrain instability...',
    'Checking inter-agent consensus failure rates...',
    'Analyzing cognitive entropy gradient across timeline...',
  ];

  useEffect(() => {
    const anomalyInterval = setInterval(() => {
      const a = generateAnomaly(agents);
      if (a) {
        setAnomalies(prev => [a, ...prev.slice(0, 4)]);
      }
    }, 4500);

    const thoughtInterval = setInterval(() => {
      setThinking(thoughts[Math.floor(Math.random() * thoughts.length)]);
    }, 3000);

    return () => {
      clearInterval(anomalyInterval);
      clearInterval(thoughtInterval);
    };
  }, [agents]);

  const criticalCount = anomalies.filter(a => a.severity === 'high').length;

  return (
    <div className="fixed bottom-4 left-56 w-80 z-40">
      {/* Header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center gap-2 px-3 py-2 glass-dark rounded-t-xl border border-violet-500/30 hover:border-violet-400/50 transition-colors"
      >
        <div className="relative">
          <Eye size={14} className="text-violet-400" />
          <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-violet-400 rounded-full animate-pulse" />
        </div>
        <span className="text-xs font-bold text-violet-300 font-mono">SELF-OBSERVATION AGENT</span>
        {criticalCount > 0 && (
          <span className="ml-auto flex items-center gap-1 text-[9px] text-red-400 bg-red-400/10 px-1.5 py-0.5 rounded-full">
            <AlertTriangle size={9} />
            {criticalCount} critical
          </span>
        )}
        <span className="text-slate-500">
          {collapsed ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </span>
      </button>

      {!collapsed && (
        <div className="glass-dark border border-t-0 border-violet-500/20 rounded-b-xl overflow-hidden">
          {/* Thinking indicator */}
          <div className="px-3 py-2 border-b border-slate-800/50 bg-violet-950/20">
            <div className="flex items-center gap-2">
              <Brain size={10} className="text-violet-400 shrink-0 animate-pulse" />
              <span className="text-[9px] text-violet-300 font-mono leading-tight">{thinking}</span>
            </div>
          </div>

          {/* Swarm health summary */}
          <div className="px-3 py-2 border-b border-slate-800/50">
            <div className="grid grid-cols-3 gap-2">
              {[
                { label: 'Entropy', value: `${(systemMetrics.avgEntropy * 100).toFixed(0)}%`, status: systemMetrics.avgEntropy > 0.5 ? 'warn' : 'ok' },
                { label: 'Halluc', value: `${(systemMetrics.hallucinationRate * 100).toFixed(1)}%`, status: systemMetrics.hallucinationRate > 0.05 ? 'warn' : 'ok' },
                { label: 'Coherence', value: `${(systemMetrics.swarmCoherence * 100).toFixed(0)}%`, status: systemMetrics.swarmCoherence < 0.7 ? 'warn' : 'ok' },
              ].map(m => (
                <div key={m.label} className="text-center bg-slate-900/50 rounded p-1.5">
                  <div className="text-[8px] text-slate-500">{m.label}</div>
                  <div className={`text-[11px] font-bold font-mono ${m.status === 'warn' ? 'text-amber-400' : 'text-emerald-400'}`}>{m.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Anomaly feed */}
          <div className="px-3 py-2">
            <div className="flex items-center gap-1 mb-2">
              <Zap size={10} className="text-amber-400" />
              <span className="text-[9px] text-slate-500 uppercase tracking-widest">Detected Anomalies</span>
            </div>
            {anomalies.length === 0 ? (
              <div className="text-[9px] text-slate-600 text-center py-2">No anomalies detected</div>
            ) : (
              <div className="space-y-1.5">
                {anomalies.map(a => (
                  <div key={a.id} className="flex items-start gap-2 px-2 py-1.5 rounded bg-slate-900/60 border border-slate-800/50">
                    <div className="w-1.5 h-1.5 rounded-full mt-1 shrink-0" style={{ background: SEVERITY_COLORS[a.severity] }} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1">
                        <span className="text-[9px] font-mono font-bold" style={{ color: SEVERITY_COLORS[a.severity] }}>
                          {a.type.replace(/_/g, ' ').toUpperCase()}
                        </span>
                      </div>
                      <div className="text-[9px] text-slate-400 truncate">{a.message}</div>
                      <div className="text-[8px] text-slate-600">{a.agent} · {new Date(a.timestamp).toLocaleTimeString()}</div>
                    </div>
                    <div className="text-[9px] font-mono shrink-0" style={{ color: SEVERITY_COLORS[a.severity] }}>
                      {(a.entropy * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Isolation warning */}
          <div className="px-3 pb-2">
            <div className="flex items-center gap-1.5 px-2 py-1.5 rounded bg-slate-900/60 border border-amber-500/20">
              <Shield size={10} className="text-amber-400" />
              <span className="text-[8px] text-slate-500">Autonomous shutdown authority: </span>
              <span className="text-[8px] text-amber-400 font-bold">DISABLED</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
