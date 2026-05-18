import { useNexoraStore } from '../../store/nexoraStore';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis, ScatterChart, Scatter, Line
} from 'recharts';
import { AlertTriangle, TrendingUp, Brain, Zap, Wifi } from 'lucide-react';

const ENTROPY_LEVELS = [
  { range: '0.0 – 0.3', label: 'STABLE', desc: 'Deterministic reasoning', color: '#10b981', bg: 'bg-emerald-400/10' },
  { range: '0.3 – 0.6', label: 'CREATIVE', desc: 'Divergent exploration', color: '#06b6d4', bg: 'bg-cyan-400/10' },
  { range: '0.6 – 0.8', label: 'VOLATILE', desc: 'Hallucination risk rising', color: '#f59e0b', bg: 'bg-amber-400/10' },
  { range: '0.8 – 1.0', label: 'CHAOTIC', desc: 'Confabulation zone', color: '#ef4444', bg: 'bg-red-400/10' },
];

function entropyColor(e: number) {
  if (e > 0.8) return '#ef4444';
  if (e > 0.6) return '#f59e0b';
  if (e > 0.3) return '#06b6d4';
  return '#10b981';
}

function entropyLabel(e: number) {
  if (e > 0.8) return 'CHAOTIC';
  if (e > 0.6) return 'VOLATILE';
  if (e > 0.3) return 'CREATIVE';
  return 'STABLE';
}

export default function EntropyView() {
  const { agents, timeline, systemMetrics, dataSource, connectionStatus } = useNexoraStore();
  const isLiveData = dataSource === 'live' && connectionStatus === 'connected';

  // Sort agents by entropy
  const byEntropy = [...agents].sort((a, b) => b.entropy - a.entropy);
  const topCritical = byEntropy.slice(0, 8);

  // Timeline entropy data
  const entropyHistory = timeline.slice(-30).map(snap => ({
    time: snap.label,
    entropy: parseFloat((snap.entropy * 100).toFixed(1)),
    hallucination: parseFloat((snap.hallucinationRate * 100).toFixed(2)),
    depth: parseFloat(snap.reasoningDepth.toFixed(1)),
    fragmentation: parseFloat((snap.memoryFragmentation * 100).toFixed(1)),
  }));

  // Radar data per cluster
  const clusters = ['Reasoning', 'Memory', 'Planning', 'Validation', 'Orchestration'];
  const radarData = clusters.map(cluster => {
    const clusterAgents = agents.filter(a => a.cluster === cluster);
    const avg = clusterAgents.reduce((acc, a) => acc + a.entropy, 0) / (clusterAgents.length || 1);
    return {
      cluster: cluster.slice(0, 6),
      entropy: parseFloat((avg * 100).toFixed(1)),
      hallucinations: clusterAgents.reduce((acc, a) => acc + a.hallucinations, 0),
      depth: parseFloat((clusterAgents.reduce((acc, a) => acc + a.reasoningDepth, 0) / (clusterAgents.length || 1)).toFixed(1)),
    };
  });

  // Scatter: entropy vs reasoning depth
  const scatterData = agents.map(a => ({
    x: parseFloat((a.entropy * 100).toFixed(1)),
    y: a.reasoningDepth,
    name: a.name,
    status: a.status,
  }));

  return (
    <div className="w-full h-full flex flex-col overflow-hidden">
      {/* Header metrics */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-800/50 bg-slate-950/50 shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-500 uppercase tracking-widest">Cognitive Entropy Analytics</span>
          {isLiveData && (
            <span className="text-[8px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded-full font-mono border border-emerald-500/30 flex items-center gap-1">
              <Wifi size={8} /> LIVE
            </span>
          )}
        </div>
        <div className="ml-auto flex items-center gap-2">
          {ENTROPY_LEVELS.map(lv => (
            <div key={lv.label} className={`flex items-center gap-1.5 px-2 py-1 rounded ${lv.bg}`}>
              <span className="w-2 h-2 rounded-full" style={{ background: lv.color }} />
              <span className="text-[9px] font-bold" style={{ color: lv.color }}>{lv.label}</span>
              <span className="text-[8px] text-slate-500">{lv.range}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Top row: entropy timeline + agent list */}
        <div className="grid grid-cols-3 gap-4">
          {/* Entropy + Hallucination timeline */}
          <div className="col-span-2 glass rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <TrendingUp size={13} className="text-amber-400" />
                <span className="text-xs font-semibold text-slate-200">Entropy × Hallucination Timeline</span>
              </div>
              <div className="flex items-center gap-3 text-[9px] text-slate-500">
                <span className="flex items-center gap-1"><span className="w-2 h-0.5 bg-amber-400 inline-block" /> Entropy %</span>
                <span className="flex items-center gap-1"><span className="w-2 h-0.5 bg-red-400 inline-block" /> Hallucination %</span>
                <span className="flex items-center gap-1"><span className="w-2 h-0.5 bg-cyan-400 inline-block" /> Reason Depth</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={entropyHistory}>
                <defs>
                  <linearGradient id="entropyGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="halluGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f30" />
                <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} interval={5} />
                <YAxis tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 8, fontSize: 10 }}
                  labelStyle={{ color: '#94a3b8' }}
                />
                <Area type="monotone" dataKey="entropy" stroke="#f59e0b" fill="url(#entropyGrad)" strokeWidth={1.5} dot={false} />
                <Area type="monotone" dataKey="hallucination" stroke="#ef4444" fill="url(#halluGrad)" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="depth" stroke="#06b6d4" strokeWidth={1.5} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Critical agents */}
          <div className="glass rounded-xl p-4 flex flex-col">
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle size={13} className="text-red-400" />
              <span className="text-xs font-semibold text-slate-200">High Entropy Agents</span>
            </div>
            <div className="flex-1 overflow-y-auto space-y-1.5">
              {topCritical.map(agent => (
                <div key={agent.id} className="flex items-center gap-2 px-2 py-1.5 rounded bg-slate-900/60 border border-slate-800/50">
                  <div>
                    <div className="text-[10px] font-mono text-slate-200">{agent.name}</div>
                    <div className="text-[8px] text-slate-500">{agent.cluster}</div>
                  </div>
                  <div className="ml-auto text-right">
                    <div className="text-[10px] font-bold font-mono" style={{ color: entropyColor(agent.entropy) }}>
                      {(agent.entropy * 100).toFixed(1)}%
                    </div>
                    <div className="text-[8px]" style={{ color: entropyColor(agent.entropy) }}>
                      {entropyLabel(agent.entropy)}
                    </div>
                  </div>
                  <div className="w-12 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{ width: `${agent.entropy * 100}%`, background: entropyColor(agent.entropy) }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom row: Scatter + Radar + Distribution */}
        <div className="grid grid-cols-3 gap-4">
          {/* Entropy vs Depth scatter */}
          <div className="glass rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <Brain size={13} className="text-violet-400" />
              <span className="text-xs font-semibold text-slate-200">Entropy vs Reasoning Depth</span>
            </div>
            <ResponsiveContainer width="100%" height={160}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f30" />
                <XAxis dataKey="x" name="Entropy %" unit="%" tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} label={{ value: 'Entropy %', position: 'insideBottom', fill: '#475569', fontSize: 8, dy: 8 }} />
                <YAxis dataKey="y" name="Depth" tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} axisLine={false} label={{ value: 'Depth', angle: -90, position: 'insideLeft', fill: '#475569', fontSize: 8 }} />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3', stroke: '#1e3a5f' }}
                  contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 8, fontSize: 10 }}
                  formatter={(value, name) => [value, name]}
                />
                <Scatter data={scatterData} fill="#06b6d4" opacity={0.7} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Cluster Radar */}
          <div className="glass rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <Zap size={13} className="text-cyan-400" />
              <span className="text-xs font-semibold text-slate-200">Cluster Entropy Radar</span>
            </div>
            <ResponsiveContainer width="100%" height={160}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#1e3a5f50" />
                <PolarAngleAxis dataKey="cluster" tick={{ fill: '#475569', fontSize: 8 }} />
                <Radar name="Entropy" dataKey="entropy" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.2} strokeWidth={1.5} />
                <Radar name="Depth" dataKey="depth" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.15} strokeWidth={1.5} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 8, fontSize: 10 }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Distribution bars */}
          <div className="glass rounded-xl p-4">
            <div className="text-xs font-semibold text-slate-200 mb-3">Agent Distribution</div>
            <div className="space-y-2">
              {ENTROPY_LEVELS.map(lv => {
                const [min, max] = lv.range.split(' – ').map(parseFloat);
                const count = agents.filter(a => a.entropy >= min && a.entropy < max).length;
                const pct = (count / agents.length) * 100;
                return (
                  <div key={lv.label}>
                    <div className="flex justify-between items-center mb-0.5">
                      <div className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full" style={{ background: lv.color }} />
                        <span className="text-[10px] font-bold" style={{ color: lv.color }}>{lv.label}</span>
                        <span className="text-[9px] text-slate-500">{lv.desc}</span>
                      </div>
                      <span className="text-[10px] font-mono text-slate-300">{count} agents</span>
                    </div>
                    <div className="h-2.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{ width: `${pct}%`, background: lv.color }}
                      />
                    </div>
                    <div className="text-[8px] text-slate-600 mt-0.5">{pct.toFixed(0)}% of swarm</div>
                  </div>
                );
              })}
            </div>

            {/* Global entropy indicator */}
            <div className="mt-4 pt-3 border-t border-slate-800/50">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[9px] text-slate-500">Swarm Entropy Index</span>
                <span className="text-xs font-bold font-mono" style={{ color: entropyColor(systemMetrics.avgEntropy) }}>
                  {(systemMetrics.avgEntropy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="relative h-3 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="absolute inset-y-0 left-0 rounded-full"
                  style={{
                    width: `${systemMetrics.avgEntropy * 100}%`,
                    background: `linear-gradient(90deg, #10b981, #06b6d4, #f59e0b, #ef4444)`,
                  }}
                />
                <div
                  className="absolute inset-y-0 w-0.5 bg-white"
                  style={{ left: `${systemMetrics.avgEntropy * 100}%` }}
                />
              </div>
              <div className="flex justify-between text-[8px] text-slate-600 mt-0.5">
                <span>Stable</span>
                <span>Creative</span>
                <span>Volatile</span>
                <span>Chaotic</span>
              </div>
            </div>
          </div>
        </div>

        {/* Memory fragmentation timeline */}
        <div className="glass rounded-xl p-4">
          <div className="text-xs font-semibold text-slate-200 mb-3">Memory Fragmentation vs Token Throughput</div>
          <ResponsiveContainer width="100%" height={100}>
            <AreaChart data={entropyHistory}>
              <defs>
                <linearGradient id="fragGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f20" />
              <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} interval={8} />
              <YAxis tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 8, fontSize: 10 }} />
              <Area type="monotone" dataKey="fragmentation" stroke="#8b5cf6" fill="url(#fragGrad)" strokeWidth={1.5} dot={false} name="Mem Fragmentation %" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
